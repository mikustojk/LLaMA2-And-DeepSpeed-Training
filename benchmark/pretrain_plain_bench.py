from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
TARGET_GLOBAL_BATCH_SIZE = 8

from LLaMA2 import ModelConfig, Transformer  # noqa: E402
from Pretrain_Dataset import PretrainDataset  # noqa: E402
from benchmark_utils import (  # noqa: E402
    benchmark_metrics,
    cuda_memory_gb,
    infinite_batches,
    masked_loss,
    move_batch,
    now,
    print_and_save_metrics,
    resolve_project_path,
    seed_everything,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单卡原生 PyTorch 预训练吞吐和显存 benchmark"
    )
    parser.add_argument("--data_path", default="data/seq_monkey/test_1000.jsonl")
    parser.add_argument("--tokenizer_path", default="tokenizer_k")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=18)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--n_kv_heads", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--measure_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--result_path", default="benchmark/results/plain_pretrain.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark 需要 CUDA GPU。")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU/PyTorch 不支持 BF16，无法进行统一的 BF16 benchmark。")
    if args.measure_steps < 1 or args.warmup_steps < 0:
        raise ValueError("measure_steps 必须大于0，warmup_steps 不能小于0。")
    if args.micro_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("micro_batch_size 和 gradient_accumulation_steps 必须大于0。")
    global_batch_size = (
        args.micro_batch_size * args.gradient_accumulation_steps
    )
    if global_batch_size != TARGET_GLOBAL_BATCH_SIZE:
        raise ValueError(
            "当前单卡 benchmark 固定 global batch size 为 "
            f"{TARGET_GLOBAL_BATCH_SIZE}，但 micro_batch_size="
            f"{args.micro_batch_size} 与 gradient_accumulation_steps="
            f"{args.gradient_accumulation_steps} 的乘积为 {global_batch_size}。"
        )
    if args.max_length < 2:
        raise ValueError("max_length 必须至少为2。")
    if args.dim < 1 or args.n_layers < 1 or args.n_heads < 1 or args.n_kv_heads < 1:
        raise ValueError("模型维度、层数和注意力头数必须大于0。")
    if args.dim % args.n_heads != 0:
        raise ValueError("dim 必须能被 n_heads 整除。")
    if args.n_heads % args.n_kv_heads != 0:
        raise ValueError("n_heads 必须能被 n_kv_heads 整除。")
    if args.gradient_clipping < 0:
        raise ValueError("gradient_clipping 不能小于0。")
    print(
        "Effective batch config: "
        f"micro_batch_size={args.micro_batch_size}, "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
        f"global_batch_size={global_batch_size}",
        flush=True,
    )

    seed_everything(args.seed)
    device = torch.device("cuda")
    tokenizer_path = resolve_project_path(args.tokenizer_path)
    data_path = resolve_project_path(args.data_path)
    print(f"Loading tokenizer from: {tokenizer_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path)
    )
    if len(tokenizer) != 6144:
        raise ValueError(
            f"Tokenizer vocab size must be 6144, got {len(tokenizer)}: {tokenizer_path}"
        )
    print(f"Loading dataset from: {data_path}", flush=True)
    dataset = PretrainDataset(
        str(data_path),
        tokenizer,
        max_length=args.max_length,
    )
    if len(dataset) < args.micro_batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} samples, fewer than micro_batch_size "
            f"{args.micro_batch_size}."
        )
    print(f"Dataset size: {len(dataset)}", flush=True)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        generator=loader_generator,
    )
    batches = infinite_batches(loader)

    config = ModelConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=len(tokenizer),
        max_seq_len=args.max_length,
        dropout=0.0,
    )
    print(
        f"Model config: dim={config.dim}, layers={config.n_layers}, "
        f"heads={config.n_heads}, kv_heads={config.n_kv_heads}",
        flush=True,
    )
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    model.train()
    accumulation = args.gradient_accumulation_steps

    def run_optimizer_step() -> tuple[torch.Tensor, int]:
        optimizer.zero_grad(set_to_none=True)
        last_loss = None
        valid_tokens = 0

        for _ in range(accumulation):
            batch = next(batches)
            x, y, loss_mask, batch_valid_tokens = move_batch(batch, device)
            valid_tokens += batch_valid_tokens

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(x, targets=y)
                loss = masked_loss(outputs, loss_mask)

            (loss / accumulation).backward()
            last_loss = loss

        if args.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.gradient_clipping,
            )
        optimizer.step()
        assert last_loss is not None
        return last_loss, valid_tokens

    print("开始 Plain PyTorch 预热...")
    for _ in range(args.warmup_steps):
        run_optimizer_step()

    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = now()
    valid_tokens = 0
    last_loss = None
    for _ in range(args.measure_steps):
        last_loss, step_valid_tokens = run_optimizer_step()
        valid_tokens += step_valid_tokens
    synchronize(device)
    elapsed = now() - start

    assert last_loss is not None
    peak_allocated, peak_reserved = cuda_memory_gb(device)
    sequence_length = args.max_length - 1
    metrics = benchmark_metrics(
        elapsed_seconds=elapsed,
        measure_steps=args.measure_steps,
        global_batch_size=global_batch_size,
        sequence_length=sequence_length,
        valid_tokens=valid_tokens,
        last_loss=float(last_loss.detach().cpu()),
        peak_allocated_gb=peak_allocated,
        peak_reserved_gb=peak_reserved,
        method="plain_pytorch",
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        metadata={
            "data_path": str(data_path),
            "tokenizer_path": str(tokenizer_path),
            "model_dim": config.dim,
            "model_n_layers": config.n_layers,
            "model_n_heads": config.n_heads,
            "model_n_kv_heads": config.n_kv_heads,
            "vocab_size": config.vocab_size,
            "precision": "bf16",
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "gradient_clipping": args.gradient_clipping,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gpu_count": 1,
            "attention_implementation": "sdpa_if_available",
            "autocast_controller": "torch_autocast",
        },
    )
    print_and_save_metrics(metrics, args.result_path)


if __name__ == "__main__":
    main()
