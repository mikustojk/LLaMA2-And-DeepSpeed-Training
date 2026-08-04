from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
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
        description="单卡 DeepSpeed ZeRO-2 预训练吞吐和显存 benchmark"
    )
    parser.add_argument("--data_path", default="data/seq_monkey/test_1000.jsonl")
    parser.add_argument("--tokenizer_path", default="tokenizer_k")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--deepspeed_config", default="benchmark/ds_config_bench.json")
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
    parser.add_argument("--result_path", default="benchmark/results/deepspeed_pretrain.json")
    parser.add_argument("--local_rank", type=int, default=-1)
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
    configured_global_batch_size = (
        args.micro_batch_size * args.gradient_accumulation_steps
    )
    if configured_global_batch_size != TARGET_GLOBAL_BATCH_SIZE:
        raise ValueError(
            "当前单卡 benchmark 固定 global batch size 为 "
            f"{TARGET_GLOBAL_BATCH_SIZE}，但 micro_batch_size="
            f"{args.micro_batch_size} 与 gradient_accumulation_steps="
            f"{args.gradient_accumulation_steps} 的乘积为 "
            f"{configured_global_batch_size}。"
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

    seed_everything(args.seed)
    config_path = resolve_project_path(args.deepspeed_config)
    with config_path.open("r", encoding="utf-8") as f:
        ds_config = json.load(f)
    micro_batch_size = args.micro_batch_size
    accumulation = args.gradient_accumulation_steps
    # 只覆盖本次进程读取到的配置字典，不修改磁盘上的 JSON 文件。
    ds_config["train_batch_size"] = configured_global_batch_size
    ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size
    ds_config["gradient_accumulation_steps"] = accumulation
    bf16_enabled = bool(ds_config.get("bf16", {}).get("enabled", False))
    if not bf16_enabled:
        raise ValueError("benchmark requires bf16.enabled=true in the DeepSpeed config.")
    zero_stage = int(ds_config.get("zero_optimization", {}).get("stage", 0))
    if zero_stage != 2:
        raise ValueError(
            f"This benchmark is for ZeRO-2, but the config uses stage {zero_stage}."
        )
    offload_device = ds_config.get("zero_optimization", {}).get(
        "offload_optimizer", {}
    ).get("device", "none")
    if offload_device != "none":
        raise ValueError(
            "This benchmark requires optimizer offload device 'none' for a fair "
            "single-GPU comparison."
        )
    configured_gradient_clipping = float(ds_config.get("gradient_clipping", 0.0))
    if not math.isclose(
        configured_gradient_clipping,
        args.gradient_clipping,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Plain PyTorch 与 DeepSpeed 的 gradient_clipping 必须一致："
            f"命令参数为 {args.gradient_clipping}，DeepSpeed 配置为 "
            f"{configured_gradient_clipping}。"
        )

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
    if len(dataset) < micro_batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} samples, fewer than micro_batch_size "
            f"{micro_batch_size}."
        )
    print(f"Dataset size: {len(dataset)}", flush=True)

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
    model = Transformer(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        config=ds_config,
    )

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size != 1:
        raise RuntimeError(
            "当前 benchmark 脚本先固定为单卡。双卡对比需要同步调整 global batch size 后再扩展。"
        )
    expected_global_batch_size = micro_batch_size * accumulation * world_size
    if configured_global_batch_size != expected_global_batch_size:
        raise ValueError(
            "DeepSpeed 配置中的 train_batch_size 不匹配："
            f"期望 {expected_global_batch_size}，实际 {configured_global_batch_size}。"
        )
    if dist.get_rank() == 0:
        print(
            "Effective batch config: "
            f"micro_batch_size={micro_batch_size}, "
            f"gradient_accumulation_steps={accumulation}, "
            f"global_batch_size={configured_global_batch_size}",
            flush=True,
        )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        generator=loader_generator,
    )
    batches = infinite_batches(loader)
    device = model_engine.device
    model_engine.train()

    def run_optimizer_step() -> tuple[torch.Tensor, int]:
        last_loss = None
        valid_tokens = 0

        for _ in range(accumulation):
            batch = next(batches)
            x, y, loss_mask, batch_valid_tokens = move_batch(batch, device)
            valid_tokens += batch_valid_tokens

            # DeepSpeed 配置已启用 bf16，由 engine 负责混合精度。
            # 不再在 engine 外层嵌套 torch.autocast，避免重复配置和警告。
            outputs = model_engine(x, targets=y)
            loss = masked_loss(outputs, loss_mask)

            # DeepSpeed 会按照 gradient_accumulation_steps 处理梯度缩放。
            model_engine.backward(loss)
            model_engine.step()
            last_loss = loss

        assert last_loss is not None
        return last_loss, valid_tokens

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print("开始 DeepSpeed 预热...")
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
    metrics = benchmark_metrics(
        elapsed_seconds=elapsed,
        measure_steps=args.measure_steps,
        global_batch_size=configured_global_batch_size,
        sequence_length=args.max_length - 1,
        valid_tokens=valid_tokens,
        last_loss=float(last_loss.detach().cpu()),
        peak_allocated_gb=peak_allocated,
        peak_reserved_gb=peak_reserved,
        method="deepspeed_zero2",
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
            "gradient_clipping": configured_gradient_clipping,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": accumulation,
            "gpu_count": world_size,
            "zero_stage": zero_stage,
            "optimizer_offload": offload_device,
            "deepspeed_config_path": str(config_path),
            "attention_implementation": "sdpa_if_available",
            "autocast_controller": "deepspeed_bf16",
        },
    )
    if rank == 0:
        print_and_save_metrics(metrics, args.result_path)


if __name__ == "__main__":
    main()
