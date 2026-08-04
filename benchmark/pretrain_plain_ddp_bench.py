from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
EXPECTED_WORLD_SIZE = 2
TARGET_GLOBAL_BATCH_SIZE = 8

from LLaMA2 import ModelConfig, Transformer  # noqa: E402
from Pretrain_Dataset import PretrainDataset  # noqa: E402
from benchmark_utils import (  # noqa: E402
    aggregate_distributed_measurements,
    benchmark_metrics,
    cuda_memory_gb,
    infinite_batches,
    masked_loss,
    model_parameter_counts,
    move_batch,
    now,
    print_and_save_metrics,
    resolve_project_path,
    seed_everything,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="双卡 PyTorch DDP 预训练吞吐和显存 benchmark"
    )
    parser.add_argument("--data_path", default="data/seq_monkey/test_1000.jsonl")
    parser.add_argument("--tokenizer_path", default="tokenizer_k")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=18)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--n_kv_heads", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--measure_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument(
        "--result_path",
        default="benchmark/results/plain_ddp_2gpu_micro4_accum1_1000.json",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, launched_world_size: int) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark 需要 CUDA GPU。")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU/PyTorch 不支持 BF16，无法进行统一 benchmark。")
    if launched_world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"当前脚本要求 {EXPECTED_WORLD_SIZE} 个进程，实际为 "
            f"{launched_world_size}。请使用 torchrun --nproc_per_node=2。"
        )
    if args.measure_steps < 1 or args.warmup_steps < 0:
        raise ValueError("measure_steps 必须大于0，warmup_steps 不能小于0。")
    if args.micro_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("micro_batch_size 和 gradient_accumulation_steps 必须大于0。")
    global_batch_size = (
        args.micro_batch_size
        * args.gradient_accumulation_steps
        * launched_world_size
    )
    if global_batch_size != TARGET_GLOBAL_BATCH_SIZE:
        raise ValueError(
            f"双卡 benchmark 固定 global batch size 为 {TARGET_GLOBAL_BATCH_SIZE}，"
            f"但 {args.micro_batch_size} × {args.gradient_accumulation_steps} × "
            f"{launched_world_size} = {global_batch_size}。"
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
    return global_batch_size


def main() -> None:
    args = parse_args()
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch_size = validate_args(args, launched_world_size)
    env_local_rank = os.environ.get("LOCAL_RANK")
    local_rank = int(env_local_rank) if env_local_rank is not None else args.local_rank
    if local_rank < 0:
        raise RuntimeError("缺少 LOCAL_RANK，请使用 torchrun 启动双卡脚本。")
    if args.local_rank >= 0 and args.local_rank != local_rank:
        raise RuntimeError(
            f"命令行 local rank 为 {args.local_rank}，环境变量 LOCAL_RANK 为 "
            f"{local_rank}，两者不一致。"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != EXPECTED_WORLD_SIZE:
            raise RuntimeError(
                f"当前脚本要求 world_size={EXPECTED_WORLD_SIZE}，实际为 {world_size}。"
            )

        seed_everything(args.seed)
        tokenizer_path = resolve_project_path(args.tokenizer_path)
        data_path = resolve_project_path(args.data_path)
        if rank == 0:
            print(f"Loading tokenizer from: {tokenizer_path}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        if len(tokenizer) != 6144:
            raise ValueError(
                f"Tokenizer vocab size must be 6144, got {len(tokenizer)}: "
                f"{tokenizer_path}"
            )

        if rank == 0:
            print(f"Loading dataset from: {data_path}", flush=True)
        dataset = PretrainDataset(
            str(data_path),
            tokenizer,
            max_length=args.max_length,
        )
        if len(dataset) < args.micro_batch_size * world_size:
            raise ValueError("数据集样本数不足以组成一个双卡 global batch。")
        if rank == 0:
            print(f"Dataset size: {len(dataset)}", flush=True)

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        sampler.set_epoch(0)
        loader = DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
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
        if rank == 0:
            print(
                f"Model config: dim={config.dim}, layers={config.n_layers}, "
                f"heads={config.n_heads}, kv_heads={config.n_kv_heads}",
                flush=True,
            )
            print(
                "Effective batch config: "
                f"micro_batch_size_per_gpu={args.micro_batch_size}, "
                f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
                f"world_size={world_size}, global_batch_size={global_batch_size}",
                flush=True,
            )

        model = Transformer(config).to(device)
        parameter_count, trainable_parameter_count = model_parameter_counts(model)
        mlp_hidden_dim = model.layers[0].mlp.w1.out_features
        if rank == 0:
            print(
                f"Model parameters: {parameter_count:,} "
                f"({parameter_count / 1_000_000:.1f}M), "
                f"trainable={trainable_parameter_count:,}, "
                f"mlp_hidden_dim={mlp_hidden_dim}",
                flush=True,
            )
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        optimizer = torch.optim.AdamW(
            ddp_model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        ddp_model.train()
        accumulation = args.gradient_accumulation_steps

        def run_optimizer_step() -> tuple[torch.Tensor, int]:
            optimizer.zero_grad(set_to_none=True)
            last_loss = None
            valid_tokens = 0

            for micro_step in range(accumulation):
                batch = next(batches)
                x, y, loss_mask, batch_valid_tokens = move_batch(batch, device)
                valid_tokens += batch_valid_tokens
                should_sync = micro_step == accumulation - 1
                sync_context = nullcontext() if should_sync else ddp_model.no_sync()

                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = ddp_model(x, targets=y)
                        loss = masked_loss(outputs, loss_mask)
                    (loss / accumulation).backward()
                last_loss = loss

            if args.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    ddp_model.parameters(),
                    max_norm=args.gradient_clipping,
                )
            optimizer.step()
            assert last_loss is not None
            return last_loss, valid_tokens

        if rank == 0:
            print("开始双卡 PyTorch DDP 预热...", flush=True)
        for _ in range(args.warmup_steps):
            run_optimizer_step()

        synchronize(device)
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        start = now()
        local_valid_tokens = 0
        last_loss = None
        for _ in range(args.measure_steps):
            last_loss, step_valid_tokens = run_optimizer_step()
            local_valid_tokens += step_valid_tokens
        synchronize(device)
        local_elapsed = now() - start

        assert last_loss is not None
        local_allocated, local_reserved = cuda_memory_gb(device)
        aggregated = aggregate_distributed_measurements(
            device=device,
            elapsed_seconds=local_elapsed,
            valid_tokens=local_valid_tokens,
            last_loss=float(last_loss.detach().cpu()),
            peak_allocated_gb=local_allocated,
            peak_reserved_gb=local_reserved,
        )

        if rank == 0:
            metrics = benchmark_metrics(
                elapsed_seconds=aggregated["elapsed_seconds"],
                measure_steps=args.measure_steps,
                global_batch_size=global_batch_size,
                sequence_length=args.max_length - 1,
                valid_tokens=aggregated["valid_tokens"],
                last_loss=aggregated["last_loss"],
                peak_allocated_gb=aggregated["peak_allocated_gb"],
                peak_reserved_gb=aggregated["peak_reserved_gb"],
                method="plain_pytorch_ddp",
                warmup_steps=args.warmup_steps,
                seed=args.seed,
                metadata={
                    "data_path": str(data_path),
                    "tokenizer_path": str(tokenizer_path),
                    "model_dim": config.dim,
                    "model_n_layers": config.n_layers,
                    "model_n_heads": config.n_heads,
                    "model_n_kv_heads": config.n_kv_heads,
                    "model_mlp_hidden_dim": mlp_hidden_dim,
                    "model_parameter_count": parameter_count,
                    "model_trainable_parameter_count": trainable_parameter_count,
                    "vocab_size": config.vocab_size,
                    "precision": "bf16",
                    "optimizer": "AdamW",
                    "learning_rate": args.learning_rate,
                    "gradient_clipping": args.gradient_clipping,
                    "micro_batch_size": args.micro_batch_size,
                    "gradient_accumulation_steps": accumulation,
                    "gpu_count": world_size,
                    "distributed_backend": dist.get_backend(),
                    "parallelism": "ddp",
                    "per_rank_peak_allocated_gb": aggregated[
                        "per_rank_peak_allocated_gb"
                    ],
                    "per_rank_peak_reserved_gb": aggregated[
                        "per_rank_peak_reserved_gb"
                    ],
                    "attention_implementation": "sdpa_if_available",
                    "autocast_controller": "torch_autocast",
                },
            )
            print_and_save_metrics(metrics, args.result_path)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
