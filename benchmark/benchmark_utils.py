from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple

import numpy as np
import torch
import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def model_parameter_counts(model: torch.nn.Module) -> Tuple[int, int]:
    """返回模型的总参数量和可训练参数量；共享参数只统计一次。"""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def infinite_batches(loader: Iterable[Tuple[torch.Tensor, ...]]) -> Iterator[Tuple[torch.Tensor, ...]]:
    while True:
        for batch in loader:
            yield batch


def masked_loss(outputs: Any, loss_mask: torch.Tensor) -> torch.Tensor:
    """与 pretrain_ds.py 保持一致的逐 token loss 计算。"""
    token_loss = outputs.last_loss.reshape(loss_mask.shape)
    mask = loss_mask.to(device=token_loss.device, dtype=token_loss.dtype)
    return (token_loss * mask).sum() / mask.sum().clamp_min(1.0)


def move_batch(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    x, y, loss_mask = batch
    valid_tokens = int(loss_mask.sum().item())
    return (
        x.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
        loss_mask.to(device, non_blocking=True),
        valid_tokens,
    )


def benchmark_metrics(
    *,
    elapsed_seconds: float,
    measure_steps: int,
    global_batch_size: int,
    sequence_length: int,
    valid_tokens: int,
    last_loss: float,
    peak_allocated_gb: float,
    peak_reserved_gb: float,
    method: str,
    warmup_steps: int,
    seed: int,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be greater than zero.")
    nominal_tokens = measure_steps * global_batch_size * sequence_length
    metrics: Dict[str, Any] = {
        "method": method,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_optimizer_step": elapsed_seconds / measure_steps,
        "global_batch_size": global_batch_size,
        "sequence_length": sequence_length,
        "nominal_tokens": nominal_tokens,
        "nominal_tokens_per_second": nominal_tokens / elapsed_seconds,
        "valid_tokens": valid_tokens,
        "valid_tokens_per_second": valid_tokens / elapsed_seconds,
        "samples_per_second": measure_steps * global_batch_size / elapsed_seconds,
        "peak_memory_allocated_gb": peak_allocated_gb,
        "peak_memory_reserved_gb": peak_reserved_gb,
        "last_loss": last_loss,
        "seed": seed,
    }
    if metadata:
        metrics.update(metadata)
    return metrics


def cuda_memory_gb(device: torch.device) -> Tuple[float, float]:
    allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    return allocated, reserved


def print_and_save_metrics(metrics: Dict[str, Any], result_path: str | Path | None) -> None:
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if result_path is None:
        return

    output_path = resolve_project_path(result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"结果已保存到: {output_path}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now() -> float:
    return time.perf_counter()


def aggregate_distributed_measurements(
    *,
    device: torch.device,
    elapsed_seconds: float,
    valid_tokens: int,
    last_loss: float,
    peak_allocated_gb: float,
    peak_reserved_gb: float,
) -> Dict[str, Any]:
    """聚合多进程 benchmark 指标，吞吐使用最慢 rank 的耗时。"""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed process group must be initialized.")

    world_size = dist.get_world_size()

    elapsed_tensor = torch.tensor(elapsed_seconds, dtype=torch.float64, device=device)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)

    valid_tokens_tensor = torch.tensor(valid_tokens, dtype=torch.int64, device=device)
    dist.all_reduce(valid_tokens_tensor, op=dist.ReduceOp.SUM)

    loss_tensor = torch.tensor(last_loss, dtype=torch.float64, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    loss_tensor /= world_size

    local_memory = torch.tensor(
        [peak_allocated_gb, peak_reserved_gb],
        dtype=torch.float64,
        device=device,
    )
    gathered_memory = [torch.empty_like(local_memory) for _ in range(world_size)]
    dist.all_gather(gathered_memory, local_memory)
    per_rank_allocated = [float(item[0].item()) for item in gathered_memory]
    per_rank_reserved = [float(item[1].item()) for item in gathered_memory]

    return {
        "elapsed_seconds": float(elapsed_tensor.item()),
        "valid_tokens": int(valid_tokens_tensor.item()),
        "last_loss": float(loss_tensor.item()),
        "peak_allocated_gb": max(per_rank_allocated),
        "peak_reserved_gb": max(per_rank_reserved),
        "per_rank_peak_allocated_gb": per_rank_allocated,
        "per_rank_peak_reserved_gb": per_rank_reserved,
    }
