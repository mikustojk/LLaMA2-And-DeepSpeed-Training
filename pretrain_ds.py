import argparse
import json
import os
import re
import shutil

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from LLaMA2 import ModelConfig,Transformer
from Pretrain_Dataset import PretrainDataset


def _extract_step(tag):
    if tag is None:
        return None
    match = re.fullmatch(r"step-(\d+)", str(tag).strip())
    return int(match.group(1)) if match else None


def _load_deepspeed_batch_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    try:
        train_batch_size = int(config["train_batch_size"])
        micro_batch_size = int(config["train_micro_batch_size_per_gpu"])
        accumulation_steps = int(config["gradient_accumulation_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "DeepSpeed 配置必须提供整数 train_batch_size、"
            "train_micro_batch_size_per_gpu 和 gradient_accumulation_steps。"
        ) from exc

    if train_batch_size < 1 or micro_batch_size < 1 or accumulation_steps < 1:
        raise ValueError("DeepSpeed 配置中的 batch 参数必须大于0。")
    return train_batch_size, micro_batch_size, accumulation_steps


def cleanup_old_checkpoints(save_dir, keep_last_n):
    if keep_last_n < 1:
        return

    checkpoint_dirs = []
    for name in os.listdir(save_dir):
        path = os.path.join(save_dir, name)
        if os.path.isdir(path) and _extract_step(name) is not None:
            checkpoint_dirs.append(path)

    checkpoint_dirs.sort(
        key=lambda path: _extract_step(os.path.basename(path))
    )

    for path in checkpoint_dirs[:-keep_last_n]:
        print(f"Removing old checkpoint: {path}")
        shutil.rmtree(path)


def save_checkpoint_with_retention(
    model_engine,
    save_dir,
    step,
    rank,
    keep_last_n,
):
    model_engine.save_checkpoint(
        save_dir,
        tag=f"step-{step}",
        client_state={"global_step": step},
    )

    # 所有进程完成保存后，只让 rank 0 清理旧 checkpoint。
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        cleanup_old_checkpoints(save_dir, keep_last_n)

    if dist.is_initialized():
        dist.barrier()

def parse_args():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl",
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tokenizer_k",
    )

    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="ds_config.json",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="模型隐藏维度；旧的768/12 checkpoint需要显式传入 --dim 768。",
    )

    parser.add_argument(
        "--n_layers",
        type=int,
        default=18,
        help="Transformer层数；旧的768/12 checkpoint需要显式传入 --n_layers 12。",
    )

    parser.add_argument(
        "--n_heads",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--n_kv_heads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--keep_last_n",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--resume_tag",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/pretrain_test",
    )

    #deeospeed启动时会自动传入这个参数
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
    )

    return parser.parse_args()

def main():
    args=parse_args()

    if args.max_length < 2:
        raise ValueError("max_length must be at least 2.")
    if args.max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    if args.save_every < 0:
        raise ValueError("save_every cannot be negative.")
    if args.log_every < 1:
        raise ValueError("log_every must be at least 1.")
    if args.keep_last_n < 1:
        raise ValueError("keep_last_n must be at least 1.")
    if args.dim < 1 or args.n_layers < 1 or args.n_heads < 1 or args.n_kv_heads < 1:
        raise ValueError("模型维度、层数和注意力头数必须大于0。")
    if args.dim % args.n_heads != 0:
        raise ValueError("dim 必须能被 n_heads 整除。")
    if args.n_heads % args.n_kv_heads != 0:
        raise ValueError("n_heads 必须能被 n_kv_heads 整除。")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0.")
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f"Pretraining data file does not exist: {args.data_path}")
    if not os.path.isfile(args.deepspeed_config):
        raise FileNotFoundError(
            f"DeepSpeed config file does not exist: {args.deepspeed_config}"
        )

    configured_global_batch, micro_batch_size, accumulation_steps = (
        _load_deepspeed_batch_config(args.deepspeed_config)
    )

    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_path)

    print("Tokenizer vocab size:",len(tokenizer))

    if len(tokenizer)!=6144:
        raise ValueError("Tokenizer vocab size is not 6144, please check the tokenizer path.")

    # 加载数据集
    print("Loading dataset...")
    dataset=PretrainDataset(args.data_path,tokenizer,max_length=args.max_length)    

    print("Dataset size:",len(dataset))

    # 创建模型配置
    model_config=ModelConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        vocab_size=len(tokenizer),
        max_seq_len=args.max_length,
        dropout=0.0,
    )

    print(
        "Building model with "
        f"dim={model_config.dim}, layers={model_config.n_layers}, "
        f"heads={model_config.n_heads}, kv_heads={model_config.n_kv_heads}..."
    )
    model=Transformer(model_config)

    # 定义优化器
    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9,0.95),
        weight_decay=0.1,
    )

    print("Initializing DeepSpeed...")

    model_engine,optimizer,_,_=deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        config=args.deepspeed_config,
    )

    # deepspeed初始化后才能确定rank和GPU的数量
    if dist.is_initialized():
        rank=dist.get_rank()
        world_size=dist.get_world_size()
    else:
        rank=0
        world_size=1

    expected_global_batch = micro_batch_size * accumulation_steps * world_size
    if configured_global_batch != expected_global_batch:
        raise ValueError(
            "DeepSpeed 配置中的 train_batch_size 与实际 world size 不匹配："
            f"期望 {expected_global_batch}，实际 {configured_global_batch}。"
        )

    if rank==0:
        print("Deepspeed initialized, rank:",rank,"world_size:",world_size)
        print("Device:",model_engine.device)

    loaded_step=0
    if args.resume_dir is not None:
        if not os.path.isdir(args.resume_dir):
            raise FileNotFoundError(
                f"Resume checkpoint directory does not exist: {args.resume_dir}"
            )

        if rank==0:
            print(
                "Loading checkpoint from:",
                args.resume_dir,
                "tag:",
                args.resume_tag or "latest",
            )

        load_path, client_state = model_engine.load_checkpoint(
            args.resume_dir,
            tag=args.resume_tag,
        )

        if load_path is None:
            raise RuntimeError(
                f"Failed to load checkpoint from {args.resume_dir}."
            )

        client_step=0
        if client_state is not None:
            client_step=int(client_state.get("global_step", 0))

        engine_step=int(getattr(model_engine, "global_steps", 0))

        tag_for_step=args.resume_tag
        if tag_for_step is None:
            latest_file=os.path.join(args.resume_dir, "latest")
            if os.path.isfile(latest_file):
                with open(latest_file, "r", encoding="utf-8") as f:
                    tag_for_step=f.read().strip()

        tag_step=_extract_step(tag_for_step) or 0
        loaded_step=max(client_step,engine_step,tag_step)

        # 旧 checkpoint 可能没有保存 global_step；以解析出的恢复步数为准。
        # 这样下一次 model_engine.step() 会从 loaded_step 继续递增。
        model_engine.global_steps=loaded_step

        if rank==0:
            print("Checkpoint loaded:",load_path)
            print("Resuming from global step:",loaded_step)

    if loaded_step>=args.max_steps:
        if rank==0:
            print(
                f"Checkpoint is already at step {loaded_step}; "
                f"max_steps is {args.max_steps}. Nothing to train."
            )
        return

    # 多GPU时使用DistributedSampler
    if world_size>1:
        sampler=DistributedSampler(dataset,num_replicas=world_size,rank=rank,shuffle=True)
        shuffle=False
    else:
        sampler=None
        shuffle=True

    train_loader=DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )

    os.makedirs(args.save_dir,exist_ok=True)

    model_engine.train()

    global_step=loaded_step
    last_logged_step=loaded_step
    last_saved_step=loaded_step

    for epoch in range(100000):
        if sampler is not None:
            sampler.set_epoch(epoch)

        # 读取一个batch的数据
        for x,y,loss_mask in train_loader:
            x=x.to(model_engine.device,non_blocking=True)
            y=y.to(model_engine.device,non_blocking=True)
            loss_mask=loss_mask.to(model_engine.device,non_blocking=True)

            outputs=model_engine(x,targets=y)

            #outputs.last_loss的原始形状是[batch_size*seq_len]
            token_loss=outputs.last_loss.view(x.size(0),-1)

            loss_mask=loss_mask.to(token_loss.dtype)

            assert token_loss.shape == loss_mask.shape

            # 只计算loss_mask=1的位置
            loss=(token_loss*loss_mask).sum()
            loss=loss/loss_mask.sum().clamp_min(1.0)

            model_engine.backward(loss)
            model_engine.step()

            global_step=model_engine.global_steps

            if(
                rank==0
                and global_step>last_logged_step
                and global_step%args.log_every==0
            ):
                print(f"Step {global_step}, loss: {loss.item():.6f}")
                last_logged_step=global_step

            if(
                args.save_every>0
                and global_step>last_saved_step
                and global_step%args.save_every==0
            ):
                if rank==0:
                    print(f"Saving checkpoint at step {global_step} to: {args.save_dir}")
                save_checkpoint_with_retention(
                    model_engine,
                    args.save_dir,
                    global_step,
                    rank,
                    args.keep_last_n,
                )
                last_saved_step=global_step

            if global_step>=args.max_steps:
                break
        if global_step>=args.max_steps:
            break

    if rank==0:
        print("Training finished")

    # 所有rank都要调用save_checkpoint
    if global_step!=last_saved_step:
        if rank==0:
            print("Saving final checkpoint to:",args.save_dir)
        save_checkpoint_with_retention(
            model_engine,
            args.save_dir,
            global_step,
            rank,
            args.keep_last_n,
        )
        last_saved_step=global_step

    if rank==0:
        print(f"Latest checkpoint is step {last_saved_step}.")

if __name__=="__main__":
    main()
