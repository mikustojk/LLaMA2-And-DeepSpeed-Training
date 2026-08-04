# Plain PyTorch 与 DeepSpeed 预训练对比

这套脚本包含单卡对比和双卡对比，使用相同的 LLaMA2 配置、Tokenizer、预训练数据、BF16、AdamW、梯度裁剪和全局 batch size。模型配置固定为 `dim=1024`、`n_layers=18`、`n_heads=16`、`n_kv_heads=8`，与项目方案和 SFT 模型配置一致。

## 测量定义

- 预热：默认 20 个 optimizer steps，不计入稳定吞吐结果。
- 正式测量：默认 1000 个 optimizer steps。
- 单卡全局 batch size 固定为 8。默认使用 micro batch 1、梯度累积 8 次；micro batch 扫描使用 `1/8、2/4、4/2` 三组配置。
- 序列长度为 `max_length - 1`，默认是 511。
- 测量区间不保存 checkpoint，也不打印每一步 loss。
- 结果包括 step time、samples/s、nominal tokens/s、valid tokens/s、峰值显存、最后一个 loss 和完整实验配置。

当前 `PretrainDataset` 会在取样时读取 JSONL 并调用 tokenizer，所以这版结果是端到端数据管道吞吐。两种脚本使用同一个 Dataset，比较仍然是公平的；后续如果只比较 GPU 计算，需要再增加预加载 token ID 的版本。

DeepSpeed 脚本的 `--micro_batch_size` 和 `--gradient_accumulation_steps` 会覆盖当前进程内读取到的 JSON batch 配置，不会修改 `ds_config_bench.json`。脚本会校验两者乘积必须等于 8。

双卡实验继续固定全局 batch size 为 8。每张 GPU 的 micro batch 为 4，梯度累积为 1，因此 `4 × 1 × 2 = 8`。Plain PyTorch 使用 DDP，DeepSpeed 使用 ZeRO-2。两个脚本都用 `DistributedSampler` 将数据分给两个 rank，并按照最慢 rank 的耗时计算两张卡的总吞吐。结果 JSON 同时记录每个 rank 的 allocated 和 reserved 峰值显存。

## 第一阶段：单卡 micro batch 扫描

先用最大的候选 micro batch 做单步 smoke test，确认两种方案都不会 OOM：

```bash
cd /mnt/0608/guohyx/LLaMA2
mkdir -p benchmark/results

CUDA_VISIBLE_DEVICES=0 uv run python benchmark/pretrain_plain_bench.py \
  --data_path data/seq_monkey/test_1000.jsonl \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --warmup_steps 1 \
  --measure_steps 1 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/plain_micro4_smoke.json

CUDA_VISIBLE_DEVICES=0 uv run deepspeed benchmark/pretrain_ds_bench.py \
  --data_path data/seq_monkey/test_1000.jsonl \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --warmup_steps 1 \
  --measure_steps 1 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/deepspeed_micro4_smoke.json
```

smoke test 通过后，依次运行下面四个正式测试。每条命令使用同一张空闲 GPU，按顺序执行，不要并发运行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python benchmark/pretrain_plain_bench.py \
  --data_path data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl \
  --micro_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --warmup_steps 20 \
  --measure_steps 1000 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/plain_micro2_accum4_1000.json

CUDA_VISIBLE_DEVICES=0 uv run deepspeed benchmark/pretrain_ds_bench.py \
  --data_path data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl \
  --micro_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --warmup_steps 20 \
  --measure_steps 1000 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/deepspeed_micro2_accum4_1000.json

CUDA_VISIBLE_DEVICES=0 uv run python benchmark/pretrain_plain_bench.py \
  --data_path data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --warmup_steps 20 \
  --measure_steps 1000 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/plain_micro4_accum2_1000.json

CUDA_VISIBLE_DEVICES=0 uv run deepspeed benchmark/pretrain_ds_bench.py \
  --data_path data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --warmup_steps 20 \
  --measure_steps 1000 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/deepspeed_micro4_accum2_1000.json
```

已有的 `micro=1、accumulation=8` 正式结果作为基线。完成四个新测试后，先比较三组配置的 tokens/s 和峰值显存，再决定双卡实验采用哪一组 micro batch。

## 第二阶段：双卡对比

双卡测试必须在两张 GPU 都有足够空闲显存时运行。Plain DDP 和 DeepSpeed 要按顺序运行，不能同时占用两张卡。

先运行 smoke test：

```bash
cd /mnt/0608/guohyx/LLaMA2
mkdir -p benchmark/results
nvidia-smi

CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  benchmark/pretrain_plain_ddp_bench.py \
  --data_path data/seq_monkey/test_1000.jsonl \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --warmup_steps 1 \
  --measure_steps 1 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/plain_ddp_2gpu_micro4_accum1_smoke.json

CUDA_VISIBLE_DEVICES=0,1 uv run deepspeed \
  benchmark/pretrain_ds_2gpu_bench.py \
  --data_path data/seq_monkey/test_1000.jsonl \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --warmup_steps 1 \
  --measure_steps 1 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/deepspeed_2gpu_micro4_accum1_smoke.json
```

两条 smoke test 都成功并生成 JSON 后，再按顺序运行正式测试：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  benchmark/pretrain_plain_ddp_bench.py \
  --data_path data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --warmup_steps 20 \
  --measure_steps 1000 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/plain_ddp_2gpu_micro4_accum1_1000.json

CUDA_VISIBLE_DEVICES=0,1 uv run deepspeed \
  benchmark/pretrain_ds_2gpu_bench.py \
  --data_path data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --warmup_steps 20 \
  --measure_steps 1000 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/deepspeed_2gpu_micro4_accum1_1000.json
```

双卡结果需要做两类比较：

1. 在两张卡上比较 DDP 与 DeepSpeed 的 step time、总 tokens/s 和每卡峰值显存。
2. 将双卡结果分别与对应的单卡 `micro=4、accumulation=2` 结果比较。扩展倍数等于“双卡吞吐 ÷ 单卡吞吐”，扩展效率等于“扩展倍数 ÷ 2”。

双卡脚本只输出 benchmark JSON，不保存 checkpoint。

## 第三阶段：增大模型规模

当前基线配置为 `dim=1024、n_layers=18、n_heads=16、n_kv_heads=8`，实际参数量为 215,127,040，约 215.1M。第一档放大配置使用 `dim=1536、n_layers=24、n_heads=24、n_kv_heads=8`，头维度仍为 64，实际参数量应为 613,492,224，约 613.5M。脚本会在运行时重新统计参数量并写入结果 JSON。

先按顺序运行两组 smoke test：

```bash
cd /mnt/0608/guohyx/LLaMA2
mkdir -p benchmark/results
nvidia-smi

CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  benchmark/pretrain_plain_ddp_bench.py \
  --data_path data/seq_monkey/test_1000.jsonl \
  --dim 1536 \
  --n_layers 24 \
  --n_heads 24 \
  --n_kv_heads 8 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --warmup_steps 1 \
  --measure_steps 1 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/plain_ddp_2gpu_613m_smoke.json

CUDA_VISIBLE_DEVICES=0,1 uv run deepspeed \
  benchmark/pretrain_ds_2gpu_bench.py \
  --data_path data/seq_monkey/test_1000.jsonl \
  --dim 1536 \
  --n_layers 24 \
  --n_heads 24 \
  --n_kv_heads 8 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --warmup_steps 1 \
  --measure_steps 1 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/deepspeed_2gpu_613m_smoke.json
```

两组 smoke test 都成功后，再按顺序运行正式测试：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  benchmark/pretrain_plain_ddp_bench.py \
  --data_path data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl \
  --dim 1536 \
  --n_layers 24 \
  --n_heads 24 \
  --n_kv_heads 8 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --warmup_steps 20 \
  --measure_steps 1000 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/plain_ddp_2gpu_613m_1000.json

CUDA_VISIBLE_DEVICES=0,1 uv run deepspeed \
  benchmark/pretrain_ds_2gpu_bench.py \
  --data_path data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl \
  --dim 1536 \
  --n_layers 24 \
  --n_heads 24 \
  --n_kv_heads 8 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --warmup_steps 20 \
  --measure_steps 1000 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/deepspeed_2gpu_613m_1000.json
```

这组实验继续固定 global batch size 为 8，只改变模型规模。结果比较使用 tokens/s 和每卡峰值显存；如果任一方案 OOM，要保留完整报错和当时的 `nvidia-smi` 输出。

## 快速 smoke test

先用 1000 条测试子集确认依赖、模型和显存都能正常运行：

```bash
cd /mnt/0608/guohyx/LLaMA2
mkdir -p benchmark/results

CUDA_VISIBLE_DEVICES=0 uv run python benchmark/pretrain_plain_bench.py \
  --data_path data/seq_monkey/test_1000.jsonl \
  --warmup_steps 1 \
  --measure_steps 1 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/plain_smoke.json

CUDA_VISIBLE_DEVICES=0 uv run deepspeed benchmark/pretrain_ds_bench.py \
  --data_path data/seq_monkey/test_1000.jsonl \
  --warmup_steps 1 \
  --measure_steps 1 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/deepspeed_smoke.json
```

## 完整数据的 1000 steps 命令

完整数据文件包含约 13,000,000 条 JSONL 样本，正式比较时显式指定它：

```bash
cd /mnt/0608/guohyx/LLaMA2
mkdir -p benchmark/results

CUDA_VISIBLE_DEVICES=0 uv run python benchmark/pretrain_plain_bench.py \
  --data_path data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl \
  --warmup_steps 20 \
  --measure_steps 1000 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/plain_full_1000.json

CUDA_VISIBLE_DEVICES=0 uv run deepspeed benchmark/pretrain_ds_bench.py \
  --data_path data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl \
  --warmup_steps 20 \
  --measure_steps 1000 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/deepspeed_full_1000.json
```

## 10000 steps命令

如果要测 10000 个 optimizer steps，只需要把 `--measure_steps` 改成 10000：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python benchmark/pretrain_plain_bench.py \
  --data_path data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl \
  --warmup_steps 20 \
  --measure_steps 10000 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/plain_full_10000.json

CUDA_VISIBLE_DEVICES=0 uv run deepspeed benchmark/pretrain_ds_bench.py \
  --data_path data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl \
  --warmup_steps 20 \
  --measure_steps 10000 \
  --gradient_clipping 1.0 \
  --result_path benchmark/results/deepspeed_full_10000.json
```

## 结果解释

先比较 `seconds_per_optimizer_step`、`nominal_tokens_per_second` 和峰值显存。两种方案必须使用相同的 `measure_steps` 和 `global_batch_size`。

1000 steps 通常已经足够测量稳定吞吐；10000 steps 更适合检查长期 loss、显存泄漏和周期性抖动。10000 steps 的运行时间约为 1000 steps 的十倍，不会自动带来十倍的测量精度。

正式结果建议每种方案连续运行 3 次，比较中位数。两种方案要使用相同的 `--data_path`、`--measure_steps`、模型参数和 GPU。脚本不会保存 checkpoint。
