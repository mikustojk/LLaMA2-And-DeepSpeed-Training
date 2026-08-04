# LLaMA2-And-DeepSpeed-Training

手动实现轻量级 LLaMA2 风格 Decoder-only 模型，完成 Tokenizer、预训练、SFT，并比较 Plain PyTorch 与 DeepSpeed 的训练性能。

## 项目内容

- `LLaMA2.py`：RMSNorm、RoPE、GQA、SwiGLU、DecoderLayer 和 Transformer。
- `BPE Tokenizer.py`：训练 ByteLevel BPE Tokenizer。
- `Pretrain_Dataset.py`、`SFT_Dataset.py`：预训练和指令微调数据处理及 loss mask。
- `pretrain_ds.py`、`sft_ds.py`：DeepSpeed 训练入口。
- `benchmark/`：Plain PyTorch、DDP 和 DeepSpeed 的速度、吞吐、显存对比脚本。
- `data/sft/learning_report_sft_v5_10000.jsonl`：学习报告 SFT 数据示例。

## 当前配置

- Tokenizer 词表：6144
- 模型配置：`dim=1024`、`n_layers=18`、`n_heads=16`、`n_kv_heads=8`
- 参数量：约 215.1M
- 训练精度：bf16

## 性能对比

benchmark 的运行命令和指标定义见 [`benchmark/README.md`](benchmark/README.md)，完整实验结果保存在 `benchmark/results/`。

## 数据和模型权重

完整预训练语料、模型权重、checkpoint 和本地虚拟环境不随代码仓库提交，请根据项目方案自行准备。
