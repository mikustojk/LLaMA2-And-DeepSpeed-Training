# 项目协作规则

本文件适用于 `D:\\bupt\\多模态大模型\\动手搭建LLaMA2` 项目。

当前项目主线是比较 Plain PyTorch、DeepSpeed 和 Megatron 的预训练与 SFT，不把嵌入式部署作为主要交付目标。当前优先完成 Plain PyTorch 与 DeepSpeed，Megatron 暂缓。

## 1. 每次协作前后的必做事项

1. 处理项目问题前，先阅读同目录的 `项目方案.md`，以其中记录的当前状态、路径和训练配置为准。
2. 每次项目相关回答结束前，都要维护 `项目方案.md`：
   - 如果项目状态发生变化，更新对应的完成项和下一步计划。
   - 如果只是解释问题，也在“变更记录”中追加一条简短记录，说明本次确认的事实或决策。
3. 不把计划中的内容写成已经完成的内容。必须区分“已验证”“已配置”“待执行”和“未验证”。

## 2. 修改代码和数据的权限

1. 用户没有明确要求修改时，只阅读、解释和诊断，不直接改代码、数据集或训练配置。
2. 用户明确要求修改时，先在回答中说明：将修改哪些文件、哪些配置、修改目的是什么，然后再执行修改。
3. 不自动启动长时间训练、占用多张 GPU 的任务或删除 checkpoint。用户明确要求运行时，先给出预计 GPU、显存和磁盘影响。
4. 不删除旧数据、旧 checkpoint 或旧脚本，除非用户明确指定删除范围。
5. 所有代码编辑使用 `apply_patch`，不要用重定向或脚本覆盖文件。保留用户已有改动，避免无关格式化。

## 3. 训练和实验规则

1. 训练实验必须记录模型配置、Tokenizer、数据集、精度、GPU 数量、global batch size、gradient accumulation、学习率和测量步数。
2. 对比 Plain PyTorch、DeepSpeed 和 Megatron 时，优先固定：模型结构、输入 token、精度、优化器、global batch size、序列长度和随机种子。
3. 训练吞吐至少报告 samples/s 和 tokens/s；显存至少报告每张卡的峰值 allocated 和 reserved 显存。
4. GPU 计时必须在计时区间前后调用 `torch.cuda.synchronize()`，并在正式测量前清空峰值显存统计。
5. 当前 SFT 数据的有效 mode 是 `reading` 和 `courseware`。除非数据集和模型已经同步扩展，否则不要把其他 mode 当作有效测试输入。
6. 程序计算的总时长、有效率、分数和等级是可信数值来源，模型只负责生成 AI 分析和 AI 建议。
7. Megatron 不是当前模型的直接包装器。若引入 Megatron，必须记录模型结构、Tokenizer、数据格式和并行方式是否与当前 LLaMA2 一致。

## 4. 回答和解释规则

1. 使用中文，先给结论，再给与当前代码对应的原因和操作步骤。
2. 解释 Python、PyTorch、Transformer 或 DeepSpeed 概念时，优先引用项目中的实际变量、张量形状和函数调用。
3. 不把推测说成事实。涉及训练是否成功、模型效果、显存占用和速度时，给出日志、代码或测量结果依据。
4. 发现路径、参数、数据格式或训练模式不一致时，先指出具体不一致，再给修改方案。
5. 给出命令时使用适合用户当前环境的完整路径和可复制命令，并说明命令是否会启动训练或写入文件。
6. 需要最新库行为、官方命令或当前模型信息时，优先查阅官方文档并附来源；不要凭旧版本参数直接断言。

## 5. 项目中的关键约定

- Tokenizer：`tokenizer_k`，词表大小 6144。
- 模型实现：`LLaMA2.py`。
- 预训练数据集类：`Pretrain_Dataset.py`。
- SFT 数据集类：`SFT_Dataset.py`。
- DeepSpeed 预训练入口：`pretrain_ds.py`。
- DeepSpeed SFT 入口：`sft_ds.py`。
- SFT 测试入口：`test_sft.py`。
- 当前 SFT 数据：`data/sft/learning_report_sft_v5_10000.jsonl`。
- 当前 SFT 输出：`checkpoints/sft_report_v5`。
- 当前 SFT 测试 checkpoint：`checkpoints/sft_report_v5/model.pth`。

## 6. 安全边界

1. 不执行 `git reset --hard`、递归删除或覆盖性迁移，除非用户明确授权。
2. 不上传、发布或推送项目文件。
3. 训练数据可能包含用户项目数据，处理时只在当前项目范围内读写。
4. 如果环境权限、GPU 占用或文件缺失阻止任务，说明实际阻塞原因，并给出最小替代方案。
