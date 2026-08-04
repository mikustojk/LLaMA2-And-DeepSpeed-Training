import os
import sys

import torch
import deepspeed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from SFT_Dataset import SFTDataset

# =========================
# 1. 基本配置
# =========================

BASE_MODEL_DIR="models/happy-llm-215M-sft"
BASE_CHECKPOINT="checkpoints/sft_report_v2/model.pth"

TOKENIZER_PATH="tokenizer_k"
DATA_PATH="data/sft/learning_report_sft_v5_10000.jsonl"
DS_CONFIG="ds_config.json"

MAX_LENGTH=512
TARGET_STEPS = 2500
SAVE_DIR = "checkpoints/sft_report_v5"

for required_path in (BASE_MODEL_DIR, TOKENIZER_PATH, DATA_PATH, DS_CONFIG):
    if not os.path.exists(required_path):
        raise FileNotFoundError(f"Required SFT path does not exist: {required_path}")
if not os.path.isfile(BASE_CHECKPOINT):
    raise FileNotFoundError(f"Base checkpoint does not exist: {BASE_CHECKPOINT}")
if MAX_LENGTH < 2 or TARGET_STEPS < 1:
    raise ValueError("MAX_LENGTH must be at least 2 and TARGET_STEPS must be positive.")

# =========================
# 2. 导入模型
# =========================

sys.path.insert(0,BASE_MODEL_DIR)
from k_model import ModelConfig,Transformer

# =========================
# 3. 加载 tokenizer 和 Base 模型
# =========================

# 加载tokenizer和base模型
print("Loading tokenizer...")
tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_PATH)

print("Tokenizer vocab size:",len(tokenizer))

config=ModelConfig(
    dim=1024,
    n_layers=18,
    n_heads=16,
    n_kv_heads=8,
    vocab_size=len(tokenizer),
    max_seq_len=MAX_LENGTH,
    dropout=0.0,
)

print("Building model...")
model=Transformer(config)

print("Loading base checkpoint...")
state_dict=torch.load(BASE_CHECKPOINT,map_location="cpu")

# 有些权重名称可能带有_orig_mod. 前缀
new_state_dict={}

for key,value in state_dict.items():
    if key.startswith("_orig_mod."):
        key=key[len("_orig_mod."):]
    new_state_dict[key]=value

missing,unexpected=model.load_state_dict(new_state_dict,strict=False)

print("Missing keys:",missing)
print("Unexpected keys:",unexpected)
if missing or unexpected:
    raise RuntimeError(
        "Base checkpoint does not match the SFT model architecture. "
        f"Missing keys: {missing}; Unexpected keys: {unexpected}"
    )

# =========================
# 4. 创建优化器和 DeepSpeed
# =========================

optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(0.9,0.95),weight_decay=0.1)

print("Initializing DeepSpeed...")

model_engine,optimizer,_,_=deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    optimizer=optimizer,
    config=DS_CONFIG,
)

print("Device:",model_engine.device)

# =========================
# 5. 创建数据集和 DataLoader
# =========================

dataset=SFTDataset(DATA_PATH,tokenizer,max_length=MAX_LENGTH)

print("Dataset size:",len(dataset))

train_loader=DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

# =========================
# 6. 训练
# =========================

model_engine.train()

last_step=0
epoch=0

while model_engine.global_steps<TARGET_STEPS:
    epoch+=1
    print(f"Epoch {epoch}...")

    for x,y,loss_mask in train_loader:

        x=x.to(model_engine.device)
        y=y.to(model_engine.device)
        loss_mask=loss_mask.to(model_engine.device)

        # 前向传播
        outputs=model_engine(x,targets=y)

        # last_loss是每个token的损失
        token_loss=outputs.last_loss.view(x.size(0),-1)

        loss_mask=loss_mask.to(token_loss.dtype)

        # 只计算assistant生成的部分的平均损失
        valid_tokens=loss_mask.sum()

        if valid_tokens.item()==0:
            continue

        loss=(token_loss*loss_mask).sum()/valid_tokens

        # deepspeed进行反向传播和优化
        model_engine.backward(loss)
        model_engine.step()

        current_step=model_engine.global_steps
        if current_step!=last_step:
            print(f"Step {current_step}, loss: {loss.item():.4f}")
            last_step=current_step
        if current_step>=TARGET_STEPS:
            break

# =========================
# 7. 保存模型
# =========================

os.makedirs(SAVE_DIR,exist_ok=True)

# 保存deepspeed和checkpoint
model_engine.save_checkpoint(SAVE_DIR,tag=f"step-{TARGET_STEPS}")

# 保存普通模型的权重，方便后续推理
torch.save(model_engine.module.state_dict(),os.path.join(SAVE_DIR,"model.pth"))

print("Training completed. Model saved to",SAVE_DIR)

