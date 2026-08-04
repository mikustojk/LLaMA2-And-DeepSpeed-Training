import random
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator

# 读取json文件并提取文本数据
def read_texts_from_jsonl(file_path:str)->Generator[str,None,None]:
    """读取json文件并提取文本数据"""
    with open(file_path,'r',encoding='utf-8') as f:
        for line_num,line in enumerate(f,1):
            try:
                data=json.loads(line)
                if not isinstance(data,dict) or 'text' not in data:
                    print(f"Missing 'text' field on line {line_num}")
                    continue
                text=data['text']
                if not isinstance(text,str) or not text:
                    print(f"Invalid text field on line {line_num}")
                    continue
                yield text
            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {line_num}")
                continue
            except UnicodeDecodeError:
                print(f"Error decoding UTF-8 on line {line_num}")
                continue

# 创建配置文件,定义tokenizer的参数和特殊标记
def create_tokenizer_config(save_dir:str)->None:
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (# chat_template与Qwen2.5一致
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置文件
    with open(os.path.join(save_dir,"tokenizer_config.json"),"w",encoding='utf-8') as f:
        json.dump(config,f,ensure_ascii=False,indent=4)
    
    # 创建special_token_map.json
    special_token_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "additional_special_tokens": ["<s>","</s>"]
    }

    with open(os.path.join(save_dir,"special_tokens_map.json"),"w",encoding='utf-8') as f:
        json.dump(special_token_map,f,ensure_ascii=False,indent=4)

def train_tokenizer(data_path:str,save_dir:str,vocab_size:int=8192)->None:
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Tokenizer training data does not exist: {data_path}")
    if vocab_size < len(["<unk>","<s>","</s>","<|im_start|>","<|im_end|>"]):
        raise ValueError("vocab_size is too small for the required special tokens.")
    os.makedirs(save_dir,exist_ok=True)

    # 初始化tokenizer
    tokenizer=Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer=NFKC()# 给tokenizer添加NFKC标准化器
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)# 添加预分词器
    tokenizer.decoder=decoders.ByteLevel()# 添加解码器

    # 配置特殊token,会作为完整的token而不会被拆分
    special_tokens=[
        "<unk>",
        "<s>",
        "</s>",
        "<|im_start|>",
        "<|im_end|>"
    ]

    # 配置BPE训练器
    trainer=trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,# 提高低频词过滤
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 训练tokenizer
    print(f"Training tokenizer on data from {data_path}...")
    texts=read_texts_from_jsonl(data_path)
    # length 应该是样本条数，而不是 JSONL 文件的字节数；这里省略它，避免进度条失真。
    tokenizer.train_from_iterator(texts,trainer=trainer)

    # 验证特殊token映射
    try:
        assert tokenizer.token_to_id("<unk>")==0
        assert tokenizer.token_to_id("<s>")==1
        assert tokenizer.token_to_id("</s>")==2
        assert tokenizer.token_to_id("<|im_start|>")==3
        assert tokenizer.token_to_id("<|im_end|>")==4
    except AssertionError as e:
        print("Error: Special token mapping is incorrect:", e)
        raise

    # 保存tokenizer
    tokenizer.save(os.path.join(save_dir,"tokenizer.json"))

    # 创建配置文件
    create_tokenizer_config(save_dir)
    print(f"Tokenizer training complete. Files saved to {save_dir}.")

def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 测试基本属性
    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # 测试聊天模板
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]
    
    print("\n=== 聊天模板测试 ===")
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        # add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")

    # 测试编码解码
    print("\n=== 编码解码测试 ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt)

    # 测试特殊token处理
    print("\n=== 特殊token处理 ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print("Special tokens preserved:", decoded == test_text)

def main():
    data_path = "data/seq_monkey/mobvoi_seq_monkey_general_open_corpus.jsonl"
    save_dir = "tokenizer_k"

    train_tokenizer(data_path,save_dir,vocab_size=6144)

    eval_tokenizer(save_dir)

if __name__=="__main__":
    main()
