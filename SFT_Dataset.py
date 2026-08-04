import json
import os
from array import array

import numpy as np
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    # 多轮对话数据集
    # 输入是上一轮对话的内容，输出是当前轮对话的内容
    def __init__(self,data_path,tokenizer,max_length=512):
        super().__init__()
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"SFT data file does not exist: {data_path}")
        if max_length < 2:
            raise ValueError("max_length must be at least 2.")
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id for SFT loss masking.")
        self.data_path=data_path
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.padding=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.eos_token_id=tokenizer.eos_token_id
        self.assistant_marker_ids=tokenizer(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )["input_ids"]
        if not self.assistant_marker_ids:
            raise ValueError("Could not tokenize the assistant chat-template marker.")
        self._offsets=array('Q',[0])
        with open(data_path,'rb') as f:
            self._offsets.append(0)# 第一行的起始字节偏移量为0
            while f.readline():
                self._offsets.append(f.tell())
        self.total_lines=len(self._offsets)-1

    def __len__(self):
        return self.total_lines

    def generate_loss_mask(self,input_ids,valid_length=None):# 作用是把需要assitant生成的（即夹在<|im_start|>assistant\n和eos_token_id之间的部分）标记为1，其他部分标记为0
        mask=[0]*(len(input_ids))
        a_sequence=self.assistant_marker_ids
        a_length=len(a_sequence)
        n=len(input_ids) if valid_length is None else min(valid_length,len(input_ids))
        i=0

        while i<=n-a_length:
            # 检查当前位置是否匹配目标子序列
            match=True
            for k in range(a_length):
                if input_ids[i+k]!=a_sequence[k]:
                    match=False
                    break
            if match:
                # 只在未 padding 的有效文本范围内查找 eos，避免 pad_token_id 与 eos_token_id 相同时误判。
                j=None
                for idx in range(i+a_length,n):
                    if input_ids[idx]==self.eos_token_id:
                        j=idx
                        break
                start=i+a_length
                # 找不到 eos 时，说明回答被 max_length 截断，标记到有效文本末尾。
                end=(j+1) if j is not None else n
                for pos in range(start,min(end,n)):
                    mask[pos]=1
                # 跳过当前子序列，避免重叠匹配
                i+=a_length
            else:
                i+=1
        return mask
    
    def __getitem__(self,index:int):
        with open(self.data_path,'rb') as f:
            f.seek(self._offsets[index])
            line=f.readline().decode('utf-8')
        sample=json.loads(line)
        text=self.tokenizer.apply_chat_template(sample,tokenize=False,add_generation_prompt=False)
        input_id=self.tokenizer(
            text,
            add_special_tokens=False,
        ).data['input_ids'][:self.max_length]
        text_len=len(input_id)
        # 没满最大长度的剩余部分
        padding_len=self.max_length-text_len
        input_id+=[self.padding]*padding_len
        loss_mask=self.generate_loss_mask(input_id,text_len)

        input_id=np.array(input_id)
        x=np.array(input_id[:-1]).astype(np.int64)
        y=np.array(input_id[1:]).astype(np.int64)
        loss_mask=np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(loss_mask)
