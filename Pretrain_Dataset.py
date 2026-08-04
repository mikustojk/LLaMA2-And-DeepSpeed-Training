import json
import os
from array import array

import numpy as np
import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_length=512):
        super().__init__()
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Pretraining data file does not exist: {data_path}")
        if max_length < 2:
            raise ValueError("max_length must be at least 2.")
        self.data_path=data_path
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.padding=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # 预计算每行的起始字节的偏移量
        # 使用定长无符号整数保存字节偏移，避免 1300 万行时 Python int/list 带来过大的内存开销。
        self._offsets=array('Q',[0])
        with open(data_path,'rb') as f:
            while f.readline():
                self._offsets.append(f.tell())# 记录每行的起始字节偏移量    f.tell()返回文件当前的字节位置
        self.total_lines=len(self._offsets)-1# 总行数

    def __len__(self):
        return self.total_lines
    
    def __getitem__(self,index:int):
        with open(self.data_path,'rb') as f:
            f.seek(self._offsets[index])# 定位到指定行的起始字节偏移量
            line=f.readline().decode('utf-8')# 读取该行并解码为字符串
        sample=json.loads(line)
        text=f"{self.tokenizer.bos_token}{sample['text']}"
        input_id=self.tokenizer(text).data['input_ids'][:self.max_length]  # tokenizer返回BatchEncoding对象，取其中的input_ids，并截断到最大长度
        text_len=len(input_id)
        # 没满最大长度的剩余部分
        padding_len=self.max_length-text_len
        input_id+=[self.padding]*padding_len
        # 0表示不计算损失
        loss_mask=[1]*text_len+[0]*padding_len

        input_id=np.array(input_id)
        x=np.array(input_id[:-1]).astype(np.int64)# x为前n-1个元素
        y=np.array(input_id[1:]).astype(np.int64)# y为后n-1个元素
        loss_mask=np.array(loss_mask[1:]).astype(np.int64)# mask去掉第一个位置为了和y对齐

        return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(loss_mask)
