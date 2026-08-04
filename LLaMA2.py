from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math
from typing import Optional
from transformers.modeling_outputs import CausalLMOutputWithPast
#--------------------------------------------------------
# 定义超参数
class ModelConfig(PretrainedConfig):
    model_type="Tiny-k"
    def __init__(
            self,
            dim:int=1024, # 模型维度
            n_layers:int=18, # transformer层数
            n_heads:int=16, # 注意力头数
            n_kv_heads:int=8, # 键值头的数量
            vocab_size:int=6144, # 词汇表大小
            hidden_dim:int=None, # 隐藏层维度
            multiple_of:int=64,
            norm_eps:float=1e-5, # 归一化的epsilon值
            max_seq_len:int=512, # 最大序列长度
            dropout:float=0.0, # dropout率
            flash_attention:bool=True, # 是否使用flash attention
            **kwargs
    ):
        if dim < 1 or n_layers < 1 or n_heads < 1:
            raise ValueError("dim、n_layers 和 n_heads 必须大于0。")
        if dim % n_heads != 0:
            raise ValueError("dim 必须能被 n_heads 整除。")
        if n_kv_heads is not None:
            if n_kv_heads < 1 or n_heads % n_kv_heads != 0:
                raise ValueError("n_kv_heads 必须大于0且能整除 n_heads。")
        if max_seq_len < 2:
            raise ValueError("max_seq_len 必须至少为2。")
        self.dim=dim
        self.n_layers=n_layers
        self.n_heads=n_heads
        self.n_kv_heads=n_kv_heads
        self.vocab_size=vocab_size
        self.hidden_dim=hidden_dim
        self.multiple_of=multiple_of
        self.norm_eps=norm_eps
        self.max_seq_len=max_seq_len
        self.dropout=dropout
        self.flash_attention=flash_attention
        super().__init__(**kwargs)
#--------------------------------------------------------
# 构建RMSNorm
class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float):
        super().__init__()
        # eps防止除0
        self.eps=eps
        # 一个可学习的权重参数
        self.weight=nn.Parameter(torch.ones(dim))

    def _norm(self,x):
        # rsqrt是计算平方根的倒数
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    
    def forward(self,x):
        # 先将x转为float，计算完后再转回去
        output=self._norm(x.float()).type_as(x)
        # 最后乘权重并返回
        return output*self.weight

#---------------------------------------------------------    
#---------------------------------------------------------
# 构建LLaMA2 Attention
# 选择使用GQA，可以提高效率并降低显存占用
#---------------------------------------------------------
#---------------------------------------------------------

# repeat_kv,需要将k和v的维度扩展成和q一样，才可以进行计算
def repeat_kv(x:torch.Tensor, n_rep:int)->torch.Tensor:
    # 获取tensor形状： 批量大小，序列长度，kv头数，每个头的维度大小
    bs,slen,n_kv_heads,head_dim=x.shape

    # 重复次数为1直接返回
    if n_rep==1:
        return x
    
    # 对张量进行扩展和重塑以重复键值对
    return (
        x[:,:,:,None,:] # 在第四个维度（头的维度）前添加一个维度
        .expand(bs,slen,n_kv_heads,n_rep,head_dim) # 扩展到指定的重复次数
        .reshape(bs,slen,n_kv_heads*n_rep,head_dim) # 重塑
    )

#--------------------------------------------------------
# RoPE位置编码
# 这里的dim是dim//head，因为每个头单独进行旋转嵌入
def precompute_freqs_cis(dim:int,end:int,theta:float=10000.0):
    # end:要计算的位置总数
    # [:(dim//2)]进行切片（dim为偶数时不改变结果）
    freqs=1.0/(theta**(torch.arange(0,dim,2)[:(dim//2)].float()/dim))
    t=torch.arange(end,device=freqs.device)
    # 计算外积
    freqs=torch.outer(t,freqs).float()
    # cos和sin的维度都是[end, dim//2]
    # 即对于每个m从0~end-1，都维护一个R_m，每个R_m里面都有dim/2个cos和sin
    freqs_cos=torch.cos(freqs)
    freqs_sin=torch.sin(freqs)
    return freqs_cos,freqs_sin

# 调整freqs_cis的形状，与x的维度对齐，可以直接进行广播
def reshape_for_boardcast(freqs_cis:torch.Tensor,x:torch.Tensor):
    # x.shape=[batch_size,seq_len,n_heads,head_dim]
    # freqs_cis.shape=[seq_len,head_dim]
    ndim=x.ndim
    assert 0<=1<ndim
    # 确保freqs_cis的形状与x的第二个和最后一个维度匹配
    assert freqs_cis.shape==(x.shape[1],x.shape[-1])
    # enumarate返回 (下标，索引值)
    shape=[d if i==1 or i==ndim-1 else 1 for i,d in enumerate(x.shape)]
    return freqs_cis.view(shape) # freqs_cis(1,seq_len,1,head_dim)

# 实现RoPE旋转嵌入
def apply_rotary_emb(# LLaMA2只有q和k需要RoPE
    xq:torch.Tensor,
    xk:torch.Tensor,
    freqs_cos:torch.Tensor,
    freqs_sin:torch.Tensor,
)-> Tuple[torch.Tensor,torch.Tensor]:
    # xq,xk[batch_size,seq_len,dim//n_head,n_head_dim]
    # 将q和k张量变成浮点类型，并重塑形状分离实部虚部（即将q和k的元素两两分组，每组和R_m的2x2小块相乘）
    xq_r,xq_i=xq.float().reshape(xq.shape[:-1]+(-1,2)).unbind(-1)
    xk_r,xk_i=xk.float().reshape(xk.shape[:-1]+(-1,2)).unbind(-1)

    #xq_r,xq_i,xk_r,xk_i的形状都是[batch_size,seq_len,dim//n_head,n_head_dim/2]
    
    # 重塑freqs_cos和freqs_sin的形状以便广播
    freqs_cos=reshape_for_boardcast(freqs_cos,xq_r)
    freqs_sin=reshape_for_boardcast(freqs_sin,xq_r)
    # freqs_cis(1,seq_len,1,n_head_dim/2)
    # 分别计算旋转后的实部和虚部
    xq_out_r=xq_r*freqs_cos-xq_i*freqs_sin
    xq_out_i=xq_r*freqs_sin+xq_i*freqs_cos
    xk_out_r=xk_r*freqs_cos-xk_i*freqs_sin
    xk_out_i=xk_r*freqs_sin+xk_i*freqs_cos

    # 将两个维度进行合并，还原形状
    xq_out=torch.stack([xq_out_r,xq_out_i],dim=-1).flatten(3)
    xk_out=torch.stack([xk_out_r,xk_out_i],dim=-1).flatten(3)
    # 返回旋转后的q和k
    return xq_out.type_as(xq),xk_out.type_as(xk)

#-------------------------------------------------------- 

# 组装LLaMA2 Attention
class Attention(nn.Module):
    def __init__(self,args:ModelConfig):
        super().__init__()
        # 确定键值头的数量
        self.n_kv_heads=args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保总头数可以被键值头整除
        assert args.n_heads%self.n_kv_heads==0

        # 并行处理，默认为1
        model_parallel=1
        # 计算本地头数
        self.n_local_head=args.n_heads//model_parallel
        self.n_local_kv_head=self.n_kv_heads//model_parallel
        # 重复次数，用于扩展kv的尺寸
        self.n_rep=self.n_local_head//self.n_local_kv_head
        self.head_dim=args.dim//args.n_heads

        # 定义权重矩阵
        self.wq=nn.Linear(args.dim,args.n_heads*self.head_dim,bias=False)
        self.wk=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        self.wv=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        # 输出矩阵
        self.wo=nn.Linear(args.n_heads*self.head_dim,args.dim,bias=False)

        # 定义dropout
        self.attn_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout

        # 是否使用flash attention
        self.flash=bool(args.flash_attention) and hasattr(
            torch.nn.functional,'scaled_dot_product_attention'
        )
        if not self.flash:
            if args.flash_attention:
                print("Warning: Flash Attention is not supported in this version of PyTorch. Falling back to standard attention implementation.")
            else:
                print("Flash Attention is disabled. Using the standard attention implementation.")
            mask=torch.full((1,1,args.max_seq_len,args.max_seq_len),float('-inf'))
            # diagonal=1表示上三角矩阵，mask的上三角部分为-inf，下三角部分为0
            mask=torch.triu(mask,diagonal=1)
            self.register_buffer("mask",mask)

    def forward(self,x:torch.Tensor,freqs_cos:torch.Tensor,freqs_sin:torch.Tensor):
        bsz,seq_len,_=x.shape

        xq,xk,xv=self.wq(x),self.wk(x),self.wv(x)
        # 调整矩阵大小
        xq=xq.view(bsz,seq_len,self.n_local_head,self.head_dim)
        xk=xk.view(bsz,seq_len,self.n_local_kv_head,self.head_dim)
        xv=xv.view(bsz,seq_len,self.n_local_kv_head,self.head_dim)
        # RoPE
        xq,xk=apply_rotary_emb(xq,xk,freqs_cos,freqs_sin)
        # repeat_kv
        xk=repeat_kv(xk,self.n_rep)
        xv=repeat_kv(xv,self.n_rep)

        # 将头作为批量维度（即交换seq_len和n_local_head）
        xq=xq.transpose(1,2)
        xk=xk.transpose(1,2)
        xv=xv.transpose(1,2)

        # 根据是否使用flash attention选择不同的计算方式
        if self.flash:
            # attn_mask=None表示不使用mask，dropout_p=self.dropout表示使用指定的dropout率，is_causal=True表示使用因果注意力（不看未来token）
            output=torch.nn.functional.scaled_dot_product_attention(xq,xk,xv,attn_mask=None,dropout_p=self.dropout if self.dropout else 0.0,is_causal=True)
        else:
            scores=torch.matmul(xq,xk.transpose(-2,-1))/math.sqrt(self.head_dim)
            assert hasattr(self,'mask')
            scores=scores+self.mask[:,:,:seq_len,:seq_len]
            scores=F.softmax(scores,dim=-1)
            scores=self.attn_dropout(scores)
            output=torch.matmul(scores,xv)

        output=output.transpose(1,2).contiguous().view(bsz,seq_len,-1)

        # 最终投影回残差流（重新混合所有信息）
        output=self.wo(output)
        output=self.resid_dropout(output)
        return output
    
#--------------------------------------------------------
#--------------------------------------------------------
# 构建MLP
#--------------------------------------------------------
#--------------------------------------------------------

class MLP(nn.Module):
    def __init__(self,dim:int,hidden_dim:int,multiple_of:int,dropout:float):
        super().__init__()
        # 不指定隐藏层维度就设置为4倍的dim
        # 将其减少到2/3，然后确保是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim=4*dim
            hidden_dim=int(hidden_dim*2/3)
            hidden_dim=multiple_of*int((hidden_dim+multiple_of-1)//multiple_of)

        self.w1=nn.Linear(dim,hidden_dim,bias=False)
        self.w2=nn.Linear(hidden_dim,dim,bias=False)
        self.w3=nn.Linear(dim,hidden_dim,bias=False)
        
        # 定义dropout
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x:torch.Tensor):
        return self.dropout(self.w2(F.silu(self.w1(x))*self.w3(x)))
    
#--------------------------------------------------------
#--------------------------------------------------------
# 构建decoder block
#--------------------------------------------------------
#--------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self,layer_id:int,args:ModelConfig):
        super().__init__()

        self.n_heads=args.n_heads
        self.dim=args.dim
        self.head_dim=args.dim//args.n_heads
        self.layer_id=layer_id

        self.RMSNorm1=RMSNorm(args.dim,args.norm_eps)
        self.attention=Attention(args)
        self.RMSNorm2=RMSNorm(args.dim,args.norm_eps)
        self.mlp=MLP(args.dim,args.hidden_dim,args.multiple_of,args.dropout)

    def forward(self,x:torch.Tensor,freqs_cos:torch.Tensor,freqs_sin:torch.Tensor):
        _x=x
        x=self.RMSNorm1(x)
        x=self.attention(x,freqs_cos,freqs_sin)
        x=x+_x
        _x=x

        x=self.RMSNorm2(x)
        x=self.mlp(x)
        x=x+_x
        return x
    
#--------------------------------------------------------
#--------------------------------------------------------
# 构建LLaMA2模型，即将所有的decoder block堆叠起来
#--------------------------------------------------------
#--------------------------------------------------------

class Transformer(PreTrainedModel):
    config_class=ModelConfig# 配置类
    last_loss:Optional[torch.Tensor]# 最后一次的损失值

    def __init__(self,args:ModelConfig):
        super().__init__(args)
        self.args=args
        self.vocab_size=args.vocab_size
        self.n_layers=args.n_layers

        # 层定义
        # embedding层
        self.tok_embeddings=nn.Embedding(args.vocab_size,args.dim)
        # dropout层
        self.dropout=nn.Dropout(args.dropout)
        # decoder层
        self.layers=torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id,args))
        # norm层
        self.norm=RMSNorm(args.dim,args.norm_eps)
        # 输出层
        self.output=nn.Linear(args.dim,args.vocab_size,bias=False)

        # 参数计算
        # 将词嵌入的权重与输出层共享
        self.tok_embeddings.weight=self.output.weight
        freqs_cos,freqs_sin=precompute_freqs_cis(self.args.dim//self.args.n_heads,args.max_seq_len)
        # register_buffer用于存放不需要学习的tensor
        self.register_buffer("freqs_cos",freqs_cos,False)
        self.register_buffer("freqs_sin",freqs_sin,False)

        # 初始化权重
        self.apply(self.__init__weights)
        # 对残差投影进行特殊的缩放初始化
        for pn,p in self.named_parameters():
            if pn.endswith("w2.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(p,mean=0.0,std=0.02/math.sqrt(2*args.n_layers))

        # 初始化最后一次前向传播的损失属性
        self.last_loss=None
        self.OUT=CausalLMOutputWithPast() # 输出容器
        self._no_split_modules=[name for name,_ in self.named_modules()] # 不分割的模块列表

    def __init__weights(self,module):
        # 初始化权重
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,tokens:torch.Tensor,targets:Optional[torch.Tensor]=None,**kwargs)->torch.Tensor:
        """
        - tokens: Optional[torch.Tensor], 输入 token 张量。
        - targets: Optional[torch.Tensor], 目标 token 张量。
        - kv_cache: bool, 是否使用键值缓存。
        - kwargs: 其他关键字参数。

        - self.OUT: CausalLMOutputWithPast, 包含 logits 和损失。
        
        """

        if "input_ids" in kwargs:
            tokens=kwargs["input_ids"]
        if "labels" in kwargs:
            targets=kwargs["labels"]

        # 前向传播函数
        _bsz,seq_len=tokens.shape
        # 通过embedding和dropout层
        h=self.tok_embeddings(tokens)
        h=self.dropout(h)
        # 获取相对位置的freqs
        freqs_cos=self.freqs_cos[:seq_len]
        freqs_sin=self.freqs_sin[:seq_len]

        # 通过decoder层
        for layer in self.layers:
            h=layer(h,freqs_cos,freqs_sin)
        h=self.norm(h)

        if targets is not None:
            # 如果给定了目标，则计算损失
            logits=self.output(h)
            self.last_loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=0,reduction="none")
        else:
            # 推理时的小优化，只对最后一个位置前向传播
            logits=self.output(h[:,[-1],:])
            self.last_loss=None

        # 设置输出
        self.OUT.__setitem__("logits",logits)
        self.OUT.__setitem__("last_loss",self.last_loss)
        return self.OUT

    @torch.inference_mode()
    def generate(self,idx,stop_id=None,max_new_tokens=256,temperature=1.0,top_k=None):
        """
        给定输入序列idx[bsz,seq_len]长整型向量,通过多次生成新token来扩展序列
        在 model.eval() 模式下使用,效率较低的版本,没有使用kv cache
        """
        index=idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果上下文序列过长，截断他到最大长度
            idx_cond=idx if idx.shape[1]<=self.args.max_seq_len else idx[:,-self.args.max_seq_len:]

            # 前向传播获取序列中最后一个位置的logits
            logits=self(idx_cond).logits
            logits=logits[:,-1,:]# 只保留最后一个时间步的输出

            if temperature==0.0:
                # 选择最有可能的索引
                _,idx_next=torch.topk(logits,1,dim=-1)
            else:
                # 缩放logits并使用softmax
                logits=logits/temperature
                if top_k is not None:
                    v,_=torch.topk(logits,min(top_k,logits.size(-1)))
                    logits[logits<v[:,[-1]]]=-float("Inf")
                probs=F.softmax(logits,dim=-1)
                idx_next=torch.multinomial(probs,num_samples=1)
            

            if stop_id is not None:
                finished=idx_next.squeeze(-1).eq(stop_id)
                if bool(finished.all()):
                    break

            # 将新生成的token拼接到输入序列中
            idx=torch.cat((idx,idx_next),dim=1)

        return idx[:,index:]# 只返回新生成的token
