"""Qwen scoring with causal right padding, suitable for a DDP wrapper."""
from array import array
import pickle
from pathlib import Path
import torch
import numpy as np
from torch import nn
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,TaskType,get_peft_model,PeftModel

PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def trim(ids,n):
    if len(ids)<=n:return list(ids)
    h=(n+1)//2;return list(ids[:h])+(list(ids[-(n-h):]) if n>h else [])
def pack(rows,limit,maximum=256):
    batch=[];longest=0
    for row in rows:
        size=len(row[-1]);new=max(longest,size)
        if batch and (new*(len(batch)+1)>limit or len(batch)>=maximum):yield batch;batch=[];longest=0
        batch.append(row);longest=max(longest,size)
    if batch:yield batch

class Tokens:
    def __init__(self,path,cache,instruction):
        self.tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='right')
        self.prefix=self.tok.encode(PREFIX+f'<Instruct>: {instruction}\n<Query>: ',add_special_tokens=False)
        self.middle=self.tok.encode('\n<Document>: ',add_special_tokens=False);self.suffix=self.tok.encode(SUFFIX,add_special_tokens=False)
        with open(cache,'rb') as f:self.cache=pickle.load(f)
    def pair(self,q,d,limit=2048):
        qt=self.cache['queries'][q];dt=self.cache['documents'][d]
        budget=limit-len(self.prefix)-len(self.middle)-len(self.suffix)
        qb=min(len(qt),max(budget//2,budget-len(dt)))
        return array('I',self.prefix+trim(qt,qb)+self.middle+trim(dt,budget-qb)+self.suffix)
    def tensors(self,rows,device):
        lengths=np.asarray([len(r) for r in rows],dtype=np.int64)
        padded=np.full((len(rows),int(lengths.max())),self.tok.pad_token_id,dtype=np.int64)
        for i,row in enumerate(rows):padded[i,:len(row)]=row
        return torch.from_numpy(padded).to(device),torch.from_numpy(lengths).to(device)

class Ranker(nn.Module):
    def __init__(self,path,train=False,adapter=None,checkpointing=False):
        super().__init__()
        base=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
        base.config.use_cache=False
        if adapter:base=PeftModel.from_pretrained(base,str(adapter),is_trainable=train)
        elif train:base=get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','v_proj']))
        if train and checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});base.enable_input_require_grads()
        self.base=base
        tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.yes=tok.convert_tokens_to_ids('yes');self.no=tok.convert_tokens_to_ids('no')
    def forward(self,ids,lengths):
        base=self.base.get_base_model() if hasattr(self.base,'get_base_model') else self.base
        h=base.model(input_ids=ids,use_cache=False).last_hidden_state
        last=h[torch.arange(len(ids),device=ids.device),lengths-1];logits=base.lm_head(last)
        return logits[:,self.yes].float()-logits[:,self.no].float()
