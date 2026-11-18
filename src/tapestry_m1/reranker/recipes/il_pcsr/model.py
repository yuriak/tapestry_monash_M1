"""Independent single-GPU task/loss workers with cached tokens and resumable epochs."""
import argparse
import gc
import math
import pickle
import random
import time
from pathlib import Path
import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from .common import INSTRUCTIONS, PREFIX, SUFFIX, batches, dump, load, metrics, pair_ids, paired_ci, positive_groups

def causal_scores(model,rows,pad_id,yes,no):
    # Right padding consists only of FUTURE positions. Causal attention means it
    # cannot affect the last real token; no arbitrary padding mask is needed.
    # Select real-token hidden states BEFORE the large vocabulary projection.
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    device=next(base.parameters()).device
    lengths=torch.tensor([len(r) for r in rows],device=device)
    ids=torch.full((len(rows),int(lengths.max())),pad_id,dtype=torch.long,device=device)
    for i,row in enumerate(rows): ids[i,:len(row)]=torch.as_tensor(list(row),dtype=torch.long,device=device)
    hidden=base.model(input_ids=ids,use_cache=False).last_hidden_state
    last=hidden[torch.arange(len(rows),device=device),lengths-1]
    logits=base.lm_head(last)
    return logits[:,yes].float()-logits[:,no].float()

class Scorer:
    def __init__(self,a,task):
        self.a=a; self.task=task; self.gradient_checkpointing=True; self.tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,padding_side='left')
        self.yes=self.tok.convert_tokens_to_ids('yes'); self.no=self.tok.convert_tokens_to_ids('no')
        self.prefix=self.tok.encode(PREFIX+f'<Instruct>: {INSTRUCTIONS[task]}\n<Query>: ',add_special_tokens=False)
        self.middle=self.tok.encode('\n<Document>: ',add_special_tokens=False); self.suffix=self.tok.encode(SUFFIX,add_special_tokens=False)
        with open(Path(a.data)/'tokens.pkl','rb') as f: self.tokens=pickle.load(f)
        self.cache={}; self.audit={'pairs':0,'query_truncated':0,'document_truncated':0}
    def ids(self,q,d):
        key=(q,d)
        if key not in self.cache:
            from array import array
            qt=self.tokens['queries'][q]; dt=self.tokens['documents'][self.task][d]
            row=pair_ids(qt,dt,self.prefix,self.middle,self.suffix)
            available=8192-len(self.prefix)-len(self.middle)-len(self.suffix)
            qb=min(len(qt),max(available//2,available-len(dt)))
            self.audit['pairs']+=1; self.audit['query_truncated']+=len(qt)>qb; self.audit['document_truncated']+=len(dt)>available-qb
            self.cache[key]=array('I',row)
        return self.cache[key]
    def model(self,train=False,adapter=None):
        m=AutoModelForCausalLM.from_pretrained(self.a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
        m.config.use_cache=False
        if adapter: m=PeftModel.from_pretrained(m,str(adapter),is_trainable=train)
        elif train: m=get_peft_model(m,LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','v_proj']))
        if train and self.gradient_checkpointing:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); m.enable_input_require_grads()
        return m
    def score(self,m,rows):
        return causal_scores(m,rows,self.tok.pad_token_id,self.yes,self.no)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

