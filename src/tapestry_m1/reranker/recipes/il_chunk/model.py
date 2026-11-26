"""Qwen scoring with causal right padding, suitable for a DDP wrapper."""
from array import array
import pickle
from pathlib import Path
import torch
import numpy as np
from torch import nn
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,TaskType,get_peft_model,PeftModel

from ..scoring import Ranker, PREFIX, SUFFIX, trim, pack

def merge_ranges(ranges):
    result=[]
    for start,end in sorted(set(ranges)):
        if result and start<=result[-1][1]:result[-1]=(result[-1][0],max(end,result[-1][1]))
        else:result.append((start,end))
    return result

class Tokens:
    def __init__(self,path,data_dir):
        from .common import load
        from .il_metrics import INSTRUCTIONS
        data_dir=Path(data_dir);self.task=data_dir.name
        self.tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='right')
        self.prefix=self.tok.encode(PREFIX+f'<Instruct>: {INSTRUCTIONS[self.task]}\n<Query>: ',add_special_tokens=False)
        self.middle=self.tok.encode('\n<Document>: ',add_special_tokens=False);self.suffix=self.tok.encode(SUFFIX,add_special_tokens=False);self.sep=self.tok.encode('\n...\n',add_special_tokens=False)
        with open(data_dir.parent/'chunks.pkl','rb') as f:obj=pickle.load(f)
        self.cache=obj['tokens'];self.spans=obj['spans']
        data=load(data_dir/'data.json');self.qindex={q:i for i,q in enumerate(data['query_ids'])};self.dindex={d:i for i,d in enumerate(data['document_ids'])}
        self.selection=np.load(data_dir/'selection.npy',mmap_mode='r')
    def selected(self,ids,spans,indices):
        ranges=merge_ranges([spans[int(i)] for i in indices]);out=[]
        for start,end in ranges:
            if out:out+=self.sep
            out+=list(ids[start:end])
        return out
    def pair(self,q,d,limit=8192):
        sel=self.selection[self.qindex[q],self.dindex[d]]
        qt=self.selected(self.cache['queries'][q],self.spans['queries'][q],sel[:2])
        dt=self.selected(self.cache['documents'][self.task][d],self.spans[self.task][d],sel[2:])
        budget=limit-len(self.prefix)-len(self.middle)-len(self.suffix)
        qb=min(len(qt),max(budget//2,budget-len(dt)))
        return array('I',self.prefix+trim(qt,qb)+self.middle+trim(dt,budget-qb)+self.suffix)
    def longest_pair(self):
        # Synthetic maximum shape for bounded memory/throughput probes, not training.
        return array('I',self.prefix+([self.tok.encode('law',add_special_tokens=False)[0]]*(8192-len(self.prefix)-len(self.suffix)))+self.suffix)
    def tensors(self,rows,device):
        lengths=np.asarray([len(r) for r in rows],dtype=np.int64)
        padded=np.full((len(rows),int(lengths.max())),self.tok.pad_token_id,dtype=np.int64)
        for i,row in enumerate(rows):padded[i,:len(row)]=row
        return torch.from_numpy(padded).to(device),torch.from_numpy(lengths).to(device)

