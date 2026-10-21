"""Qwen3-Embedding query formatting, pooling and encoding.
Source: https://huggingface.co/Qwen/Qwen3-Embedding-4B
"""
from tapestry_m1.core.io import ContractError
from tapestry_m1.runtime.transformers import LocalTransformer


def embedding_query(instruction,query):
    return f"Instruct: {instruction}\nQuery:{query}"


def last_token_pool(hidden,mask):
    import torch
    if mask.shape[1]!=hidden.shape[1] or (mask.sum(dim=1)==0).any():raise ContractError("invalid attention mask")
    if bool(mask[:,-1].all()):return hidden[:,-1]
    positions=torch.arange(mask.shape[1],device=mask.device).expand_as(mask).masked_fill(mask==0,-1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0],device=hidden.device),positions]


class Qwen3Encoder(LocalTransformer):
    def __init__(self, model_dir, config):
        super().__init__(model_dir, config, causal_lm=False)

    def encode(self,texts,is_query=False,batch_size=None):
        import numpy as np
        torch=self.torch;vectors=[];audit=[];batch_size=batch_size or self.config["embedding_batch"]
        for start in range(0,len(texts),batch_size):
            batch=texts[start:start+batch_size]
            formatted=[embedding_query(self.config["instruction"],t) for t in batch] if is_query else batch
            lengths=[len(self.tokenizer.encode(t,add_special_tokens=True)) for t in formatted]
            audit.extend({"tokens":n,"truncated":n>self.limit,"retained_tokens":min(n,self.limit)} for n in lengths)
            inputs=self.tokenizer(formatted,padding=True,truncation=True,max_length=self.limit,return_tensors="pt").to("cuda")
            with torch.inference_mode():
                hidden=self.model(**inputs,use_cache=False).last_hidden_state
                out=torch.nn.functional.normalize(last_token_pool(hidden,inputs["attention_mask"]).float(),p=2,dim=1)
            vectors.append(out.cpu().numpy())
            del inputs,hidden,out
        return np.concatenate(vectors),audit
