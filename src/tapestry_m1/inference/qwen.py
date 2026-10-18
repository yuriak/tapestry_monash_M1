"""Official Qwen3 last-token embeddings and yes/no reranker recipes.
Sources: https://huggingface.co/Qwen/Qwen3-Embedding-4B
         https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
"""
from pathlib import Path
from tapestry_m1.io import ContractError


def embedding_query(instruction,query):
    return f"Instruct: {instruction}\nQuery:{query}"


def reranker_text(instruction,query,document):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def last_token_pool(hidden,mask):
    import torch
    if mask.shape[1]!=hidden.shape[1] or (mask.sum(dim=1)==0).any():raise ContractError("invalid attention mask")
    if bool(mask[:,-1].all()):return hidden[:,-1]
    positions=torch.arange(mask.shape[1],device=mask.device).expand_as(mask).masked_fill(mask==0,-1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0],device=hidden.device),positions]


def yes_probability(logits,no_id,yes_id):
    import torch
    if no_id==yes_id or min(no_id,yes_id)<0 or max(no_id,yes_id)>=logits.shape[-1]:raise ContractError("invalid yes/no token IDs")
    selected=logits[..., [no_id,yes_id]].float()
    if not torch.isfinite(selected).all():raise ContractError("nonfinite yes/no logits")
    return torch.softmax(selected,dim=-1)[...,1]


class QwenAdapter:
    def __init__(self,model_dir,role,config):
        import torch
        from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer
        self.torch=torch;self.role=role;self.config=config
        self.tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False,padding_side="left")
        cls=AutoModel if role=="embedding" else AutoModelForCausalLM
        dtype={"bfloat16":torch.bfloat16,"float16":torch.float16,"float32":torch.float32}[config["dtype"]]
        self.model=cls.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False,torch_dtype=dtype,attn_implementation=config["attention_backend"]).eval().to("cuda")
        self.limit=config["max_tokens"]
        if self.limit<16 or self.limit>self.model.config.max_position_embeddings:raise ContractError("token limit outside model context")
        if role=="reranker":
            self.prefix=self.tokenizer.encode(PREFIX,add_special_tokens=False);self.suffix=self.tokenizer.encode(SUFFIX,add_special_tokens=False)
            self.no=self.tokenizer.convert_tokens_to_ids("no");self.yes=self.tokenizer.convert_tokens_to_ids("yes")
            if self.tokenizer.encode("no",add_special_tokens=False)!=[self.no] or self.tokenizer.encode("yes",add_special_tokens=False)!=[self.yes] or self.no==self.yes:raise ContractError("yes/no are not distinct single tokens")
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
    def score(self,queries,documents,batch_size=None):
        torch=self.torch;all_scores=[];audit=[];batch_size=batch_size or self.config["reranker_batch"]
        if len(queries)!=len(documents):raise ContractError("query/document pair sizes differ")
        budget=self.limit-len(self.prefix)-len(self.suffix)
        if budget<1:raise ContractError("reranker prompt exceeds token budget")
        for start in range(0,len(queries),batch_size):
            pairs=[reranker_text(self.config["instruction"],q,d) for q,d in zip(queries[start:start+batch_size],documents[start:start+batch_size])]
            tokens=self.tokenizer(pairs,padding=False,truncation=False,add_special_tokens=True)["input_ids"]
            audit.extend({"tokens":len(t)+len(self.prefix)+len(self.suffix),"truncated":len(t)>budget,"retained_tokens":min(len(t),budget)+len(self.prefix)+len(self.suffix)} for t in tokens)
            inputs=self.tokenizer.pad({"input_ids":[self.prefix+t[:budget]+self.suffix for t in tokens]},padding=True,return_tensors="pt").to("cuda")
            with torch.inference_mode():
                logits=self.model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,:]
                scores=yes_probability(logits,self.no,self.yes).cpu().tolist()
            all_scores.extend(scores);del inputs,logits
        return all_scores,audit
    def close(self):
        import gc
        del self.model;gc.collect();self.torch.cuda.empty_cache()
