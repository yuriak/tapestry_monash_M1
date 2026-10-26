"""Qwen3-Reranker prompt and yes/no scoring.
Source: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
"""
from tapestry_m1.core.io import ContractError
from tapestry_m1.runtime.transformers import LocalTransformer


def reranker_text(instruction,query,document):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def yes_probability(logits,no_id,yes_id):
    import torch
    if no_id==yes_id or min(no_id,yes_id)<0 or max(no_id,yes_id)>=logits.shape[-1]:raise ContractError("invalid yes/no token IDs")
    selected=logits[..., [no_id,yes_id]].float()
    if not torch.isfinite(selected).all():raise ContractError("nonfinite yes/no logits")
    return torch.softmax(selected,dim=-1)[...,1]


def yes_logit_difference(logits, no_id, yes_id):
    import torch
    # Share the existing token-ID and finiteness validation.
    yes_probability(logits, no_id, yes_id)
    result = logits[..., yes_id].float() - logits[..., no_id].float()
    if not torch.isfinite(result).all():
        raise ContractError("nonfinite yes/no logit difference")
    return result


class Qwen3Reranker(LocalTransformer):
    def __init__(self, model_dir, config):
        super().__init__(model_dir, config, causal_lm=True)
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.no = self.tokenizer.convert_tokens_to_ids("no")
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        if self.tokenizer.encode("no", add_special_tokens=False) != [self.no] or self.tokenizer.encode("yes", add_special_tokens=False) != [self.yes] or self.no == self.yes:
            raise ContractError("yes/no are not distinct single tokens")

    def score(self,queries,documents,batch_size=None):
        return self._score(queries, documents, batch_size, raw=False)

    def score_logits(self, queries, documents, batch_size=None):
        """Uncalibrated yes-minus-no logits for margin-aware negative selection."""
        return self._score(queries, documents, batch_size, raw=True)

    def _score(self,queries,documents,batch_size=None,*,raw=False):
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
                scores=(yes_logit_difference(logits,self.no,self.yes) if raw else yes_probability(logits,self.no,self.yes)).cpu().tolist()
            all_scores.extend(scores);del inputs,logits
        return all_scores,audit
