"""Site-neutral contracts for the IL-PCSR Qwen data-protocol adaptation."""
import json
import os
from pathlib import Path
import numpy as np

INSTRUCTIONS = {
    'lsr': 'Retrieve the statutory provision relevant to the legal issues in this masked Indian court judgment.',
    'pcr': 'Retrieve a precedent relevant to the legal reasoning in this masked Indian court judgment.',
}
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n'); os.replace(tmp, path)

def load(path):
    return json.loads(Path(path).read_text())

def trim(tokens, budget):
    """Deterministic head/tail; no generated summaries or label-guided selection."""
    if len(tokens) <= budget: return list(tokens)
    head = (budget + 1)//2
    return list(tokens[:head]) + (list(tokens[-(budget-head):]) if budget > head else [])

def pair_ids(q, d, prefix, middle, suffix, limit=8192):
    available = limit-len(prefix)-len(middle)-len(suffix)
    # Balanced initial allocation; unused short-document budget goes to the query.
    q_budget = min(len(q), max(available//2, available-len(d)))
    d_budget = available-q_budget
    return prefix+trim(q,q_budget)+middle+trim(d,d_budget)+suffix

def batches(items, length, tokens, maximum=64):
    chunk=[]; longest=0
    for item in items:
        size=length(item); proposed=max(longest,size)
        if chunk and (len(chunk)>=maximum or proposed*(len(chunk)+1)>tokens):
            yield chunk; chunk=[]; longest=0
        chunk.append(item); longest=max(longest,size)
    if chunk: yield chunk

def query_metrics(ranked, gold):
    gold=set(gold); hits=np.array([int(x in gold) for x in ranked], dtype=float)
    n=len(gold); cumulative=np.cumsum(hits)
    f=[2*float(cumulative[min(k,len(hits))-1])/(k+n) if len(hits) else 0. for k in range(1,11)]
    ap=float(np.sum(cumulative*hits/np.arange(1,len(hits)+1))/n) if n else 0.
    positive=np.flatnonzero(hits)
    return {'f1':f, 'ap':ap, 'rr':float(1/(positive[0]+1)) if len(positive) else 0.,
            'recall':float(hits.sum()/n) if n else 0., 'has_gold':bool(n),
            'any_hit':bool(hits.sum()), 'all_hit':bool(n and hits.sum()==n)}

def metrics(candidates, scores, gold, k=None):
    per={}
    for q, ids in candidates.items():
        values=np.asarray(scores[q]); assert len(ids)==len(values) and np.isfinite(values).all()
        ranked=[ids[i] for i in np.argsort(-values,kind='stable')]
        per[q]=query_metrics(ranked,gold[q])
    f=np.mean([v['f1'] for v in per.values()],axis=0)
    if k is None: k=int(np.argmax(f))+1
    eligible=[v for v in per.values() if v['has_gold']]
    summary={'k':k,'macro_f1':float(f[k-1]),'f1_by_k':f.tolist(),
             'map':float(np.mean([v['ap'] for v in per.values()])),
             'mrr':float(np.mean([v['rr'] for v in per.values()])),
             'queries':len(per),'gold_queries':len(eligible),
             'recall_gold_queries':float(np.mean([v['recall'] for v in eligible])),
             'any_hit_gold_queries':float(np.mean([v['any_hit'] for v in eligible]))}
    return summary,per

def positive_groups(gold, epoch, epochs):
    """Each query appears once/epoch; rotate all positives within the full budget."""
    width=max(1,(len(gold)+epochs-1)//epochs)
    return [gold[((epoch-1)*width+j)%len(gold)] for j in range(width)]

def paired_ci(a,b,k_a,k_b,seed=42):
    ids=sorted(set(a)&set(b)); rng=np.random.default_rng(seed)
    delta=np.array([a[q]['f1'][k_a-1]-b[q]['f1'][k_b-1] for q in ids])
    draws=np.concatenate([delta[rng.integers(0,len(delta),(1000,len(delta)))].mean(1) for _ in range(10)])
    return {'difference':float(delta.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'queries':len(ids),'bootstrap':10000}
