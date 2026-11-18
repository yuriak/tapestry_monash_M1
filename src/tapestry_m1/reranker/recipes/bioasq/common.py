"""Small deterministic IO and query-level ranking metrics for mini BioASQ."""
import json, math, os
from pathlib import Path
import numpy as np

def load(path):
    with open(path) as f: return json.load(f)

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)); tmp.replace(path)

def metrics(candidates, scores, gold):
    per = {}
    for q, docs in candidates.items():
        relevant = set(gold[q]); n = len(relevant)
        order = sorted(range(len(docs)), key=lambda i: -scores[q][i])
        ranked = [docs[i] for i in order]
        hits = [int(d in relevant) for d in ranked]
        ap = sum(sum(hits[:i+1]) / (i+1) for i in range(min(10, len(hits))) if hits[i])
        dcg = sum(h / math.log2(i+2) for i,h in enumerate(hits[:10]))
        ideal = sum(1 / math.log2(i+2) for i in range(min(n, 10)))
        per[q] = {'map10': ap / min(n,10) if n else 0.,
                  'ndcg10': dcg / ideal if ideal else 0.,
                  'mrr10': next((1/(i+1) for i,h in enumerate(hits[:10]) if h), 0.)}
        for k in [10,100,500,1000]: per[q][f'recall{k}'] = sum(hits[:k])/n if n else 0.
    return {k: float(np.mean([r[k] for r in per.values()])) for k in next(iter(per.values()))}, per

def views(candidates, scores, gold):
    result = {}
    for k in [100,500,1000]:
        cs = {q:ds[:k] for q,ds in candidates.items()}
        ss = {q:vs[:k] for q,vs in scores.items()}
        result[str(k)] = metrics(cs,ss,gold)[0]
    return result
