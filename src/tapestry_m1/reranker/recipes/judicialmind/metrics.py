"""Graded nDCG@10 with complete qrels denominator, including missing candidates."""
import numpy as np

def one(ranked,gold):
    values=np.array([max(0,gold.get(d,0)) for d in ranked],dtype=float)
    ideal=np.sort(np.array([max(0,x) for x in gold.values()],dtype=float))[::-1][:10]
    denom=np.sum((2**ideal-1)/np.log2(np.arange(len(ideal))+2))
    dcg=np.sum((2**values[:10]-1)/np.log2(np.arange(min(10,len(values)))+2))
    hits=values>0;positive=np.flatnonzero(hits);n=sum(v>0 for v in gold.values())
    return {'ndcg10':float(dcg/denom) if denom else 0.,'mrr':float(1/(positive[0]+1)) if len(positive) else 0.,
      'recall':float(hits.sum()/n) if n else 0.}
def evaluate(candidates,scores,gold):
    per={}
    for q,ids in candidates.items():
        v=np.asarray(scores[q]);assert len(v)==len(ids) and np.isfinite(v).all()
        per[q]=one([ids[i] for i in np.argsort(-v,kind='stable')],gold[q])
    return {k:float(np.mean([x[k] for x in per.values()])) for k in ['ndcg10','mrr','recall']},per

def paired(a,b):
    ids=sorted(a);delta=np.array([a[q]['ndcg10']-b[q]['ndcg10'] for q in ids]);rng=np.random.default_rng(42)
    samples=np.concatenate([delta[rng.integers(0,len(ids),(1000,len(ids)))].mean(1) for _ in range(10)])
    return {'ndcg10_difference':float(delta.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),'resampling':'query, within seed','replicates':10000}
