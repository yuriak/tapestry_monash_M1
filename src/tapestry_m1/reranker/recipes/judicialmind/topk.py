"""Summarize nested original retrieval pools, preserving matched model comparisons."""
import numpy as np
from .metrics import evaluate

def summarize(frozen,scores,countries,ks=(100,200,500),baseline_scores=None):
    result={}
    for k in ks:
        candidates={q:ds[:k] for q,ds in frozen['candidates'].items()}
        cut={q:scores[q][:len(ds)] for q,ds in candidates.items()}
        _,per=evaluate(candidates,cut,frozen['gold'])
        baseline_per=None
        if baseline_scores is not None:
            _,baseline_per=evaluate(candidates,{q:baseline_scores[q][:len(ds)] for q,ds in candidates.items()},frozen['gold'])
        oracle={}
        for q,ds in candidates.items():
            g=frozen['gold'][q];ideal=sorted(g.values(),reverse=True)[:10];available=sorted([g.get(d,0) for d in ds],reverse=True)[:10]
            denom=sum((2**v-1)/np.log2(i+2) for i,v in enumerate(ideal))
            oracle[q]=sum((2**v-1)/np.log2(i+2) for i,v in enumerate(available))/denom if denom else 0.
        result[str(k)]={}
        for country in ['mixed','au','in']:
            qs=[q for q in candidates if country=='mixed' or countries[q]==country]
            if not qs:continue
            result[str(k)][country]={'queries':len(qs),**{m:float(np.mean([per[q][m] for q in qs])) for m in ['ndcg10','mrr','recall']},'oracle_ndcg10':float(np.mean([oracle[q] for q in qs]))}
            if baseline_per is not None:
                delta=[per[q]['ndcg10']-baseline_per[q]['ndcg10'] for q in qs]
                result[str(k)][country].update(ndcg_delta=float(np.mean(delta)),improved=sum(d>1e-10 for d in delta),worsened=sum(d< -1e-10 for d in delta),tied=sum(abs(d)<=1e-10 for d in delta))
    return result
