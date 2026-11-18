"""Native IL metrics, with matched Top100/200 views."""
from .il_metrics import load,dump,metrics as native_metrics

def metrics(candidates,scores,gold):return native_metrics(candidates,scores,gold)
def views(candidates,scores,gold):
    return {str(k):native_metrics({q:ds[:k] for q,ds in candidates.items()},{q:vs[:k] for q,vs in scores.items()},gold)[0] for k in [100,200]}
