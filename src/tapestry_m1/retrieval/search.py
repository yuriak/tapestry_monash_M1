"""Exact native-collection ranking; no relevance judgements enter retrieval."""
from tapestry_m1.core.io import ContractError


def exact_topk(doc_ids,document_vectors,query_vectors,k=100):
    import numpy as np
    if type(k) is not int or k<1:raise ContractError("k must be positive")
    if len(doc_ids)!=len(set(doc_ids)) or not doc_ids:raise ContractError("candidate IDs must be nonempty and unique")
    try:docs=np.asarray(document_vectors,dtype=np.float32);queries=np.asarray(query_vectors,dtype=np.float32)
    except (ValueError,TypeError) as exc:raise ContractError("vectors must be numeric") from exc
    if docs.ndim!=2 or queries.ndim!=2 or docs.shape[0]!=len(doc_ids) or docs.shape[1]!=queries.shape[1] or not docs.shape[1]:raise ContractError("vector dimensions/IDs mismatch")
    if not np.isfinite(docs).all() or not np.isfinite(queries).all():raise ContractError("nonfinite vectors")
    dn=np.linalg.norm(docs,axis=1);qn=np.linalg.norm(queries,axis=1)
    if (dn==0).any() or (qn==0).any():raise ContractError("zero-norm vectors")
    scores=(queries/qn[:,None])@(docs/dn[:,None]).T
    if not np.isfinite(scores).all():raise ContractError("nonfinite similarity")
    names=np.asarray(doc_ids);result=[]
    for row in scores:
        order=np.lexsort((names,-row))[:min(k,len(doc_ids))]
        result.append([{"doc_id":doc_ids[int(i)],"rank":rank,"score":float(row[i])} for rank,i in enumerate(order,1)])
    return result
