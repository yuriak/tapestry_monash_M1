"""Exact native-collection ranking; no relevance judgements enter retrieval."""
from tapestry_m1.io import ContractError


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


def reorder(original,scores):
    import math
    if len(original)!=len(scores) or len({r["doc_id"] for r in original})!=len(original):raise ContractError("reranker candidate count/identity mismatch")
    if any(isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s) for s in scores):raise ContractError("nonfinite/invalid reranker score")
    pairs=sorted(zip(original,scores),key=lambda x:(-x[1],x[0]["doc_id"]))
    output=[{"doc_id":old["doc_id"],"rank":i,"score":float(score),"retrieval_rank":old["rank"],"retrieval_score":old["score"]} for i,(old,score) in enumerate(pairs,1)]
    if {r["doc_id"] for r in output}!={r["doc_id"] for r in original}:raise ContractError("reranking changed candidate set")
    return output


def check_candidates(original,reranked):
    from tapestry_m1.data.schema import ranking,query_key
    def index(rows):
        out={}
        for row in rows:
            ranking(row);key=query_key(row)
            if key in out:raise ContractError("duplicate query ranking")
            out[key]=row
        return out
    a,b=index(original),index(reranked)
    if not a or set(a)!=set(b):raise ContractError("query sets differ")
    for key in a:
        if a[key]["collection_id"]!=b[key]["collection_id"] or {r["doc_id"] for r in a[key]["candidates"]}!={r["doc_id"] for r in b[key]["candidates"]}:raise ContractError("reranker changed eligible candidate set")
    return {"queries":len(a),"candidate_sets_identical":True}
