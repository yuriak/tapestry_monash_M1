"""Candidate-preserving reranking and rank lineage."""
from tapestry_m1.core.io import ContractError


def reorder(original,scores):
    import math
    if len(original)!=len(scores) or len({r["doc_id"] for r in original})!=len(original):raise ContractError("reranker candidate count/identity mismatch")
    if any(isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s) for s in scores):raise ContractError("nonfinite/invalid reranker score")
    pairs=sorted(zip(original,scores),key=lambda x:(-x[1],x[0]["doc_id"]))
    output=[{"doc_id":old["doc_id"],"rank":i,"score":float(score),"retrieval_rank":old["rank"],"retrieval_score":old["score"]} for i,(old,score) in enumerate(pairs,1)]
    if {r["doc_id"] for r in output}!={r["doc_id"] for r in original}:raise ContractError("reranking changed candidate set")
    return output


def check_candidates(original,reranked):
    from tapestry_m1.contracts.records import ranking,query_key
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
