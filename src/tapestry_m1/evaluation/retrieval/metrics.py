"""Explicit graded-gain nDCG and judged-relevance metrics."""
import math
from tapestry_m1.core.io import ContractError
from tapestry_m1.contracts.records import number


def metrics(doc_ids, relevance, gain="exponential", mrr_depth=None):
    if len(doc_ids) != len(set(doc_ids)):
        raise ContractError("duplicate retrieved documents")
    if gain not in ["exponential", "linear"]:
        raise ContractError("unknown gain convention")
    if mrr_depth is not None and (type(mrr_depth) is not int or mrr_depth <= 0):
        raise ContractError("MRR depth must be positive or null for submitted ranking")
    for v in relevance.values():
        number(v, "relevance", nonnegative=True)
        if gain == "exponential" and v > 50:
            raise ContractError("exponential relevance grade too large; review label semantics")
    relevant = {d for d, r in relevance.items() if r > 0}
    if not relevant:
        raise ContractError("no positive judgements; metric denominator undefined")
    gain_fn = (lambda r: 2.0 ** r - 1.0) if gain == "exponential" else float
    actual = sum(gain_fn(relevance.get(d, 0)) / math.log2(i + 2) for i, d in enumerate(doc_ids[:10]))
    ideal = sum(gain_fn(r) / math.log2(i + 2) for i, r in enumerate(sorted(relevance.values(), reverse=True)[:10]))
    rr = next((1.0 / i for i, d in enumerate(doc_ids[:mrr_depth] if mrr_depth else doc_ids, 1) if d in relevant), 0.0)
    return {"ndcg@10": actual / ideal, "recall@10": len(relevant.intersection(doc_ids[:10])) / len(relevant),
            "recall@100": len(relevant.intersection(doc_ids[:100])) / len(relevant),
            "mrr" if mrr_depth is None else f"mrr@{mrr_depth}": rr}
