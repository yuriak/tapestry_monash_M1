"""Query-paired percentile intervals; never pool run seeds."""
from collections import defaultdict
import hashlib
import math
import random
from tapestry_m1.io import ContractError, dumps
from tapestry_m1.data.schema import number


def percentile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    lo = math.floor(x)
    hi = math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def paired_compare(a, b, seed=20261018, resamples=10000):
    if type(resamples) is not int or resamples < 2 or type(seed) is not int:
        raise ContractError("bootstrap requires integer seed and at least two resamples")
    if not a.get("definition") or a.get("definition") != b.get("definition"):
        raise ContractError("metric definitions/depths differ")
    for name in ["documents.jsonl", "queries.jsonl", "qrels.jsonl"]:
        if not a.get("input_sha256", {}).get(name) or a.get("input_sha256", {}).get(name) != b.get("input_sha256", {}).get(name):
            raise ContractError("benchmark input fingerprints differ")
    def index(result):
        out = {}
        for row in result["per_query"]:
            key = tuple(row[k] for k in ["dataset_id", "split", "collection_id", "query_id"])
            if key in out:
                raise ContractError("duplicate per-query identity")
            out[key] = row["metrics"]
            for metric, value in out[key].items():
                number(value, metric)
        return out
    aa, bb = index(a), index(b)
    if not aa or set(aa) != set(bb):
        raise ContractError("paired comparisons require exactly the same query/split/collection identities")
    groups = defaultdict(list)
    for key in sorted(aa):
        if set(aa[key]) != set(bb[key]):
            raise ContractError("metric names differ")
        groups[key[:3]].append((key, {m: bb[key][m] - aa[key][m] for m in aa[key]}))
    output = []
    for group, pairs in sorted(groups.items()):
        rng = random.Random(seed)
        names = sorted(pairs[0][1])
        if any(set(p[1]) != set(names) for p in pairs):
            raise ContractError("inconsistent metrics across queries")
        samples = {m: [] for m in names}
        for _ in range(resamples):
            indices = [rng.randrange(len(pairs)) for _ in pairs]
            for m in names:
                samples[m].append(sum(pairs[i][1][m] for i in indices) / len(indices))
        output.append({"dataset_id": group[0], "split": group[1], "collection_id": group[2], "queries": len(pairs),
            "metrics": {m: {"mean_delta": sum(p[1][m] for p in pairs) / len(pairs),
                            "ci95": [percentile(samples[m], .025), percentile(samples[m], .975)]} for m in names}})
    return {"schema_version": 1, "direction": "B-A", "method": "paired_query_percentile", "seed": seed,
            "resamples": resamples, "confidence": .95, "query_identity_sha256": hashlib.sha256(dumps(sorted(aa)).encode()).hexdigest(),
            "run_a": a["run"], "run_b": b["run"], "groups": output,
            "interpretation": "Conditional query-sampling uncertainty for this pair of runs; not training-seed variation. No pooling across seeds."}
