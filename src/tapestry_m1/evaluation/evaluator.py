"""Strict complete-query evaluator, independent of retrieval/model code."""
from collections import defaultdict
from pathlib import Path
import sqlite3
import tempfile
from tapestry_m1.data import schema
from tapestry_m1.data.validation import build_index
from tapestry_m1.evaluation.metrics import metrics
from tapestry_m1.io import ContractError, records, fingerprint


def evaluate(root, rankings_path, gain="exponential", mrr_depth=None, split=None):
    root = Path(root)
    with tempfile.TemporaryDirectory(prefix="m1-evaluate-") as tmp:
        with sqlite3.connect(str(Path(tmp) / "ids.sqlite")) as db:
            validation = build_index(root, db)
            queries = {schema.query_key(r): r for r in records(root / "queries.jsonl") if split is None or r["split"] == split}
            if not queries:
                raise ContractError("no queries selected")
            rels = defaultdict(dict)
            for row in records(root / "qrels.jsonl"):
                if schema.query_key(row) in queries:
                    rels[schema.query_key(row)][row["doc_id"]] = row["relevance"]
            seen = set()
            per_query = []
            run = None
            for row in records(rankings_path):
                schema.ranking(row)
                key = schema.query_key(row)
                if key not in queries:
                    raise ContractError("ranking query/split not in selected query set")
                if key in seen:
                    raise ContractError("duplicate ranking query")
                seen.add(key)
                q = queries[key]
                if row["collection_id"] != q["collection_id"]:
                    raise ContractError("ranking collection differs from query collection")
                identity = {k: row[k] for k in ["run_id", "seed", "fingerprints"]}
                if run is None:
                    run = identity
                elif run != identity:
                    raise ContractError("mixed run/seed/fingerprints in one ranking artifact")
                doc_ids = []
                for candidate in row["candidates"]:
                    doc = db.execute("SELECT collection FROM docs WHERE id=?", (candidate["doc_id"],)).fetchone()
                    if not doc or doc[0] != row["collection_id"]:
                        raise ContractError("ranked document absent or in wrong candidate collection")
                    doc_ids.append(candidate["doc_id"])
                per_query.append({k: q[k] for k in ["dataset_id", "split", "query_id", "collection_id"]} |
                    {"metrics": metrics(doc_ids, rels[key], gain, mrr_depth), "retrieved_depth": len(doc_ids),
                     "judged_positive_count": sum(v > 0 for v in rels[key].values()),
                     "unjudged_retrieved_count": sum(d not in rels[key] for d in doc_ids)})
            if seen != set(queries):
                raise ContractError(f"missing predictions for {len(set(queries)-seen)} queries; submit explicit empty candidates for zero hits")
    per_query.sort(key=lambda r: (r["dataset_id"], r["split"], r["collection_id"], r["query_id"]))
    grouped = defaultdict(list)
    for row in per_query:
        grouped[(row["dataset_id"], row["split"], row["collection_id"])].append(row)
    groups = []
    for (ds, sp, col), rows in sorted(grouped.items()):
        names = rows[0]["metrics"]
        groups.append({"dataset_id": ds, "split": sp, "collection_id": col, "evaluated_queries": len(rows),
                       "macro": {m: sum(r["metrics"][m] for r in rows) / len(rows) for m in names}})
    return {"schema_version": 1, "run": run, "definition": {"gain": gain, "discount": "log2(rank+1)",
            "mrr_depth": mrr_depth if mrr_depth else "submitted", "positive_threshold": ">0", "unjudged_policy": "zero_gain_not_claimed_truly_irrelevant"},
            "input_sha256": {p: fingerprint(root / p) for p in ["documents.jsonl", "queries.jsonl", "qrels.jsonl"]} |
                {"rankings": fingerprint(rankings_path)}, "integrity": validation, "groups": groups,
            "counts": {"evaluated": len(per_query), "skipped": 0, "errors": 0}, "per_query": per_query}
