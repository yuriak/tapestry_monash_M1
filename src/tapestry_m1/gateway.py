"""Local L0 batch validation and identity transport. No privacy protection or network I/O."""
import argparse
import copy
import json
from pathlib import Path

from tapestry_m1.data.schema import document, query, ranking, text
from tapestry_m1.io import ContractError
from tapestry_m1.inference.artifacts import atomic_bytes

SCHEMA = "m1.gateway.batch.v1"
RESERVED_LEVELS = ("L1", "L2", "L3")
SOURCE_CLASSES = ("synthetic", "benchmark_native", "public_corpus", "partner_supplied")


def validate_request(row, collections):
    if not isinstance(row, dict):
        raise ContractError("request must be an object")
    if row.get("schema_version") != SCHEMA:
        raise ContractError("unsupported schema_version")
    text(row, "request_id")
    if row.get("abstraction_level") != "L0":
        raise ContractError("only L0 is implemented; L1/L2/L3 are reserved")
    context = row.get("context")
    if not isinstance(context, dict):
        raise ContractError("context required")
    for key in ("node", "jurisdiction"):
        text(context, key)
    q, r = row.get("query"), row.get("ranking")
    if not isinstance(q, dict) or not isinstance(r, dict):
        raise ContractError("query and ranking objects required")
    query(q)
    ranking(r)
    col = q["collection_id"]
    if col not in collections:
        raise ContractError("collection is not in the explicit allowlist")
    for key in ("dataset_id", "split", "query_id", "collection_id"):
        if q[key] != r[key]:
            raise ContractError("query/ranking identity mismatch: " + key)
    evidence = row.get("evidence")
    if not isinstance(evidence, list):
        raise ContractError("evidence must be an explicit list (empty is allowed)")
    if len(evidence) > 1000 or len(r["candidates"]) > 1000:
        raise ContractError("per-request evidence/candidate limit is 1000")
    docs = {}
    for doc in evidence:
        if not isinstance(doc, dict):
            raise ContractError("evidence must contain document objects")
        document(doc)
        if doc["collection_id"] != col:
            raise ContractError("foreign evidence collection")
        if doc["doc_id"] in docs:
            raise ContractError("duplicate evidence doc_id")
        docs[doc["doc_id"]] = doc
        source = doc.get("source_reference")
        if not isinstance(source, dict):
            raise ContractError("source_reference required")
        for key in ("source_id", "revision", "original_doc_id"):
            text(source, key)
        if doc.get("source_class") not in SOURCE_CLASSES:
            raise ContractError("invalid source_class")
        for key in ("node", "jurisdiction", "source_doc_id"):
            text(doc, key)
        if not doc["source_doc_id"].startswith(col + "::"):
            raise ContractError("source_doc_id must retain the source collection namespace")
    candidates = r["candidates"]
    if set(docs) != {c["doc_id"] for c in candidates}:
        raise ContractError("evidence must cover exactly the candidate set")
    originals = []
    for c in candidates:
        if not c["doc_id"].startswith(col + "::"):
            raise ContractError("foreign candidate collection")
        for key in ("original_rank", "reranked_rank"):
            if type(c.get(key)) is not int or c[key] < 1:
                raise ContractError(key + " must be a positive integer")
        if c["reranked_rank"] != c["rank"]:
            raise ContractError("reranked_rank must equal current rank")
        originals.append(c["original_rank"])
    if set(originals) != set(range(1, len(candidates) + 1)):
        raise ContractError("original_rank must be a permutation of candidate ranks")
    return row


def passthrough(row, collections):
    """Invalid records produce a sanitized error envelope; never echo invalid payloads."""
    request_id = row.get("request_id") if isinstance(row, dict) else None
    try:
        validate_request(row, collections)
    except ContractError as exc:
        return {"schema_version": SCHEMA, "request_id": request_id if isinstance(request_id, str) else None,
                "abstraction_level": "L0", "status": "error",
                "errors": [{"code": "invalid_request", "message": str(exc)}]}
    return copy.deepcopy(row) | {"status": "ok" if row["evidence"] else "no_evidence", "errors": []}


def process_batch(rows, collections, max_records=1000):
    if not collections or any(not isinstance(c, str) or not c.strip() for c in collections):
        raise ContractError("at least one explicit collection is required")
    output, seen = [], set()
    for row in rows:
        if len(output) >= max_records:
            raise ContractError("batch record budget exceeded")
        req = row.get("request_id") if isinstance(row, dict) else None
        if isinstance(req, str):
            if req in seen:
                raise ContractError("duplicate batch request_id")
            seen.add(req)
        output.append(passthrough(row, collections))
    return output


def _rows(path, max_line_bytes):
    with Path(path).open("rb") as handle:
        while True:
            line = handle.readline(max_line_bytes + 1)
            if not line:
                return
            if len(line) > max_line_bytes:
                raise ContractError("JSONL line byte budget exceeded")
            if line.strip():
                try:
                    yield json.loads(line)
                except (ValueError, UnicodeError) as exc:
                    raise ContractError("invalid JSONL record") from exc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--collection", action="append", required=True)
    p.add_argument("--max-records", type=int, default=1000)
    p.add_argument("--max-line-bytes", type=int, default=1048576)
    a = p.parse_args()
    try:
        if not 1 <= a.max_records <= 1000 or not 1 <= a.max_line_bytes <= 1048576:
            raise ContractError("budgets must be positive and cannot exceed the documented hard limits")
        rows = process_batch(_rows(a.input, a.max_line_bytes), set(a.collection), a.max_records)
        payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows)
        atomic_bytes(a.output, payload.encode("utf-8"))
    except (ContractError, OSError, ValueError) as exc:
        p.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
