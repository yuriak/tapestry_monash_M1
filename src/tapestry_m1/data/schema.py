"""Version 1 JSONL contract; adapters preserve raw identifiers separately."""
import math
from urllib.parse import quote
from tapestry_m1.io import ContractError


def text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{key}: expected nonempty string")
    return value


def number(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"{name}: expected finite number")
    if nonnegative and value < 0:
        raise ContractError(f"{name}: negative value")
    return value


def identifier(scope, raw):
    if isinstance(raw, bool) or not isinstance(raw, (str, int)) or not str(raw).strip():
        raise ContractError("source identifier is missing or invalid")
    return scope + "::" + quote(str(raw), safe="")


def document(row):
    col = text(row, "collection_id")
    doc = text(row, "doc_id")
    text(row, "text")
    if not doc.startswith(col + "::"):
        raise ContractError("doc_id must be namespaced by collection_id")
    return row


def query(row):
    ds = text(row, "dataset_id")
    text(row, "split")
    q = text(row, "query_id")
    text(row, "query_text")
    text(row, "collection_id")
    if not q.startswith(ds + "::"):
        raise ContractError("query_id must be namespaced by dataset_id")
    return row


def qrel(row):
    for k in ["dataset_id", "split", "query_id", "doc_id"]:
        text(row, k)
    number(row.get("relevance"), "relevance", nonnegative=True)
    return row


def ranking(row):
    for k in ["dataset_id", "split", "query_id", "collection_id", "run_id"]:
        text(row, k)
    if "seed" not in row or (row["seed"] is not None and (isinstance(row["seed"], bool) or not isinstance(row["seed"], int))):
        raise ContractError("seed must be integer or explicit null (deterministic run)")
    fp = row.get("fingerprints")
    if not isinstance(fp, dict):
        raise ContractError("fingerprints required")
    for k in ["model", "index", "config"]:
        text(fp, k)
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        raise ContractError("candidates must be an ordered list")
    seen = set()
    for rank, item in enumerate(candidates, 1):
        if not isinstance(item, dict):
            raise ContractError("candidate must be an object")
        doc = text(item, "doc_id")
        if doc in seen:
            raise ContractError("duplicate retrieved doc_id")
        seen.add(doc)
        if type(item.get("rank")) is not int or item["rank"] != rank:
            raise ContractError("candidate ranks must be consecutive and agree with list order")
        number(item.get("score"), "score")
    return row


def query_key(row):
    return row["dataset_id"], row["split"], row["query_id"]
