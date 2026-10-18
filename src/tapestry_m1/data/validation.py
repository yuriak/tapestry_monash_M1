"""Disk-backed identity joins; never retain corpus text in memory."""
from collections import Counter
import hashlib
from pathlib import Path
import sqlite3
import tempfile
from tapestry_m1.data import schema
from tapestry_m1.io import ContractError, records, text_key


def build_index(root, db):
    root = Path(root)
    db.executescript("""
    CREATE TABLE docs(id TEXT PRIMARY KEY, collection TEXT NOT NULL);
    CREATE TABLE queries(dataset TEXT, split TEXT, id TEXT, collection TEXT, qhash TEXT,
      PRIMARY KEY(dataset, split, id));
    CREATE INDEX query_hash ON queries(dataset, qhash);
    CREATE TABLE pairs(dataset TEXT, split TEXT, id TEXT, qhash TEXT, PRIMARY KEY(dataset, split, id));
    CREATE TABLE rels(dataset TEXT, split TEXT, query TEXT, doc TEXT, relevance REAL,
      PRIMARY KEY(dataset, split, query, doc));
    """)
    counts = Counter()
    try:
        for row in records(root / "documents.jsonl"):
            schema.document(row)
            db.execute("INSERT INTO docs VALUES (?,?)", (row["doc_id"], row["collection_id"]))
            counts["documents"] += 1
        for row in records(root / "queries.jsonl"):
            schema.query(row)
            h = hashlib.sha256(text_key(row["query_text"]).encode()).hexdigest()
            old = db.execute("SELECT split FROM queries WHERE dataset=? AND qhash=? AND split!=? LIMIT 1", (row["dataset_id"], h, row["split"])).fetchone()
            if old or db.execute("SELECT id FROM queries WHERE dataset=? AND id=? AND split!=? LIMIT 1", (row["dataset_id"], row["query_id"], row["split"])).fetchone():
                raise ContractError("split contamination: normalized query text occurs across splits")
            db.execute("INSERT INTO queries VALUES (?,?,?,?,?)", (*schema.query_key(row), row["collection_id"], h))
            counts["queries"] += 1
        for row in records(root / "qrels.jsonl"):
            schema.qrel(row)
            q = db.execute("SELECT collection FROM queries WHERE dataset=? AND split=? AND id=?", schema.query_key(row)).fetchone()
            doc = db.execute("SELECT collection FROM docs WHERE id=?", (row["doc_id"],)).fetchone()
            if not q:
                raise ContractError("unresolved qrels query (or wrong split)")
            if not doc:
                raise ContractError("gold document absent from eligible candidate collection; denominator cannot be reduced")
            if q[0] != doc[0]:
                raise ContractError("qrels joins the wrong collection")
            db.execute("INSERT INTO rels VALUES (?,?,?,?,?)", (*schema.query_key(row), row["doc_id"], row["relevance"]))
            counts["qrels"] += 1
        pairs_path = root / "training_pairs.jsonl"
        if pairs_path.exists():
            for row in records(pairs_path):
                for key in ["dataset_id", "split", "pair_id", "query_text", "answer", "positive_text"]:
                    schema.text(row, key)
                if not row["pair_id"].startswith(row["dataset_id"] + "::"):
                    raise ContractError("training pair_id namespace differs from dataset")
                h = hashlib.sha256(text_key(row["query_text"]).encode()).hexdigest()
                if db.execute("SELECT id FROM pairs WHERE dataset=? AND split!=? AND (qhash=? OR id=?) LIMIT 1",
                              (row["dataset_id"], row["split"], h, row["pair_id"])).fetchone():
                    raise ContractError("training split contamination")
                db.execute("INSERT INTO pairs VALUES (?,?,?,?)", (row["dataset_id"], row["split"], row["pair_id"], h))
                counts["training_pairs"] += 1
        missing = db.execute("""SELECT q.id FROM queries q WHERE NOT EXISTS
        (SELECT 1 FROM rels r WHERE r.dataset=q.dataset AND r.split=q.split AND r.query=q.id AND r.relevance>0) LIMIT 1""").fetchone()
        if missing:
            raise ContractError("query has no positive qrels; explicitly resolve before evaluation")
        db.commit()
    except sqlite3.IntegrityError as exc:
        raise ContractError(f"duplicate document/query/qrel/training-pair identity: {exc}") from exc
    return {"counts": {k: counts[k] for k in ["documents", "queries", "qrels"]},
            "collections": [x[0] for x in db.execute("SELECT DISTINCT collection FROM docs ORDER BY collection")],
            "qrels_resolved": counts["qrels"], "training_pairs_validated": counts["training_pairs"], "judgments_complete": "not_asserted"}


def validate(root):
    with tempfile.TemporaryDirectory(prefix="m1-validation-") as tmp:
        with sqlite3.connect(str(Path(tmp) / "ids.sqlite")) as db:
            return build_index(root, db)
