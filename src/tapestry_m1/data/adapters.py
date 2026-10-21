"""Minimal source-specific normalization; no inference, chunking or invented qrels."""
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
from tapestry_m1.contracts.records import identifier, number, text
from tapestry_m1.data.validation import validate
from tapestry_m1.core.io import ContractError, atomic_json, dumps, fingerprint, records

FILES = ["documents.jsonl", "queries.jsonl", "qrels.jsonl", "training_pairs.jsonl"]


def normalize(spec, raw_root, destination, max_rows=None):
    if not spec["acquisition_allowed"]:
        raise ContractError("blocked_pending_license_review: normalization is disabled")
    if max_rows is not None and (type(max_rows) is not int or max_rows < 1):
        raise ContractError("max_rows must be positive")
    raw_root, destination = Path(raw_root), Path(destination)
    inputs = {name: fingerprint(raw_root / name) for name in spec["files"]}
    extraction = {"version": 2 if spec["adapter"] == "open_au_qa" else 1, "max_rows": max_rows, "native_boundaries": True, "text_unchanged": True}
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        if manifest["input_sha256"] != inputs or manifest["extraction"] != extraction or manifest["source"] != spec:
            raise ContractError("existing normalization has different inputs/config; choose a new output directory")
        for name, sha in manifest["output_sha256"].items():
            if fingerprint(destination / name) != sha:
                raise ContractError("normalized cache checksum mismatch")
        return manifest | {"cache_reused": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + destination.name, dir=destination.parent))
    handles = {name: (stage / name).open("w", encoding="utf-8") for name in FILES}
    counts = Counter()
    truncated = False
    ds, col = spec["dataset_id"], spec["collection_id"]
    def emit(name, row):
        handles[name].write(dumps(row) + "\n")
        counts[name] += 1
    def source_rows(name):
        nonlocal truncated
        for i, row in enumerate(records(raw_root / name), 1):
            if max_rows is not None and i > max_rows:
                truncated = True
                break
            yield i, row
    def provenance(filename, i):
        return {"repo_id": spec["repo_id"], "revision": spec["revision"], "file": filename, "source_row": i}
    try:
        kind = spec["adapter"]
        if kind in ["legal_rag_bench", "tax_guidance", "open_au_corpus"]:
            for i, row in source_rows("corpus.jsonl"):
                raw = row[{"legal_rag_bench": "id", "tax_guidance": "_id", "open_au_corpus": "version_id"}[kind]]
                doc = {"collection_id": col, "doc_id": identifier(col, raw), "original_doc_id": str(raw),
                       "text": row["text"], "provenance": provenance("corpus.jsonl", i), "jurisdiction": row.get("jurisdiction", "Australia")}
                if kind == "open_au_corpus":
                    doc |= {"source_doc_id": str(raw), "metadata": {k:v for k,v in row.items() if k != "text"}}
                else:
                    doc |= {"title": row.get("title", ""), "metadata": {k:v for k,v in row.items() if k not in ["id", "_id", "text", "title"]}}
                emit("documents.jsonl", doc)
        if kind == "legal_rag_bench":
            for i, row in source_rows("qa.jsonl"):
                qid = identifier(ds, row["id"])
                emit("queries.jsonl", {"dataset_id": ds, "split": "test", "query_id": qid, "original_query_id": str(row["id"]),
                    "query_text": row["question"], "collection_id": col, "provenance": provenance("qa.jsonl", i)})
                emit("qrels.jsonl", {"dataset_id": ds, "split": "test", "query_id": qid, "doc_id": identifier(col, row["relevant_passage_id"]),
                    "relevance": 1, "original_doc_id": row["relevant_passage_id"], "original_label": "relevant_passage_id"})
        elif kind == "tax_guidance":
            for i, row in source_rows("queries.jsonl"):
                emit("queries.jsonl", {"dataset_id": ds, "split": "test", "query_id": identifier(ds, row["_id"]),
                    "original_query_id": row["_id"], "query_text": row["text"], "collection_id": col, "provenance": provenance("queries.jsonl", i)})
            for i, row in source_rows("default.jsonl"):
                emit("qrels.jsonl", {"dataset_id": ds, "split": "test", "query_id": identifier(ds, row["query-id"]),
                    "doc_id": identifier(col, row["corpus-id"]), "relevance": row["score"], "original_label": row["score"],
                    "original_doc_id": row["corpus-id"], "provenance": provenance("default.jsonl", i)})
        elif kind == "open_au_qa":
            for i, row in source_rows("qa.jsonl"):
                text(row, "question"); text(row, "answer"); text(row, "text")
                source = row.get("source")
                if not isinstance(source, dict):
                    raise ContractError("QA source metadata is missing")
                text(source, "text")
                emit("training_pairs.jsonl", {"dataset_id": ds, "split": "train", "pair_id": identifier(ds, f"row-{i}"),
                    "original_pair_id": None, "generated_row_identity": True, "query_text": row["question"], "answer": row["answer"],
                    "positive_text": source["text"], "formatted_text": row["text"], "generation_prompt": row.get("prompt"),
                    "source_doc_id": source.get("version_id"),
                    "source_metadata": {k:v for k,v in source.items() if k != "text"}, "provenance": provenance("qa.jsonl", i),
                    "purpose": "future_training_not_retrieval_qrels", "corpus_identity_resolved": False})
        elif kind not in ["open_au_corpus", "legal_rag_bench", "tax_guidance"]:
            raise ContractError("unknown source adapter")
        for handle in handles.values():handle.close()
        validation = validate(stage)
        observed = {"documents": counts["documents.jsonl"], "queries": counts["queries.jsonl"], "qrels": counts["qrels.jsonl"],
                    "training_pairs": counts["training_pairs.jsonl"]}
        discrepancies = {key: {"proposal": expected, "observed": observed[key]} for key, expected in spec["expected_counts"].items()
                         if key in observed and expected != observed[key]}
        manifest = {"schema_version": 1, "source": spec, "input_sha256": inputs,
                    "output_sha256": {name: fingerprint(stage / name) for name in FILES}, "extraction": extraction,
                    "status": "partial" if truncated else "validated", "truncated": truncated, "observed_counts": observed,
                    "expected_count_discrepancies": discrepancies, "validation": validation,
                    "warnings": spec.get("unresolved_flags", []) + (["training_only_no_evaluation_qrels"] if kind == "open_au_qa" else [])}
        atomic_json(stage / "manifest.json", manifest)
        os.rename(stage, destination)
        return manifest
    except Exception:
        for handle in handles.values():
            if not handle.closed:handle.close()
        # Only our own unpublished staging directory; verified source inputs are never removed.
        shutil.rmtree(stage)
        raise
