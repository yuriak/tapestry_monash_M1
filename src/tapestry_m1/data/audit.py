"""Conservative bounded-memory text/source overlap evidence (not a privacy test)."""
from collections import Counter
from pathlib import Path
import hashlib
from urllib.parse import urlsplit, urlunsplit
from tapestry_m1.core.io import records, text_key, fingerprint, ContractError


def canonical_url(url):
    if not isinstance(url, str) or not url:
        return None
    p = urlsplit(url)
    if not p.netloc:
        return None
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/"), "", ""))


def audit_overlap(benchmark, corpus, max_documents=None):
    benchmark, corpus = Path(benchmark), Path(corpus)
    if max_documents is not None and (type(max_documents) is not int or max_documents < 1):
        raise ContractError("audit max_documents must be positive")
    targets = []
    for row in records(benchmark / "documents.jsonl"):
        normalized = text_key(row["text"])
        targets.append({"doc_id": row["doc_id"], "text": normalized,
                        "sha": hashlib.sha256(normalized.encode()).hexdigest(),
                        "url": canonical_url(row.get("metadata", {}).get("url"))})
    findings = []; seen = 0; source_counts = Counter(); truncated = False
    for doc in records(corpus / "documents.jsonl"):
        if max_documents is not None and seen >= max_documents:
            truncated = True;break
        seen += 1
        md = doc.get("metadata", {});source_counts[str(md.get("source", "unknown"))] += 1
        text = text_key(doc["text"])
        sha = hashlib.sha256(text.encode()).hexdigest()
        url = canonical_url(md.get("url"))
        for target in targets:
            kinds = []
            if sha == target["sha"]:kinds.append("normalized_exact_text")
            elif len(target["text"]) >= 80 and target["text"] in text:kinds.append("normalized_benchmark_text_contained_in_document")
            if url and url == target["url"]:kinds.append("same_canonical_url")
            if kinds:findings.append({"benchmark_doc_id":target["doc_id"],"corpus_doc_id":doc["doc_id"],"match_types":kinds})
    import json
    manifest = json.loads((corpus / "manifest.json").read_text()) if (corpus / "manifest.json").exists() else {}
    full = not truncated and manifest.get("status") == "validated" and not manifest.get("truncated", False)
    return {"schema_version":1,"method":"NFKC + casefold + whitespace; exact hash, literal containment >=80 characters, canonical URL",
            "benchmark_document_count":len(targets),"corpus_documents_scanned":seen,"corpus_source_counts":dict(sorted(source_counts.items())),
            "complete_normalized_corpus_scanned":not truncated,"full_source_corpus_coverage":full,"matches":findings,
            "input_sha256":{"benchmark":fingerprint(benchmark/"documents.jsonl"),"corpus":fingerprint(corpus/"documents.jsonl")},
            "limitations":["Exact/containment matches and metadata only, not semantic or fuzzy overlap.",
              "No match is not proof of zero contamination, even for a complete scan.",
              "Shared ATO provenance may matter even when guidance excerpts are not literal matches.",
              "A sampled/partial corpus cannot establish full-source non-overlap."]}
