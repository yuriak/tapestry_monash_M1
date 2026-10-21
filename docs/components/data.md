# Data

Available: pinned AU source acquisition, adapters, integrity validation, overlap/length audits and token-window chunking.

## Ownership

`data/acquisition.py`, `adapters.py`, `validation.py`, `audit.py`, `length_audit.py`, `chunking.py`, `artifacts.py`.

## Input/output boundary

Source metadata/config + raw files -> versioned normalized JSONL/manifests; normalized documents + caller-supplied tokenizer/policy -> source-linked chunks.

## Extension direction

Source roles distinguish index-only, training and test material. Add medical/global/partner source adapters only after source access, exact split and leakage policy review. Do not generate retrieval qrels from QA pairs without a validated corpus join.

See [architecture](../architecture/overview.md), [contracts](../contracts/data.md) and the [component index](README.md).
