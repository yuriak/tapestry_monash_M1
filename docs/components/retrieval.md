# Retrieval

Available: Qwen3 embedding, checked vector caching/encoding, sharded index build and exact search.

## Ownership

`retrieval/qwen3.py`, `encoding.py`, `vector_cache.py`, `index.py`, `search.py`.

## Input/output boundary

Verified source/chunk records and pinned encoder config -> vectors, index/coverage manifests and ordered native/chunk candidates.

## Extension direction

Later retriever adaptation/global index work belongs here. Search does not consume qrels. Token chunk preparation belongs to data. Model GPU execution remains allocation-guarded; the full AU index is still partial.

See [architecture](../architecture/overview.md), [contracts](../contracts/data.md) and the [component index](README.md).
