# Corpus staging and sharded indexing

Token-window preparation is owned by `data/chunking.py`. Encoding/index/search are owned by `retrieval`. `workflows/corpus.py` composes them without depending on a reranker or benchmark dataset config.

`configs/retrieval/au-corpus.json` is the encoder-only composition. Source text/lineage, tokenizer revision, token window and overlap remain explicit. The initial AU corpus has 232,560 source documents; historical coverage was 1,024 staged documents / 5,755 chunks and 1,001 encoded chunks / 202 fully indexed documents. It remains partial; this layout change does not build or certify more coverage.

```bash
# CPU tokenizer/staging work; still needs the shared model/tokenizer dependencies and local tokenizer files.
./scripts/run corpus stage configs/retrieval/au-corpus.json <normalized-corpus-root> --max-documents 1024 --window 2048 --overlap 256
# GPU allocation required for encoding:
./scripts/run corpus encode configs/retrieval/au-corpus.json <normalized-corpus-root> --max-documents 1024 --max-chunks 1024 --max-seconds 600 --window 2048 --overlap 256
```

The initial window/overlap is a recorded policy, not an optimized quality claim. Stage shards retain native parent IDs and source character/token spans with source-byte checkpoints. Index manifests describe checked vector shards and explicit complete/partial coverage. Source/data/config/model/implementation changes generate new identities. The current modular source/config does not transparently resume the historical index identity; review necessary reproduction/reuse later against its original source snapshot.

Budget caps cover staged documents, total encoded coverage and encoding-loop seconds, not total job wall time including startup/staging/checking. A source-document shard may contain a very long document, so document count is not a strict byte/memory cap. One writer owns an identity; future parallel scheduling/ANN needs separate design and validation. Existing search is memory-bounded exact search, not an ANN index.

The larger AU index has no universal validated qrels. Report coverage/resources, not imaginary corpus-wide accuracy. Native Tax/Legal benchmark collections retain their own eligibility and must not be replaced by this larger index. Check capacity/quota and compute allocation before expansion; selecting this command is not permission for a full build.
