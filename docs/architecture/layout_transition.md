# Component layout transition

The initial source snapshot is `b5f12c04063c25a94f4a26e0895cfc98925edff5`. The modular layout retires the flat `io`, `cli`, `model_assets`, mixed `inference` tree and `corpus` tree. There are no parallel legacy implementations or permanent compatibility shims.

| Initial location | Current owner |
|---|---|
| `io.py`; general `inference/artifacts.py` helpers | `core/io.py`, `core/artifacts.py` |
| `data/schema.py`; gateway vocabulary | `contracts/records.py`, `contracts/gateway.py` |
| `model_assets.py`; model/device/config checks | `runtime/` |
| `corpus/chunks.py`, `corpus/audit.py` | `data/chunking.py`, `data/length_audit.py` |
| Vector cache, embedding, exact search, `corpus/index.py` | `retrieval/` |
| Reranker prompt/scoring and candidate ordering | `reranker/` |
| Combined native pipeline and corpus CLI | `workflows/` |
| Flat gateway | `gateway/batch.py`, `gateway/cli.py` |
| Retrieval metrics / bootstrap | `evaluation/retrieval/`, `evaluation/statistics/` |

Commands now use `./scripts/run data|evaluation|models|benchmark|corpus|gateway ...`. Old installed `tapestry-m1`/`tapestry-m1-model` console scripts and old import/config paths are not supported by the modular tree. Current operations are in [workflows](../README.md).

Source/config/lock identities change with the layout. The new implementation fingerprint covers the complete component source tree. Old results, preflight passes, chunk shards and index manifests retain their original hashes and scientific meaning; they are not relabelled or revalidated by this renovation. Runtime assets remain where configured. Necessary historical reproduction/resumption can be reviewed against the original source snapshot when a later task requires it. Do not manually rewrite old cache identities to make them match the new code.

The layout transition validates source imports, synthetic CPU behavior, config references and environment selection. It does not certify a GPU/model rerun or uninterrupted compatibility with old caches.
