# Qwen3 zero-shot inference and corpus staging

## One environment and fixed inputs

Data preparation, metrics and inference use the checkout's existing uv-managed `.venv`, `.python-version`, `pyproject.toml` and `uv.lock`. No separate environment is created. Use `./scripts/run ...`; it isolates Python import settings from inherited Conda environments and confines caches to ignored `.runtime/`. GPU model work must run in an actual running Slurm GPU allocation. CPU tests never load model weights.

The initial tested recipe uses PyTorch 2.6.0+cu124, Transformers 4.51.3, BF16, SDPA, a maximum of 8192 tokens, and full 2560-dimensional embeddings. These are pinned scientific settings, not universal hardware recommendations. The shared lock targets Linux x86-64 NVIDIA CUDA 12.4, with a compatible driver and a device supporting BF16. Other platforms have not been validated. Single/batch preflight tolerances are explicitly frozen in `configs/runs/native-qwen3.json` (cosine distance <=0.001, reranker probability absolute error <=0.02). Deterministic repeat checks use 1e-6. All measured errors must be retained; tolerances do not establish bitwise equivalence across different batching/hardware.

Model source configs pin the upstream commit and tokenizer commit, file sizes and available LFS SHA-256 / Git blob IDs. The official recipes are [Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) and [Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B). No floating main or remote model code is used. Models are loaded from verified local files only.

Models live at `$M1_DATA_ROOT/models/<model-name>/<revision>/`. Download the pinned revision with `hf download`, selecting root configuration/tokenizer files, weight index and safetensors; exclude alternate exports such as ONNX to avoid unnecessary copies. Verify LFS file SHA-256, and Git blob IDs for ordinary files. A `verified.json` contains model_id, revision and `files[filename]` with local sha256/bytes; the inference code rehashes every selected file before use. The checked-in model configs specify the exact required set. Do not copy weights into Git.

A small independent acquisition command is:

```bash
./scripts/run hf download Qwen/Qwen3-Embedding-4B --revision 5cf2132abc99cad020ac570b19d031efec650f2b --local-dir data/models/Qwen3-Embedding-4B/5cf2132abc99cad020ac570b19d031efec650f2b --include '*.safetensors' '*.json' '*.txt' --exclude 'onnx/*' '1_Pooling/*' '2_Normalize/*'
```

Use the portable command below to acquire exactly the checked-in file set and produce the verified manifest. `--verify-only` rechecks an existing local download without network acquisition.

```bash
./scripts/run python -m tapestry_m1.model_assets configs/models/qwen3-embedding-4b.json
./scripts/run python -m tapestry_m1.model_assets configs/models/qwen3-reranker-0.6b.json
``` Changing selected model/config files requires a new verified manifest and recipe, never silent cache reuse.

## Native benchmark execution

```bash
./scripts/run tapestry-m1-model preflight configs/runs/native-qwen3.json
./scripts/run tapestry-m1-model run-native configs/runs/native-qwen3.json
```

The English instruction is literal in the run config. Queries use the official instruction wrapper on `query_text`; documents encode only the normalized `text` field, with no extra title/footnote concatenation and no query instruction. Embeddings use the final non-padding token and float32 L2 normalization. Exact float32 normalized-dot retrieval returns min(100,N) unique native candidates, score descending then doc ID ascending. The two benchmark collections are never merged or substituted with the large AU index. Retrieval does not use qrels; the evaluator consumes them afterward.

Reranking reads the immutable X1.1 ranking, scores exactly those query/document pairs using the official chat prefix/suffix and the normalized probability of yes against no logits, then enforces per-query candidate set equality. Only the final position's vocabulary logits are materialized (`logits_to_keep=1`) to avoid unnecessary full-sequence logits. Retrieval rank/score remain auxiliary provenance. This is Top-100 reranking, not a second retrieval pass.

Preflight samples source-text length extremes without relevance judgements and checks single/batch vectors and scores, repeated-call stability, tokenizer yes/no IDs, dimension, truncation and peak PyTorch allocation. It must pass for the exact config/lock, model files, implementation and GPU/software fingerprint before formal native execution. The job ID is recorded as runtime evidence but excluded from scientific cache identity so compatible interrupted jobs can resume.

Artifacts:

- `data/runs/preflight/<recipe-fingerprint>/passed.json` records actual checks, hardware and errors.
- `data/embeddings/<collection>/<cache-fingerprint>/` holds content-addressed vector shards and explicit completion manifests.
- `data/runs/native/<dataset>/<fingerprint>/` contains `x1_1.jsonl`, `x1_2.jsonl`, length audits, per-query pair caches, evaluator outputs, paired comparison and `result.json`.

Caches fingerprint input IDs/manifest, model/tokenizer, instruction, pooling, normalization, token limits, batch/shard shape, dtype, backend, hardware/software and implementation. Unfinished shards are not accepted as complete. Published output files refuse different overwrites; completed runs revalidate recorded file hashes. Repeated/resumed timings include the actual cache state, not a fresh-run claim. Retain aborted logs as failure/recovery evidence.

Native evaluation preserves every query and collection; missing results and foreign IDs are errors. MRR is over submitted Top-100, never unlabelled MRR@10. Paired differences are X1.2 minus X1.1, with 10,000 query-paired bootstrap resamples and fixed configurable seed. Deterministic repeats are stability checks, not independent training seeds; query bootstrap does not estimate training-seed variance. Store alternative benchmark-compatible metric definitions separately and verify their protocol before comparing published reference figures.

## Larger AU corpus: resource audit before scale-up

The source remains 232,560 unchunked documents with native provenance. Use `tapestry_m1.corpus.audit.length_audit` for fixed-seed uniform reservoir sampling plus character-longest tail inspection; it also streams the entire normalized source to verify its hash and source counts. Token estimates are sampled, not an exhaustive token census. GPU-hours need measured throughput; no relevance-based chunk tuning is performed.

The initial resource audit selected 2048-token windows with 256-token overlap: estimated 1.23 million chunks and 12.6 GB of FP32 vectors, before index metadata/text/cache overhead. These are estimates for the pinned source and tokenizer. A larger context per chunk reduces vector count relative to 1024/128 while preserving more contiguous legal context; quality is unmeasured.

```bash
./scripts/run python -m tapestry_m1.corpus.cli stage configs/runs/native-qwen3.json data/normalized/au-open-australian-legal-corpus/ef45e3fec41a960919a31149eee6dab9aa39f725 --max-documents 1024 --window 2048 --overlap 256
# Only inside an allocated GPU job:
./scripts/run python -m tapestry_m1.corpus.cli encode configs/runs/native-qwen3.json data/normalized/au-open-australian-legal-corpus/ef45e3fec41a960919a31149eee6dab9aa39f725 --max-documents 1024 --max-chunks 1024 --max-seconds 600 --window 2048 --overlap 256
```

Stage output is under `data/chunks/<fingerprint>/`. Immutable shards record source offsets, per-document chunk counts, source IDs, token/character spans, unchanged source slices, tokenizer revision and policy. Checkpointing supports resuming at the next source byte offset; increasing the document budget extends the same staging pipeline. Character spans use Python Unicode character indices, not UTF-8 bytes. Snapshot filenames record coverage; partial never means full-source completion.

Index output is under `data/indices/au-corpus/<fingerprint>/`. Encoding is shard-bounded, with explicit chunk/time budgets and per-shard verification. Coverage distinguishes staged documents, documents with any encoded chunk, fully indexed source documents and total source documents. The index is normalized vector shards with memory-bounded exact search (`tapestry_m1.corpus.index.search_index`), not an ANN approximation. No corpus-wide vector concatenation is required. Resume by extending the total chunk/document budget using the same recipe; unchanged completed shards are reused.

The full AU index has no validated global qrels here. Report only coverage, resources and remaining work; a prefix sample's throughput may not represent the whole corpus. Native benchmark baselines take priority. A partial corpus build is a valid staged output, not a claimed full index or legal retrieval quality result.

## Storage and cleanup

`M1_DATA_ROOT` or `--data-root` selects ignored/external storage. Check free space before downloads and scale-up. Preserve verified source data and completed experiment artifacts. Remove only your own allowlisted orphan/cache products after checking active job references; never delete a cache used by a running job or another project. Private scheduler settings and job logs belong outside this public source tree or in ignored runtime storage.
