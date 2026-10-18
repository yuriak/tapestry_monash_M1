# Tapestry Monash M1

Legal retrieval data preparation, model-independent CPU evaluation, and pinned Qwen3 zero-shot retrieval/reranking. It installs independently of its coordinating repository. Training and federation are outside the implementation scope.

## Independent Monash environment

Use **uv 0.13.1** and managed **CPython 3.11.17**, with this checkout's `.venv`. No Conda installation is needed. Data/evaluation and inference share this one environment and `uv.lock`. The lock pins PyTorch 2.6.0 (CUDA 12.4 wheels), Transformers 4.51.3, NumPy 2.2.6 and development dependencies. Data/metric code retains standard-library implementations and model dependencies are imported only by inference paths. Future partner environments can be reviewed when introduced.

Install uv from the [official 0.13.1 release](https://github.com/astral-sh/uv/releases/tag/0.13.1), checking the release checksum for your platform, and set `M1_UV_BIN` to its absolute executable path. Alternatively place its executable at `.runtime/tools/uv/uv`. For Linux x86-64, the official `uv-x86_64-unknown-linux-gnu.tar.gz` SHA-256 is `eadd9d27c134ab5ac1ffd4473b97a741407f3ea238288f03121fc4deddf90220`.

```bash
export M1_UV_BIN=/absolute/path/to/uv
./scripts/run python -c 'import sys; print(sys.version, sys.prefix, sys.base_prefix)'
./scripts/run pytest -q
./scripts/run tapestry-m1 --help
```

The wrapper runs from the project root, clears inherited virtualenv/Conda/Python import settings, requires managed Python, and confines Python installations, package cache and temporary files to ignored `.runtime/`. `uv run --locked` creates/synchronizes `.venv` and fails on lock drift. It does not activate or reuse a cluster Conda environment. The first run needs network access and storage for the pinned CUDA dependencies and managed interpreter. A direct `.venv/bin/python` invocation is suitable after setup. See [uv project synchronization](https://docs.astral.sh/uv/concepts/projects/sync/).

## Data sources

Pinned revisions, candidate collection identity, expected proposal counts, rights flags and upstream links are in [`configs/datasets/`](configs/datasets). Revisions are deliberately fixed; update them explicitly after reviewing source changes.

| Config | Role | Split / source label | Rights and interpretation |
|---|---|---|---|
| `legal-rag-bench` | Victorian Criminal Charge Book benchmark; distinct from LegalBench-RAG | test, one relevant passage per query | Card tag CC BY-NC-SA 4.0 versus body CC BY-NC 4.0: unresolved, noncommercial local use, no bundled source data |
| `australian-tax-guidance-retrieval` | MLEB benchmark using its own 105-document collection | test qrels; source `corpus` and `queries` configs | CC BY 4.0; score 1 is positive; do not silently replace candidates with a larger index |
| `open-australian-legal-corpus` | Unchunked source documents | corpus | Custom corpus licence and source-specific rights; preserve provenance |
| `legal-training-dataset` | JudicialMind AU future reranker staging | Card suggests A/B/C file buckets; Viewer exposes one merged train split | CC BY-NC-ND 4.0; **metadata only, blocked pending licence and split review** |
| `open-australian-legal-qa` | Future generator training | Entire `qa.jsonl` is train in the pinned official loader | Inherits corpus terms; observed 2,124 versus proposal 1,918; generated row IDs are not evaluation qrels |

Each config links its pinned primary source card. Read the linked source licence before reuse, redistribution or cross-institution sharing; this repository distributes no dataset text. No access terms were accepted automatically. Neither benchmark's sparse judgements are asserted to be exhaustive.

## Acquisition and normalization

All downloaded text and artifacts go to an ignored root, default `data/`. To use another private location, set `M1_DATA_ROOT`; ensure that location is excluded from Git. Output paths shown here are examples, with no machine-specific defaults.

```bash
export M1_DATA_ROOT="$PWD/data"
./scripts/run tapestry-m1 inventory configs/datasets/legal-rag-bench.json --output outputs/legal-rag-source.json
./scripts/run tapestry-m1 acquire configs/datasets/legal-rag-bench.json --max-bytes 33554432 --output outputs/legal-rag-download.json
./scripts/run tapestry-m1 normalize configs/datasets/legal-rag-bench.json
# Repeat with australian-tax-guidance-retrieval.json and, optionally, open-australian-legal-qa.json.
```

Raw files: `$M1_DATA_ROOT/raw/<dataset_id>/<revision>/`. Normalized files: `$M1_DATA_ROOT/normalized/<dataset_id>/<revision>/`. Each normalized directory contains `documents.jsonl`, `queries.jsonl`, `qrels.jsonl`, `training_pairs.jsonl` and `manifest.json` (irrelevant streams are empty). Explicit `--data-root` overrides the variable. `normalize --output` changes the normalized directory, while other commands' `--output` writes the result JSON. Error JSON goes to stderr with exit code 2; preserve it alongside acquisition logs when a source fails.

Downloads pin source revisions, enforce a **per-file** byte budget, retain `.part` files and identity sidecars, and require correctly honored HTTP Range on resume. They validate upstream LFS SHA-256 where supplied and always record local SHA-256 and UTC acquisition time. Verified caches are reused; mismatches fail instead of overwriting. Interrupted finalization with an official hash can recover a verified file. Other uncertain files are preserved for inspection. JSON outputs also refuse to overwrite different artifacts; use a new output filename for a new run/inventory timestamp. Do not run concurrent writers against the same destination.

For the larger corpus, first inspect source layout, free disk and your CPU allocation. The pinned raw file is 9,401,179,433 bytes. Reserve space for raw plus normalized data and disk-backed ID validation (at least 40 GiB free is recommended). Downloading is optional when resource limits intervene:

```bash
./scripts/run tapestry-m1 acquire configs/datasets/open-australian-legal-corpus.json --max-bytes 10737418240
./scripts/run tapestry-m1 normalize configs/datasets/open-australian-legal-corpus.json --max-rows 1000 --output data/corpus-preview
# For a full pass, omit --max-rows and use a different output directory.
./scripts/run tapestry-m1 audit data/normalized/au-australian-tax-guidance-retrieval/982aa11494f04dee16cacf18ae1fbf59f6669845 data/corpus-preview --output outputs/overlap-preview.json
```

`--max-rows` bounds normalization of an already downloaded source, not the transfer. Use it for corpus previews; truncated benchmark inputs can fail qrels integrity checks. Parsing and corpus ID joins are streamed/disk-backed. Native text and boundaries remain unchanged: chunking, source-to-chunk lineage and index building are later responsibilities. Overlap evidence uses normalized hashes, literal containment and source URLs; sampled coverage or no literal match cannot establish zero contamination.

## Evaluation and hand-off

See [the record contract](docs/contracts.md). A later retriever consumes documents and queries, preserves their IDs/collection and emits one complete ranking record per query (including explicit empty lists for zero hits):

```bash
./scripts/run tapestry-m1 validate data/normalized/<dataset_id>/<revision>
./scripts/run tapestry-m1 evaluate data/normalized/<dataset_id>/<revision> outputs/rankings-A.jsonl --output outputs/evaluation-A.json
./scripts/run tapestry-m1 compare outputs/evaluation-A.json outputs/evaluation-B.json --seed 20261018 --resamples 10000 --output outputs/comparison.json
```

The checked-in [`tests/fixtures/`](tests/fixtures) contain **synthetic** documents and tied rankings only. A CPU end-to-end example is:

```bash
./scripts/run tapestry-m1 evaluate tests/fixtures tests/fixtures/rankings.jsonl --output outputs/synthetic-evaluation.json
./scripts/run tapestry-m1 compare outputs/synthetic-evaluation.json outputs/synthetic-evaluation.json --resamples 1000 --output outputs/synthetic-self-comparison.json
```

Metrics use exponential gain `2^relevance-1` (or explicit `--gain linear`), log2(rank+1) discount and judged-positive recall denominators within the eligible collection. MRR defaults to the entire submitted list; `--mrr-depth N` labels MRR@N. Candidate order controls ties, never an implicit sort. Unjudged documents receive zero gain for calculation, without claiming true irrelevance. Missing gold, missing predictions, mixed runs/seeds, foreign collections and malformed labels are errors.

Results retain per-query metrics, macro summaries per dataset/split/collection, run/seed/model/index/config identity and artifact hashes. Comparisons require identical benchmark inputs, metric definitions and exact query identities; query pairs are resampled together, producing B-A deltas and percentile 95% intervals. These intervals describe conditional query-sampling uncertainty, not variation across training seeds. Keep each experiment seed in its own ranking and evaluation artifact; aggregation across three runs remains a later protocol decision.

The benchmark repositories supply data rather than a standalone lightweight scoring routine. Synthetic checks validate declared metric definitions. A separate reference-backend check used pytrec-eval-terrier: binary metrics and explicitly linear graded gain agree on 200 synthetic rankings. Default exponential graded gain and supplied-list ties must not be presented as protocol-identical to different leaderboard conventions. See [zero-shot inference](docs/inference.md) for model execution and limits.

## Pinned zero-shot inference

Use `tapestry-m1-model preflight configs/runs/native-qwen3.json` and then `tapestry-m1-model run-native configs/runs/native-qwen3.json` inside an allocated GPU job. Both run through `./scripts/run` in the shared `.venv`. Read [inference and corpus-index instructions](docs/inference.md) before model acquisition or execution. GPU availability alone is insufficient: the CLI verifies a running Slurm GPU allocation. No site-specific scheduler account or partition is embedded here.
