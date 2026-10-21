# Native retrieval + reranking workflow

This workflow composes `retrieval`, `reranker`, normalized `data` and independent `evaluation`. See the [initial measured results](../experiments/zero_shot_native/results.md) and [layout transition](../architecture/layout_transition.md). Those measurements are historical; the modular implementation has only source/CPU validation until a later model run is authorized.

## Explicit inputs

- `configs/experiments/zero_shot_native/native-qwen3.json`: both native AU benchmarks.
- `configs/experiments/zero_shot_native/legal-qwen3.json`: Legal-only composition.
- Model specs are owned by `configs/retrieval/` and `configs/reranker/`.
- Source specs are explicit references to `configs/data/sources/`; they resolve to complete source metadata in the recorded recipe.

Default shared environment: managed Python 3.11.17 / uv 0.13.1, initial model pins kept. The default wrapper selects that environment and records its actual lock. First acquire model assets only when model work is authorized:

```bash
./scripts/run models configs/retrieval/qwen3-embedding-4b.json --verify-only
./scripts/run models configs/reranker/qwen3-reranker-0.6b.json --verify-only
# Without --verify-only, these commands download the exact selected pinned model files.
```

Assets live at `$M1_DATA_ROOT/models/<model-name>/<revision>/`; `verified.json` records local hashes matched against upstream LFS SHA256 or Git blob identity. Model loading is local-only and remote code is disabled.

## GPU operations — only inside a valid allocation

```bash
./scripts/run benchmark preflight configs/experiments/zero_shot_native/native-qwen3.json
./scripts/run benchmark run-native configs/experiments/zero_shot_native/native-qwen3.json
```

The guard requires a running Slurm GPU allocation and Torch CUDA availability. No site-specific account, partition or launch request is supplied here. Help/import/CPU tests never load weights. Full preflight/inference is a separate resource decision.

The initial recipe uses BF16/SDPA, maximum 8192 tokens, encoder batch 8, reranker batch 16 and 2560-dimensional normalized float32 vectors. Query formatting/pooling belong to `retrieval/qwen3.py`; prompt/yes-no probability scoring to `reranker/qwen3.py`. Documents use normalized native text only. Retrieval is exact normalized-dot Top-100 with score-descending/doc-ID-ascending order. Reranking reads the immutable serialized candidate list, retains original rank/score and enforces exactly the same candidate sets. No gold enters retrieval or reranking.

Preflight checks batch/single numerical agreement, repeat stability, token/context constraints and memory under the exact model/config/source/environment/software identity. New modular fingerprints do not inherit a historical passed preflight. Preserve old manifests; do not edit them to match a changed source tree.

Outputs live in ignored `runs/preflight/<fingerprint>` and `runs/native/<dataset>/<fingerprint>` under the data root. They include rankings, length/truncation audit, per-query metrics, paired comparison, checksums and runtime/resource metadata. Default evaluation definitions are in the [data/metric contract](../contracts/data.md). Alternative score/tie views stay separately labelled; they are not an official leaderboard claim.
