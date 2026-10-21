# Artifact identity and storage

Normalized inputs preserve dataset/collection/split/source revision and native IDs. Model specs pin model/tokenizer revision plus available upstream Git/LFS identity. New runs record the selected environment lock hash/profile, resolved component/source configs, implementation identity, hardware/software and input/artifact hashes.

`runtime/provenance.py` hashes the complete `src/tapestry_m1` Python source tree by relative path and bytes, excluding hidden editor/cache directories. This is intentionally conservative: a changed component can require a new run identity. It does not prove scientific correctness. `runtime/configuration.py` resolves model/source references relative to their owning config and records the actual lock selected by `scripts/run`.

The generic immutable-write primitives live in `core`; vector shard validation belongs to `retrieval/vector_cache.py`. Chunk staging preserves source byte checkpoints and character/token lineage. Index coverage distinguishes staged sources, partially represented sources, fully indexed sources and encoded chunks. Source/config/model/checksum mismatches fail closed.

Default storage is ignored `data/`, or explicit `$M1_DATA_ROOT`/`--data-root`. Raw sources, normalized text, model assets, vector caches, rankings, indexes and checkpoints stay outside Git. `outputs/`, `.runtime/`, `.venv/` and `.venvs/` are ignored too. Do not move assets as a side effect of choosing an environment. Keep one writer per artifact identity; finite resource budgets do not imply global corpus completion.

The [initial layout snapshot](../architecture/layout_transition.md) remains the provenance anchor of historical results. New code/config/lock fingerprints do not automatically resume the old GPU/index runs. Preserve original bytes and review any reuse explicitly when needed.
