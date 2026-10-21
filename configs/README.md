# Configuration ownership

- `data/sources/`: pinned source identity, access/role/split/collection metadata and adapter configuration.
- `retrieval/`: encoder model specification and corpus indexing recipe.
- `reranker/`: reranker model specification; future method/training configs belong here.
- `experiments/zero_shot_native/`: explicit composition of source/model settings for the initial native benchmark workflow.

Future abstractor/generator/federation/evaluation configs are added when an actual implementation needs them, rather than populated with hypothetical defaults.

Reference paths are relative to the file that declares them. `runtime.configuration.load_config` resolves encoder/reranker model paths and dataset source paths, keeping the complete resolved source specifications in runtime provenance. `environment_lock` declares a default lock; the run wrapper supplies the actual selected lock/profile, including independent stacks.

Do not put private roots, scheduler accounts, credentials or assumed partner endpoints in configs. Use `M1_DATA_ROOT`/explicit flags for local assets. Changing a resolved recipe or lock changes identity; old artifacts are never silently re-labelled. Model revision/tokenizer identity stays pinned; this directory's rearrangement is not a model upgrade.
