# Environments and source execution

Use uv **0.13.1** and managed **CPython 3.11.17** for the existing shared/CPU profiles. Install uv from the [official release](https://github.com/astral-sh/uv/releases/tag/0.13.1), verify the upstream checksum and either place the executable at `.runtime/tools/uv/uv` or set `M1_UV_BIN` to its absolute path. All model/data paths below belong to this checkout or an explicitly configured asset root; no coordinating repository is needed.

The repository is a **non-package uv project** (`tool.uv.package=false`). There is no build backend, installed console script or wheel step. The wrapper sets `PYTHONPATH` to this checkout's `src` before invoking Python modules. `pytest` also knows the source path.

| Profile | Environment | Dependencies / lock |
|---|---|---|
| `shared` (default) | Existing `.venv` | Root lock; dev + retrieval + reranker groups |
| `cpu` | Reusable `.venvs/cpu` | Same root lock; dev only, no Torch/Transformers/NumPy |
| Reviewed independent stack | `.venvs/<name>` | Its own `environments/<name>/pyproject.toml` and `uv.lock` |

```bash
./scripts/run --help
./scripts/run python -c 'import sys; print(sys.version, sys.prefix)'
./scripts/run --environment cpu --help
./scripts/run --environment cpu data --help
./scripts/run --environment cpu evaluation --help
./scripts/run --environment cpu gateway --help
./scripts/run pytest -q
```

The first call synchronizes only the selected environment's locked dependencies. It does not install this project. Shared data/evaluation/retrieval/reranking continue to use one `.venv`, rather than creating an environment per experiment. The CPU profile is a long-lived component profile; it cannot run model-dependent commands.

Root dependency groups retain the initial tested model pins: NumPy 2.2.6, PyTorch 2.6.0 (CUDA 12.4 wheels), Transformers 4.51.3, huggingface-hub 0.35.3 and safetensors 0.5.3. Retrieval and reranker include one common `model-runtime` group. CPU data/metrics/gateway code uses the standard library; token-length audit requires the model/NumPy profile and a supplied tokenizer. The initial GPU recipe is Linux x86-64 NVIDIA-specific and does not certify other platforms. It remains a historical recipe until model validation is requested again.

The wrapper clears inherited virtualenv/Conda/Python path settings, uses managed Python and confines caches/interpreters/temp files to ignored `.runtime/`. It exports `M1_ENVIRONMENT` and `M1_ENVIRONMENT_LOCK` for run provenance. `M1_DATA_ROOT` selects asset storage independently; switching environment does not move or duplicate datasets/models. You can also select a profile with `M1_ENVIRONMENT=<name>`.

## Independent stacks

See [environment ownership and template](../../environments/README.md). Create one only for a real reviewed stack/version conflict. Give it an explicit Python version, dependencies, `package=false` and its own committed lock. Existing components that can share dependencies should use the default environment. A future generator or partner framework is not installed just because its directory exists.

The root groups share a resolution/lock. An independent stack is deliberately not a member of a root uv workspace, so it can select a different Python/Torch/Transformers/CUDA combination. The common record/file contracts are the integration boundary; separate processes can exchange manifests/JSONL without importing another environment's framework.

`./scripts/run --environment <name> ...` fails before sync if the independent project's config/lock is missing. Environment names are single safe names, not arbitrary paths. To prepare a new lock use the same reviewed uv executable and project-local managed-Python/cache directories; the [template guide](../../environments/README.md) shows the procedure. Never update the shared lock implicitly as part of locking another stack.
