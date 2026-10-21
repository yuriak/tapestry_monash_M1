# Independent runtime projects

The existing shared and CPU profiles are defined in the root project; do not create one environment per numbered experiment. Independent projects are for genuine framework/Python/CUDA incompatibilities, such as a future generator-serving or partner federation stack.

`templates/standalone/pyproject.toml` is an unconfigured source-project example. It is not a generator/Slakshna implementation and has no speculative dependencies. Copy it to an owning stack directory, review Python/framework versions and dependencies, add that directory's `.python-version`, then generate and commit its own `uv.lock`.

Example preparation from the repository root, after selecting actual requirements:

```bash
# Copy the template into environments/<reviewed-stack>/ and edit its name/Python/dependencies first.
# Set M1_UV_BIN to the reviewed uv executable.
UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python" UV_CACHE_DIR="$PWD/.runtime/cache/uv" "$M1_UV_BIN" lock --project environments/<reviewed-stack> --managed-python
./scripts/run --environment <reviewed-stack> --help
```

The wrapper runs the common source through `PYTHONPATH`, not an installed project package. The stack environment lives at ignored `.venvs/<reviewed-stack>`; assets stay under the explicit shared/independent data root. Include only dependencies actually needed by that stack. Internal libraries from other repositories require explicitly reviewed pinned dependencies; no private-parent imports or sys.path hacks.

A stack's `pyproject.toml`, `.python-version`, `uv.lock` and short usage/validation README are tracked. Its environment, caches, weights and private config are ignored. Its lock and runtime/software identity must be included in resulting artifact provenance. Avoid declaring it as a root workspace member: root workspace dependency unification defeats isolation for incompatible stacks.
