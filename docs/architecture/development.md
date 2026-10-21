# Development from source

Read the [architecture](overview.md), owning [component guide](../components/README.md), [contracts](../contracts/data.md) and [environment guide](../environments/README.md) before extending a component.

1. Place logic under the stable component; put cross-component orchestration under `workflows`.
2. Define inputs/outputs and source/config/artifact identity explicitly. Keep source-native IDs and split lineage.
3. Add component config and synthetic tests next to their peers. Use optional/lazy imports for heavy runtime dependencies.
4. Keep dependencies compatible with the shared profile, or define a separately locked stack after reviewing an actual incompatibility. Do not add a model framework to the lightweight base solely because a future component might need it.
5. Update the component guide and docs index. Document intended behavior separately from executed validation and results.

```bash
./scripts/run pytest -q
./scripts/run --environment cpu pytest -q
./scripts/run --environment cpu evaluation evaluate tests/fixtures/benchmark tests/fixtures/benchmark/rankings.jsonl --output outputs/synthetic-evaluation.json
./scripts/run --environment cpu gateway tests/fixtures/gateway/request.jsonl outputs/synthetic-gateway.jsonl --collection demo
```

Tests are synthetic CPU checks. The lightweight profile skips model/NumPy-dependent tests when those dependencies are absent. GPU preflight, dataset acquisition and full indexing are explicit workflows, not test-suite side effects. A passing CPU suite does not establish model accuracy, GPU numerics or historical reproduction.

The repository has no build backend, package entry-point installation or wheel release workflow. `scripts/run` sets this checkout's `src` path and invokes Python modules. Use `python -m tapestry_m1 ...` with an explicitly configured source path when invoking the interpreter directly. Run commands from the checkout; never require a private coordinating repository import.
