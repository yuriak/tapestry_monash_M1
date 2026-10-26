# Architecture and ownership

The source is one modular Python tree executed from the checkout. Modules are organized by stable responsibility rather than milestone/week. They are not separately published packages. The public data/config boundary permits the same source tree to run in a shared environment or in a reviewed independent stack.

| Module | Owns | Must not own |
|---|---|---|
| `core` | Text/JSONL I/O, exceptions, checksums, immutable artifact primitives | Model frameworks or domain pipelines |
| `contracts` | Normalized records and gateway vocabulary | Data fetching, models or transport implementation |
| `data` | Source adapters/acquisition, normalization, source verification, overlap/length audits and token-window chunk preparation | Model ranking, generation or experiment orchestration |
| `runtime` | Pinned model acquisition/verification, config resolution, lazy local model loading, implementation fingerprint and device guard | Corpus policy, ranking decisions or complete experiments |
| `retrieval` | Encoder backend, vector caches/encoding, index build and exact search | Reranking or generation |
| `reranker` | Reranker backend, candidate-preserving ordering, local group losses/LoRA trainer and rank lineage | Retrieval/index construction or complete experiments |
| `gateway` | Bounded batch validation/transport | Rewriting or privacy guarantees |
| `abstractor` | Future query/evidence translation and controlled rewriting | Batch transport or generator training |
| `generator` | Future generation, serving/training adapters, experts/router | Dataset acquisition or federation server policy |
| `federation` | Future framework adapters and shared/local exchange boundary | A bundled hypothetical partner implementation |
| `evaluation` | Model-independent retrieval scoring/statistics; future task-specific diagnostics | Importing retriever, reranker or generator implementations |
| `workflows` | Compose components, choose inputs/outputs and bounded execution stages | Duplicated backend/model implementations |

Dependency direction: `core` and `contracts` are foundations. Components consume them and explicitly reviewed runtime/data services. Workflows import components. Components never import workflows. Data and CPU gateway/evaluation commands do not import model frameworks. The token-length audit is an optional NumPy-dependent data operation; the tokenizer is supplied by its caller. `tests/integration/test_module_boundaries.py` enforces the current component direction.

The Qwen3 encoder and reranker have separate classes in their respective components. Their common runtime loader only handles pinned local Transformer loading, dtype/backend/context setup and model lifetime. Prompt formatting, pooling and yes/no scoring stay with the owning backend.

Configuration does not define Python import boundaries. Experiment configs explicitly reference component and source configs; runtime resolution records actual selected environment lock identity. Data references and qrels retain collection/split provenance across workflows.

Extension areas are documented for abstractor, generator, federation and later evaluation. Their directory presence is not an implementation or acceptance claim. New work adds only the interfaces and dependencies justified by its own experiment.

Local training shares the existing reranker serialization and runtime assets; framework-independent group validation/reference math live under `contracts`, restartable group order and adapter/loss logic under `reranker`, checkpoint/RNG persistence under `runtime`, and the source CLI under `workflows`. Site allocation scripts and raw runtime artifacts are operator-owned, outside published source.
