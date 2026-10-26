# Documentation

The [project README](../README.md) introduces the full programme. This index separates architecture, record contracts, component ownership and operational workflows.

- Architecture: [module boundaries](architecture/overview.md), [development](architecture/development.md), [layout transition](architecture/layout_transition.md).
- Contracts: [normalized data and metrics](contracts/data.md), [gateway batch](contracts/gateway.md), [artifact identity and storage](contracts/artifacts.md), [active PEFT training v2](contracts/reranker_training_v2.md), [historical training v1](contracts/reranker_training.md).
- Components: [implementation map](components/README.md), [data](components/data.md), [retrieval](components/retrieval.md), [reranker](components/reranker.md), [gateway](components/gateway.md), [abstractor](components/abstractor.md), [generator](components/generator.md), [federation](components/federation.md), [evaluation](components/evaluation.md), [runtime](components/runtime.md).
- Environments: [setup and profiles](environments/README.md), [independent-stack template](../environments/README.md).
- Workflows: [global acquisition](workflows/global_acquisition.md), [data preparation](workflows/data_preparation.md), [native zero-shot baseline](workflows/native_baseline.md), [corpus staging/indexing](workflows/corpus_index.md), [local gateway](workflows/gateway.md).
- Experiments: [initial zero-shot results](experiments/zero_shot_native/results.md). This is a historical scientific record, not a claim of validation after the layout transition.
- Configuration: [component/source/experiment conventions](../configs/README.md).

Add documentation under the owning category. A new component guide identifies its public inputs, outputs, dependencies, implementation status and validation limits. Cross-component commands belong in workflows. Keep site-private coordination and internal instructions outside this standalone repository.

- [Base-reranker hard-negative training arm](contracts/reranker_hard.md)
