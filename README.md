# Sovereign Evidence Assistant — Monash M1

A modular research codebase for evidence-grounded assistance in medicine and law across sovereign institutional and jurisdictional nodes. The project studies how shared learning and node-local specialization can improve evidence ranking and cited answers while keeping locally governed data under node control.

## Research programme

**RQ1 — relevance under distributional heterogeneity.** Compare local and shared rerankers, shared-backbone/local-head approaches, jurisdiction routing and calibration. Measure query–evidence heterogeneity and study relevance distillation over abstracted evidence.

**RQ2 — shared and node experts over abstract representations.** Study query/evidence abstraction, shared and local generator experts, routing and staged/joint adaptation. Evaluate grounding and utility alongside named privacy attacks. The presence of a gateway interface alone does not establish privacy protection.

The programme progresses through three milestones:

| Milestone | Scope |
|---|---|
| M1.1 | Retrieval/reranking baselines, heterogeneity audit and personalized federated rerankers |
| M1.2 | Abstraction and generator adaptation, shared/local MoE experts and routing |
| M1.3 | Cross-domain collaboration, privacy/verification evaluation and reproducibility handoff |

## Components

| Component | Responsibility | Current implementation |
|---|---|---|
| Data | Acquisition, normalization, source identity, split integrity, overlap/length audits and chunking | Available for the initial AU sources |
| Retrieval | Embeddings, vector caching, index construction and search | Qwen3 encoder, sharded exact-search index |
| Reranker | Candidate scoring, rank lineage, local training and future personalization | Qwen3 0.6B zero-shot backend, standard PEFT L0/L1 local training, explicit assumed negatives and checkpoint/resume |
| Gateway | Versioned batch I/O and evidence-boundary validation | Local L0 identity adapter; no privacy protection |
| Abstractor | Input translation and controlled query/evidence rewriting | Planned; L1–L3 algorithms not implemented |
| Generator | Cited generation, LoRA adaptation, shared/local experts and routing | Planned integration surface |
| Federation | Backend adapters and governed shared/local parameter exchange | Planned; no partner backend bundled |
| Evaluation | Metrics, paired statistics, future heterogeneity, grounding, privacy and verifier evaluation | Retrieval metrics and paired bootstrap available |
| Core / contracts / runtime | Shared records, artifacts, model assets and execution checks | Available; model imports are lazy |
| Workflows | Compose components into explicit, bounded execution flows | Initial data, native benchmark and corpus flows |

Current work is an initial implementation, not completion of the whole programme. Two native AU zero-shot baselines have historical recorded results; the full AU chunk index remains partial. Local PEFT reranker training and frozen-candidate evaluation are implemented with a [v2 portable contract](docs/contracts/reranker_training_v2.md); empirical results remain separate runtime artifacts. The [v1 synthetic contract](docs/contracts/reranker_training.md) is retained for provenance. Generators, abstraction algorithms and federated experiments remain later work. See the [implementation map](docs/components/README.md) and [recorded initial results](docs/experiments/zero_shot_native/results.md).

## Repository organization

```text
src/tapestry_m1/   component source code and workflow orchestration
configs/          source, component and experiment configuration
scripts/          source execution and environment selection
environments/     independent-stack conventions and template
tests/            component tests, integration checks and synthetic fixtures
docs/             architecture, contracts, component guides and operations
```

Run directly from the checkout with a uv-managed environment. The project is not installed as a Python distribution; no wheel or editable install is required. Existing data/evaluation/retrieval/reranking share the default `.venv`. A lightweight CPU profile is available, and future incompatible stacks can use independent locks/environments. No Conda environment is required.

```bash
./scripts/run --help
```

Start with the [documentation index](docs/README.md). [Environment setup](docs/environments/README.md), [architecture](docs/architecture/overview.md), [data preparation](docs/workflows/data_preparation.md), [native baselines](docs/workflows/native_baseline.md) and [development checks](docs/architecture/development.md) contain the operational details.

## Data and execution boundaries

This repository supplies code, source metadata and synthetic examples. It does not distribute datasets, weights, checkpoints, private institutional content or site credentials. Source access and permitted use must be reviewed for each collection; public availability is not blanket redistribution permission. Retrieval collections, training splits and evaluation labels retain explicit identities and provenance.

Experiments require a declared recipe and appropriate compute allocation. Historical observations do not validate a changed implementation or a new hardware/backend combination. Large acquisition, full indexing and future research experiments are separate execution decisions.
