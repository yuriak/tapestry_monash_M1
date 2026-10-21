# Component implementation map

The full programme includes reranker heterogeneity/personalization, abstract representations, shared/node generator experts, federation, privacy and verification. Directory ownership supports those workstreams without claiming they already exist.

| Component | Implemented | Later scope |
|---|---|---|
| [Data](data.md) | Initial AU source contracts/adapters and chunk preparation | Reviewed global/medical/partner adapters and split/leakage policies |
| [Retrieval](retrieval.md) | Encoder, vectors, exact search and partial index builder | Larger collections and retriever adaptation |
| [Reranker](reranker.md) | Zero-shot scorer and fixed-candidate ordering | Training, local/shared heads, calibration, mixtures, distillation |
| [Gateway](gateway.md) | Local L0 batch validation | Agreed partner transport contracts |
| [Abstractor](abstractor.md) | Ownership placeholder only | Translator and controlled L1–L3 transformations |
| [Generator](generator.md) | Ownership placeholder only | Cited generation, adapters, MoE experts/router |
| [Federation](federation.md) | Ownership placeholder only | Accepted framework adapters and shared/local exchange |
| [Evaluation](evaluation.md) | Retrieval metrics and paired statistics | E3, grounding, privacy, verifier/calibration |
| [Runtime](runtime.md) | Assets/config/device/provenance services | Reviewed model/serving backends |

A placeholder has no executable claim or speculative dependency. Add implementations against explicit contracts when the corresponding workstream starts. Cross-component commands live under `workflows`.
