# Reranker

Available: Qwen3 0.6B zero-shot scorer, candidate-preserving ordering and rank lineage. Training/personalization remain planned.

## Ownership

`reranker/qwen3.py`, `ranking.py`.

## Input/output boundary

Query + fixed eligible candidates + model config -> scores/reordered candidates with original retrieval rank/score retained.

## Extension direction

M1.1 extensions include pooled/local/cross-node recipes, shared/local heads, calibration, jurisdiction mixtures and abstract relevance distillation. Introduce each as a reviewed backend/method, keep training separate from score/evaluation contracts. Retrieval is supplied by its owning component.

See [architecture](../architecture/overview.md), [contracts](../contracts/data.md) and the [component index](README.md).
