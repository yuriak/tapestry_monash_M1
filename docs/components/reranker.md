# Reranker

Available: Qwen3 0.6B zero-shot scorer, candidate-preserving ordering and rank lineage. Active standard PEFT supports L0/L1 local QA excerpt training, explicit assumed negatives, checkpoint/resume and frozen-candidate evaluation. See [contract v2](../contracts/reranker_training_v2.md); [v1](../contracts/reranker_training.md) remains historical.

## Ownership

`reranker/qwen3.py`, `ranking.py`; training owns `losses.py`, `peft_backend.py`, `peft_training.py`, `batches.py`; `lora.py`/`training.py` are retained historical v1 helpers. The workflow CLI composes this component; runtime owns checkpoint I/O.

## Input/output boundary

Query + fixed eligible candidates + model config -> scores/reordered candidates with original retrieval rank/score retained.

## Extension direction

M1.1 extensions include pooled/local/cross-node recipes, shared/local heads, calibration, jurisdiction mixtures and abstract relevance distillation. Introduce each as a reviewed backend/method, keep training separate from score/evaluation contracts. Retrieval is supplied by its owning component.

See [architecture](../architecture/overview.md), [contracts](../contracts/data.md) and the [component index](README.md).

Frozen base-reranker [hard-negative selection](../contracts/reranker_hard.md) keeps query/positive/random identities fixed and feeds the standard single-L0 trainer. Failure analysis is maintained independently in research.
