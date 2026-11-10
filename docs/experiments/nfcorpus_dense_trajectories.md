# NFCorpus dense-retrieval reranker trajectories

`workflows.nfcorpus_training` consumes immutable native NFCorpus source and completed
Qwen3 embedding coverage artifacts. It reuses existing encoding, exact search,
reranker serialization, PEFT backend, differentiable group scoring and full-state
checkpoint implementations. `configs/experiments/nfcorpus-dense-trajectories.json`
fixes Top100, two 200-step L0/L2 arms and checkpoints every25 steps.

The workflow's `--stage preflight` validates frozen data, metrics and actual GPU
training/checkpoint continuation before `--stage run`. Supply explicit `--config`,
`--raw`, `--retrieval`, `--output` and `--data-root` paths using the configured runtime.
Do not use an output directory with a different source/config/environment identity.

Training uses only native train supervision and one immutable seeded schedule of
1,600 unique query groups. One seeded known positive, one random assumed negative,
and two embedding-hard assumed negatives are frozen per group. All known positives,
normalized duplicate texts and aliases detected by the existing SimHash/Jaccard
helper are excluded from negatives. This approximate alias detector does not certify
semantic distinctness. Positive supervision may lie outside the retrieved Top100;
held-out candidates are never augmented. No unjudged document is called a proven
negative. L0 is unchanged BCE; L2 is unweighted four-way softmax CE, not graded CE.

Both trajectories complete before evaluation. Full dev/test curves include the base
reranker and all16 adapters; raw embedding rankings are the retrieval reference.
Model summary selection uses only dev graded nDCG@10 with earlier-step tie breaks.
Repeated test observations are descriptive diagnostics, not a fresh untouched test
for subsequent tuning. Query bootstrap intervals use one training seed and have no
multiple-comparison correction.

Outputs include immutable source/group manifests, full optimizer/RNG checkpoints,
per-step logs, resumable per-query predictions and truncation audits, per-query
metrics, dev-only selection, paired confidence intervals and `curves.json`.
`completed.json` is emitted only after all requested systems finish. An execution
failure is not completion. Logs include UTC timestamps and explicit stage/model,
step/query counters and ETA. Resumed training may append replayed steps since the
last saved checkpoint; checkpoint identities are authoritative.
