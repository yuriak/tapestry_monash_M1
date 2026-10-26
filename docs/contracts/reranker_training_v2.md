# Reranker training contract v2 — standard PEFT and local QA excerpts

This version retains the [v1 mathematical fixture](../../tests/fixtures/training/expected.json) and [v1 synthetic contract](reranker_training.md) unchanged. v1 native adapters remain historical and are not PEFT files. The active CLI uses PEFT; the native implementation is retained only for historical compatibility/tests. Neither contract implies partner adoption or formal pooled/cross-node results.

## Source and split

The approved local collection is `isaacus/open-australian-legal-qa@a2178afc2cff8af50fe4365c7d6f9cecf91dbe4b`, 2,124 source rows. Positives are unchanged `source.text` excerpts, never top-level QA-formatted `text`. Raw and normalized source hashes must match the pinned manifests. This separate collection does not assert full-corpus relevance joins; source versions not present in the larger corpus retain their original excerpt identity.

Split approximately 85/15 by connected source-version/query/excerpt identities before retrieval. Exact normalized duplicates are connected; detected near duplicates use deterministic 64-bit SimHash blocking (Hamming <=3, four 16-bit bands) and word-trigram Jaccard >=.85 for excerpts / .90 for queries. Benchmark query/text exact and detected near matches are excluded; positive-in-test containment uses normalized text (>=80 chars) and shingle blocking. These checks do not establish semantic or exhaustive fragment leakage freedom. Source aliases remain within one split.

Private outputs preserve native excerpts, original source metadata and row provenance. Passage IDs hash exact UTF-8 text in an independent namespace. Query IDs retain source-row identity; `source_query_id` is an explicitly generated normalized-query hash because no authoritative upstream query ID is supplied. A source excerpt is a known positive for its QA query, not universal corpus qrels. Underlying attribution/source conditions remain; local research approval does not grant redistribution permission.

## Frozen same-split assumed negatives

Use pinned `Qwen/Qwen3-Embedding-4B@5cf2132abc99cad020ac570b19d031efec650f2b`, unchanged instruct/query formatting, last-token pooling, float32 L2 normalization and exact cosine/dot ranking. Train and dev get separate candidate universes and rankings. Cache checksummed embedding shards and freeze Top-100 ranking files, model/config/input digests and all sampled IDs before either objective trains. Deterministic full same-split expansion is allowed only when Top-100 cannot supply distinct candidates.

Each group: positive, one random negative, two highest eligible hard negatives. Exclude all known positives, their source-version aliases, duplicate/near-identical passages and pairwise repeated candidate sources. No benchmark queries, qrels, candidate texts or frozen test Top-100 enter mining. Random selection uses sorted eligible IDs and a per-query deterministic seed. If distinct negatives are unavailable after full same-split expansion, drop the query with a reason; never borrow test/dev material for train.

The explicit schema is `m1.reranker.training.v2`, `synthetic=false`. Candidate `split` must equal group split. Negatives use `label=0`, `ambiguous=true`, `supervision=assumed`; positives use `label=1`, `ambiguous=false`, `supervision=source_positive`. These negatives may contain false negatives and are **not** certified nonrelevance. v1 validators stay strict and reject this opt-in schema. Preserve IDs, exact text SHA-256, source aliases/rows, seed, negative kind, similarity/rank and frozen ranking identity.

## PEFT, loss and epochs

Use Qwen3-Reranker-0.6B/tokenizer `e61197ed45024b0ed8a2d74b80b4d909f1255473`; yes=9693/no=2152 and the same prefix/body/suffix, suffix-preserving body truncation and final scoring position as v1. Train score is differentiable `z_yes-z_no`; inference remains normalized yes/no probability.

L0/L1 formulas and within-candidate then across-group reduction are exactly v1. Margin=1, weight=.5; effective batch=8 groups, microbatch initially one group. The last batch of each epoch contains all remaining complete query groups and is weighted by actual group count: each microbatch mean contributes `len(micro)/len(effective_tail)`. Do not pad, drop or cross the epoch boundary. Two epochs imply `2*ceil(train_groups/8)` optimizer steps, not a synthetic 100-step cap.

PEFT 0.15.2 + Accelerate 1.6.0 are pinned with the existing Torch/Transformers stack. Use `LoraConfig(task_type=CAUSAL_LM,r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none')`. AdamW lr=1e-4, weight_decay=.01, default betas/eps, gradient clip=1 and an explicit constant scheduler. BF16 base and PEFT's trainable adapter dtype are observed and logged. Seed/order/negative manifest/start state/epochs are paired across objectives. Training context is 2048; native benchmark inference intentionally restores 8192 and uses conservative batch=1.

Select the best adapter on **dev BCE for both objectives**, using the same checkpoint schedule; retain latest and best paths. Freeze selections before looking at either native test result. Synthetic active-hinge preflight uses a detached constant score offset to construct a violation, keeping the loss formula and gradient path unchanged; its numbers are engineering evidence, not research performance.

## Standard adapter and true continuation

Each immutable checkpoint contains `adapter_config.json`, `adapter_model.safetensors` from `save_pretrained(safe_serialization=True)` and a separate `continuation.pt` with optimizer/scheduler, sampler order/cursor/epoch and Python/Torch/CUDA RNG. Manifest records base/tokenizer revision, config/group/model/code/lock/runtime-version hashes, step/epoch, dev scores, best/latest selection and file SHA-256. Standard `PeftModel.from_pretrained(pinned_base,checkpoint,is_trainable=...)` must work independently of native-v1 classes.

Round-trip scoring uses fixed-input eval mode and a declared 1e-5 local tolerance; resume must advance optimizer/sampler without repeating or skipping completed batches. Adapter-only loading does not prove continuation. Only load trusted local full-state pickle after integrity checks. Output and checkpoints stay in ignored private storage; no raw examples or adapter release is implied.

## Evaluation and execution

Only rerank original stored X1.1 Top-100 IDs on both native held-out AU collections. Verify frozen ranking and normalized input hashes, candidate invariance and Recall@100 invariance. Reuse the existing linear-gain, submitted Top-100 MRR metric semantics and query-paired 10,000-resample/95% percentile comparisons: L0−zero-shot, L1−zero-shot, L1−L0. Report all queries and any regression; query bootstrap is not training-seed variability. A single paired seed is preliminary.

```bash
./scripts/run python -m tapestry_m1.workflows.au_peft prepare configs/reranker/training/au-qa-peft.json --data-root "$M1_DATA_ROOT" --run-root "$M1_RUN_ROOT"
./scripts/run python -m tapestry_m1.workflows.au_peft preflight configs/reranker/training/au-qa-peft.json --data-root "$M1_DATA_ROOT" --run-root "$M1_RUN_ROOT"
./scripts/run python -m tapestry_m1.workflows.au_peft pipeline configs/reranker/training/au-qa-peft.json --data-root "$M1_DATA_ROOT" --run-root "$M1_RUN_ROOT"
```

GPU commands require an appropriate allocation. The pipeline runs preflight, resumable split-specific mining, a five-step real-data segment plus full continuation for each objective, then both selected-adapter evaluations and paired comparisons. `status.json` reports phase/failure; `complete.json` is written only after everything succeeds. Reinvoke the **same frozen source/config/environment** to continue from checked embedding shards, latest training checkpoint, or per-query evaluation cache. Interrupted nonfinal segment directories are retained; continuation uses a fresh segment directory. No site accounts, hosts or absolute storage assumptions belong in this public contract.

The [v2 identity manifest](../../configs/reranker/training/contract_identity_v2.json) pins these methods and the unchanged v1 numeric/serialization vectors.
