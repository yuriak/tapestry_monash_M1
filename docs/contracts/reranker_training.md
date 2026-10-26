# Reranker training contract v1

Status: implementation reference for partner review, not a claim of partner acceptance or federated training. This contract can be implemented independently of this repository. Tiny authored fixtures are non-sensitive and are not an accuracy benchmark.

## Model, serialization and score

Use `Qwen/Qwen3-Reranker-0.6B` and tokenizer revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`. The local tokenizer verifies **yes = 9693**, **no = 2152**, each a distinct single token. See [16 serialized synthetic pairs and token IDs](../../tests/fixtures/training/serialization.json). The body is:

```text
<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}
```

Prepend this exact prefix (the final newline is present):

```text
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
```

Append this suffix (two newlines after both `<think>` and `</think>`):

```text
<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Default instruction: `Given a web search query, retrieve relevant passages that answer the query`. Encode prefix/suffix without added special tokens; encode body with the pinned tokenizer's `add_special_tokens=True`. Reserve prefix/suffix token lengths, retain the body's first remaining tokens, then append the entire suffix. Use left padding and an attention mask, max total length 2048 including template. Count native/retained lengths and truncations. This is the existing inference policy at a bounded training context, not a new chat template.

Read `z_yes` and `z_no` at the last scoring position after the suffix. Training score **s = z_yes − z_no**, cast to float32 without detaching. Inference remains `softmax([z_no,z_yes])[yes]`, equivalently sigmoid(s); BCE training takes s directly, never this probability.

## Group and negative-selection contract

[m1.reranker.training.v1 groups](../../tests/fixtures/training/groups.jsonl) have one query and four ordered candidates: positive, random negative, hard negative, hard negative. Required fields include collection/query/original-source identity, split, source hash, provenance, seed and candidate text/hash, label, original document identity, negative kind, supervision and ambiguity. Hash fields are SHA-256 of the exact supplied UTF-8 query or candidate text; upstream raw-file digests belong to the input manifest separately.

Split source/query identities before mining. Exclude known relevant IDs, duplicate candidates, query/source/text overlap across splits and all benchmark test material. Random candidates come only from the allowed training-side pool; hard candidates come from a versioned first-stage ranking on those training queries. An unjudged candidate is ambiguous, not certified irrelevant. The reference selector fails closed unless it has three distinct explicitly judged nonrelevant train candidates, two in the frozen ranking. A different reviewed ambiguity policy must be separately versioned, never relabeled as ground-truth negatives.

The four authored queries are two train and two dev groups with disjoint source identities/text. Hard/random tags exercise the contract; they do **not** assert actual retriever mining or hardness. The local training gate accepts only these declared synthetic inputs. Real training requires a separately approved source/negative/split eligibility contract; changing a CLI flag cannot bypass that gate.

## Losses and expected gradients

For a group with positive score p and negative scores n1,n2,n3:

```text
BCE(g)    = [softplus(-p) + softplus(n1) + softplus(n2) + softplus(n3)] / 4
margin(g) = sum_i max(0, m - p + ni) / 3
L0        = mean_groups BCE(g)
L1        = mean_groups [BCE(g) + lambda * margin(g)]
m = 1.0; lambda = 0.5
```

At the hinge boundary use zero subgradient, matching ReLU. Pairwise accuracy is the mean of strict `p > ni` over the named held-out labelled comparisons; ties fail. It is a diagnostic, not a retrieval metric.

[Independent scalar vectors](../../tests/fixtures/training/expected.json) use scores `[[1.2,0.4,-0.3,0.8],[-0.2,0.5,-1.0,0.1]]`: BCE **0.7164534789134727**, margin **0.6666666666666667**, L0 **0.7164534789134727**, L1 **1.049786812246806**. The file gives derivatives with respect to every score. A single pair p=1.2,n=0.4 gives hinge **0.2**. These hand-selected logits are algebra references, not predicted model outputs. Float64 CPU algebra/autograd should agree to 1e-12; GPU BF16 results require declared precision/tolerance.

For gradient accumulation, each complete microbatch group mean is weighted by `micro_groups/effective_groups` before backward. Never average an incomplete candidate group as if complete. Effective batch is eight query groups = 32 pairs, independent of the chosen microbatch.

## Adapter and optimization recipe

- Frozen base; LoRA q_proj/v_proj, rank 16, alpha 32, dropout 0.05. Native torch backend: `base(x) + (alpha/r) * B(A(dropout(x)))`, A Kaiming-uniform, B zero; adapter matrices and optimizer states float32, base BF16. Adapter update is cast back to base activation dtype. No merged base weights.
- Only named `lora_A`/`lora_B` parameters are trainable and present in AdamW. Attention module inventory and parameter counts must be observed, not guessed. At initialization A's gradient can be zero because B starts zero; verify finite nonzero adapter gradients and subsequent updates.
- AdamW lr 1e-4, weight decay 0.01, defaults beta=(0.9,0.999), eps=1e-8, max grad norm 1.0. Constant learning-rate scheduler is explicit and checkpointed.
- Seed 20261026, initial max length 2048, effective eight groups, initial microbatch one group. 100 optimizer updates per objective; same source/order/seed/adapter initialization/schedule/truncation for L0/L1. The two tiny train groups cycle through deterministic shuffled epochs; repeated exposure is intentional synthetic engineering smoke.
- The native adapter format is **not PEFT's file format**. Other teams may use PEFT or another backend while matching math, targets, initialization and reported reductions. No framework-source dependency is required for contract compatibility. The existing pinned Torch/Transformers stack is retained.

## Checkpoint and comparison requirements

Checkpoint at complete optimizer steps: adapter tensors, optimizer, scheduler, step, sampler epoch/order/cursor and RNG, Python/Torch/CUDA RNG states, full config/input/model/code/environment-lock hashes, dev score vector, best dev loss/step and latest/best selection. Files have SHA-256 manifests and immutable checkpoint directories. Only load locally generated trusted continuation state after hash/identity checks; do not load arbitrary downloaded pickle checkpoints.

Verify scores before/after adapter save/reload (local fixed-hardware tolerance 1e-5), continuation advances beyond the stored step and sampler order does not repeat or skip batches. CPU dropout/optimizer tests compare uninterrupted vs resumed tensors exactly. Record finite gradients, actual trainable names/count, losses/components, norms, lr, consumed group IDs, truncations, throughput, peak VRAM, hardware/software and checkpoint digests. A loss decrease on synthetic groups does not show useful generalization.

Evaluation of a lawful real pilot must rerank **exactly** frozen baseline Top-100 IDs with native held-out qrels: nDCG@10, Recall@10/@100, MRR and per-query records. Select settings/checkpoints on train/dev only. Paired query-bootstrap intervals, if used, do not represent training-seed variation. Synthetic-trained AU test scores are at most an explicitly labeled interface diagnostic; they are not AU adaptation or formal local/pool/cross-node research results.

## Portable checks and execution

From the source checkout:

```bash
./scripts/run --environment cpu training fixture tests/fixtures/training/groups.jsonl tests/fixtures/training/expected.json
./scripts/run pytest -q tests/reranker/test_training_contract.py tests/reranker/test_lora_training.py
./scripts/run training train configs/reranker/training/synthetic-warmup.json --data-root "$M1_DATA_ROOT" --output "$M1_RUN_ROOT/preflight" --preflight
./scripts/run training train configs/reranker/training/synthetic-warmup.json --data-root "$M1_DATA_ROOT" --output "$M1_RUN_ROOT/L0-first" --stop-after 50
./scripts/run training train configs/reranker/training/synthetic-warmup.json --data-root "$M1_DATA_ROOT" --output "$M1_RUN_ROOT/L0-resumed" --resume "$M1_RUN_ROOT/L0-first/checkpoints/step-000050"
```

GPU commands require an appropriate running allocation through the existing device guard; allocation details belong to the operator, not this contract. Outputs are fresh ignored/private directories, never Git. Repeat the two segments with `--objective L1`, separate outputs, same configuration. The source repository needs no wheel/install step or new dependencies. Environment and model digests in each run/checkpoint identify exact runtime inputs. Pin this contract and fixtures to the delivered Git commit and file SHA-256; partner sign-off is independent.

Primary definitions: [Qwen model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/tree/e61197ed45024b0ed8a2d74b80b4d909f1255473), [LoRA](https://arxiv.org/abs/2106.09685), [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html), [MarginRankingLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html).

The [contract identity manifest](../../configs/reranker/training/contract_identity.json) pins the exact contract and fixture SHA-256 values independently of any trainer implementation. Use a Git permalink to that manifest when proposing partner adoption.
