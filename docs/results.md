# M1-Task-02 AU zero-shot baseline results

Both AU benchmarks use native eligible collections and complete query sets: Tax Guidance 105 documents / 112 queries; Legal RAG Bench 4876 passages / 100 queries. X1.1 uses Qwen3-Embedding-4B; X1.2 only reorders its unchanged Top-100 with Qwen3-Reranker-0.6B. No training or gold candidate injection.

Model commits: embedding `5cf2132abc99cad020ac570b19d031efec650f2b`, reranker `e61197ed45024b0ed8a2d74b80b4d909f1255473`. BF16 / SDPA / 8192 maximum tokens, full 2560-dimensional normalized embeddings, exact cosine retrieval. Both stages share one locked uv environment. See [protocol and reproduction commands](inference.md) and checked-in run/model configs.

## Primary Task-01 metric protocol

Values are fractions, not percentages. nDCG@10 uses exponential gain/log2 discount; ties follow submitted score-desc/doc-ID-ascending order. MRR is over submitted Top-100. This is one frozen deterministic zero-shot inference recipe (seed null), not three independent trained seeds.

| Benchmark | Stage | nDCG@10 | Recall@10 | Recall@100 | MRR |
|---|---|---:|---:|---:|---:|
| Tax Guidance | X1.1 | 0.7719 | 0.9464 | 1.0000 | 0.7194 |
| Tax Guidance | X1.2 | 0.7044 | 0.9018 | 1.0000 | 0.6463 |
| Legal RAG Bench | X1.1 | 0.4537 | 0.6600 | 0.9600 | 0.4013 |
| Legal RAG Bench | X1.2 | 0.4743 | 0.6900 | 0.9600 | 0.4175 |

Query-paired B-A deltas (X1.2-X1.1), 10000 resamples, seed 20261018, 95% percentile CI:

| Benchmark | ΔnDCG@10 [CI] | ΔRecall@10 [CI] | ΔRecall@100 [CI] | ΔMRR [CI] |
|---|---|---|---|---|
| Tax Guidance | -0.0675 [-0.1173, -0.0209] | -0.0446 [-0.0893, +0.0000] | +0.0000 [+0.0000, +0.0000] | -0.0731 [-0.1322, -0.0166] |
| Legal RAG Bench | +0.0206 [-0.0478, +0.0890] | +0.0300 [-0.0600, +0.1100] | +0.0000 [+0.0000, +0.0000] | +0.0162 [-0.0580, +0.0901] |

Tax Guidance nDCG decreases by 0.0675, with a 95% paired interval entirely below zero. This is an observed regression under the frozen recipe, not a reason to rewrite the completed run; attribution has not been tested. The Legal RAG Bench nDCG interval contains zero; its positive point estimate is not evidence of a clear improvement under this interval. Recall@100 remains equal because reranking preserves exactly the same candidate set. Query bootstrap describes conditional query-sampling uncertainty, not training-seed variance.

## Resources and limits

Measured on NVIDIA L40S. Time includes model loading and actual stage work; peak values are PyTorch allocated VRAM, not a device-wide NVML peak. Cached/resumed runs must retain their own timing interpretation.

| Benchmark | X1.1 seconds | X1.2 seconds | Peak allocated GiB X1.1 / X1.2 | Truncated docs / queries / pairs |
|---|---:|---:|---:|---:|
| Tax Guidance | 61.5 | 2531.0 | 13.39 / 8.89 | 1 / 0 / 160 |
| Legal RAG Bench | 164.9 | 112.1 | 7.94 / 1.57 | 0 / 0 / 0 |

Length/ID audits accompany ignored run artifacts. Native source text and eligible IDs remain unchanged; any model-input truncation is explicit. A separate linear-gain/doc-ID-desc tie view is retained; it is not a full official task runner or verified reproduction of a published Appendix B reference. Do not interchange its values with the headline protocol.

The full AU corpus is a separate workstream. The bounded real index includes 1001 chunks and 202 fully indexed source documents; 1024 source documents / 5755 chunks were staged. It is **partial**, against 232560 source documents. No corpus-wide relevance/accuracy claim is made. The measured prefix extrapolates to roughly 60 single-L40S encoding hours, with substantial sampling/I/O uncertainty and additional CPU staging cost; a complete build was deferred in this bounded run. Resume/extend the documented chunk/index pipeline when a larger budget is assigned.
