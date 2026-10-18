# JSONL contract v1

UTF-8 JSON objects, one per line. Numeric labels/scores must be finite and not booleans. Empty document/query text is invalid. Blank lines are ignored. All examples in tests are synthetic.

## Identities and streams

Normalized IDs are `<scope>::<percent-encoded-original-id>` (UTF-8 URL encoding, no safe punctuation). Document scope is `collection_id`, query scope is `dataset_id`. Original string/integer source identities are retained as strings in `original_doc_id`/`original_query_id` plus file/revision/source-row provenance. Collection names are explicit config constants. A source version is recorded in the artifact path/manifest, and runs also carry index/config fingerprints; never combine different corpus versions under one collection without updating its identity.

| Stream | Required fields | Semantics |
|---|---|---|
| documents | collection_id, doc_id, text | Unique doc_id across artifact; optional title, original_doc_id, source_doc_id, jurisdiction, metadata, provenance |
| queries | dataset_id, split, query_id, query_text, collection_id | Identity tuple (dataset_id, split, query_id); same normalized query across splits in one dataset is rejected |
| qrels | dataset_id, split, query_id, doc_id, relevance | Unique query/doc judgement; nonnegative grade; >0 positive; original source label retained by adapter |
| rankings | dataset_id, split, query_id, collection_id, run_id, seed, fingerprints, candidates | Exactly one per selected query; seed integer or explicit null; fingerprints has nonempty model/index/config strings |
| training_pairs | dataset_id, split, pair_id, query_text, answer, positive_text, source_doc_id, provenance | QA-only: positive_text is source.text, formatted_text is the source question/answer concatenation; source-row generated identity marked explicitly; no implied corpus join or qrels |

A candidate is `{ "doc_id": "collection::source-id", "rank": 1, "score": 0.5 }`. Ranks are consecutive 1-based integers matching list order, IDs unique, scores finite. Equal scores keep supplied order. Scores are descriptive; metrics do not reorder them. Top-k lists may be shorter than 100 and may be empty; MRR's actual submitted depth is reported per query. `--split` requires a ranking file containing only that selected split. Missing query records are errors rather than dropped denominators.

The complete synthetic ranking shape is in [`rankings.jsonl`](../tests/fixtures/rankings.jsonl). Corpus text is not chunked; native benchmark passages remain passages. A later chunker must retain source-document lineage and resolve benchmark gold explicitly, without inventing new relevance labels. QA's source version IDs have not been joined to the current corpus; this is flagged and QA contributes no evaluation streams. JudicialMind AU rights are unresolved, so no body adapter or download is enabled.

## Integrity and manifests

SQLite-backed validation enforces unique corpus/query/qrel identities, namespace, nonempty text, exact split/collection joins, positive qrels for every evaluable query, and gold membership in the eligible collection. A missing gold document fails instead of shrinking the recall denominator. Sparse source labels do not establish exhaustive judgements. Same query identity or normalized text appearing across splits within a dataset is contamination and must be reviewed. Training-pair source splits are preserved; QA has only the official train split.

Normalization manifests contain the entire pinned source specification, original and output SHA-256, extraction version/options, native-boundary/unchanged-text policy, observed counts, proposal discrepancies, integrity counts and unresolved flags. Full extraction status is `validated`; source truncation is `partial`. This status denotes contract integrity, not licence clearance, exhaustive judgements, scientific reproducibility or model readiness. Outputs are finalized atomically after validation and reused only when all source/config/output hashes match.

## Metric definition and result contract

nDCG@10 divides top-10 DCG by ideal top-10 DCG over all source qrels for the query. Exponential gain is the default; linear gain must be selected and recorded. Relevance grades above 50 require explicit review for exponential gain. Discount is log2(rank+1). Recall@10/@100 divides judged-positive hits by all positive gold documents in the eligible candidate collection. Zero-positive queries are rejected. MRR is first positive reciprocal rank over the full submitted list unless an explicit depth is configured, in which case its key becomes `mrr@N`.

Evaluator output contains `schema_version`, `run` identity, `definition`, `input_sha256`, `integrity`, `counts` (evaluated/skipped/errors), grouped macro summaries, and `per_query`. Each per-query item contains dataset/split/collection/query identity, metrics, retrieved depth, judged-positive count and unjudged-retrieved count. Evaluation fails atomically on any input error; it never silently skips malformed queries. Result bytes are deterministic for identical inputs.

Paired comparison aligns exact query tuples within each benchmark/split/collection, checks identical input hashes and metric definitions, samples n query indices with replacement n times per replicate, and averages B-A per-query deltas. Every metric uses the same resampled indices. The reported percentile bounds interpolate linearly at .025 and .975; default seed 20261018, 10,000 resamples. Outputs record runs, seed, resample count, query identity hash and per-group means/intervals. Run seeds are never pooled. A self-comparison is a synthetic sanity check and has no model-performance interpretation.

## Audit limits

The overlap audit streams normalized corpus documents against the small benchmark collection. It records exact normalized-text hashes, literal full-benchmark-text containment for passages at least 80 characters, canonical URL matches, source counts, fingerprints and coverage. Normalization for matching is NFKC, casefold and whitespace collapse only; stored dataset text remains unchanged. It does not detect paraphrase, fragments, all duplicates or shared legal substance. No matches never proves uncontaminated evaluation; a partial corpus also leaves the unscanned source pending.
