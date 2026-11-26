# NFCorpus pilot v1: source audit and dev baseline

The portable entrypoint is `python -m tapestry_m1.workflows.nfcorpus_pilot`.
Use explicit `--config`, `--output`, and either `--archive` (prepare) or
`--data-root` (preflight/baseline). Outputs must be outside tracked source.

`prepare` validates official archive bytes, native IDs/graded qrels and cross-split
query-text duplicates. Test qrels are accessed only by the isolated join/count audit;
they do not enter BM25, negative mining, selection or metrics. Shared documents are
not split. Frozen retrieval uses title+newline+text, lowercase alphanumeric tokens,
positive-IDF Okapi BM25(k1=1.2,b=.75), unique query terms, stable ID ties and Top50.
No stemming, stop list, gold injection or candidate-depth adaptation is performed.

The pilot config declares a conservative operator coverage gate of at least .80
query hit@50 in both train/dev; this is not a universal NFCorpus quality criterion.
A failing gate prevents GPU stages. A changed gate/config requires an explicit new
versioned decision and fresh output namespace, retaining the original measurements.

`preflight` checks pinned model/token IDs, batch1/2 consistency and long-input
truncation; `baseline` repeats that check then resumes fingerprinted per-query dev
scoring. Both use unchanged Qwen serialization, BF16/SDPA, max2048 and raw float32
logit differences. Metrics use exponential graded nDCG@10, binary-positive Recall
and MRR at50. Training and sealed test evaluation are not implemented in this
Stage-A module. They must not be inferred from a completed source/retrieval stage.

Academic-use terms and source attribution are recorded in the config and runtime
manifest. Never commit the archive, raw texts, queries, qrels or scoring caches.
