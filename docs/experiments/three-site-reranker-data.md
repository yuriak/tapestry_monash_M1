# Three-site reranker data (v2)

Build one Monash, BharatGen and IITD view per benchmark. Each site obtains the authorized upstream sources locally. The existing script and data root retain their names so v1 and v2 can share prepared inputs. The default is v2. The [v1 guide](two-site-reranker-data.md) documents the historical partitions.

## Build and verify

Use Linux, Python 3.11, Git, Bash and flock. Authenticate locally with your approved IL-PCSR account; follow JudicialMind and original article access and usage conditions. The public repository contains code, rules and index digests. Generated examples, identifiers and qrels remain in the local data directory.

```bash
uv venv --python 3.11 .venv
export PYTHON="$PWD/.venv/bin/python"
uv pip install --python "$PYTHON" -r configs/reranker/two_site/requirements.txt
for ds in evidence_inference casehold bioasq; do
    bash scripts/reranker/two_site/build.sh "$ds"
done
for ds in il_pcsr judicialmind; do
    GPUS=4 bash scripts/reranker/two_site/build.sh "$ds"
done
```

The public entry point fetches pinned inputs, materializes all sites and verifies the native data loaders. A cold IL or JudicialMind retrieval needs CUDA. One to four allocated GPUs can execute the four logical embedding shards. Frozen ranking hashes detect hardware-dependent differences. A local preparation can be reused with `bash scripts/reranker/two_site/build.sh DATASET --prepared`; these builds use CPU only. The four existing input layouts are in the v1 guide. For JudicialMind provide `prepared/data.json`, `prepared/tokens.pkl` and native Top500 `retrieval/frozen.json`. Keep shared candidate corpora available only to authorized users.

By default each site directory is `data/two-site/DATASET/v2/SITE`. DATA_ROOT overrides the parent of DATASET. In each site use `--raw SITE` for CaseHOLD, `--data SITE` for EI, or `--data SITE/data --retrieval SITE/retrieval` for BioASQ and JudicialMind. For IL append `/lsr` or `/pcr` to both paths. The [centralized training guide](centralized-rerankers.md) describes the other arguments. Reserve separate model output directories. Standalone validation uses `PYTHONPATH=src "$PYTHON" -m tapestry_m1.reranker.two_site.verify DATASET --version v2 --root data/two-site/DATASET --config configs/reranker/two_site`.

## Frozen memberships

Counts below are distinct native queries, in train / dev / test order. EI includes all native prompts; training and primary Top-1 require aligned evidence. Each site directory has its own index.jsonl. The version directory contains the detailed counts and file checksums. The shareable reference_v2.json binds memberships and rankings by aggregate hashes.

| Dataset | Monash | BharatGen | IITD |
|---|---:|---:|---:|
| bioasq | 1,267 / 142 / 232 | 1,122 / 122 / 250 | 1,218 / 136 / 225 |
| casehold | 13,119 / 1,599 / 1,623 | 15,294 / 1,943 / 1,885 | 14,096 / 1,772 / 1,806 |
| evidence_inference | 3,614 / 441 / 441 | 3,210 / 385 / 392 | 3,292 / 411 / 391 |
| il_pcsr | 1,619 / 197 / 190 | 1,568 / 196 / 198 | 1,830 / 234 / 239 |
| judicialmind | 972 / 240 / 217 | 11,767 / 143 / 144 | 13,232 / 129 / 151 |

## Routing and supervision

- **IL-PCSR.** Delhi district/high courts favor Monash, western courts (Bombay, Gujarat, Rajasthan, Madhya Pradesh) favor BharatGen, and other regional courts favor IITD. Supreme Court and unknown sources use equal weights. The exact probabilities and aliases are in the v2 and base configuration. Duplicate judgments, LSR/PCR, gold relations and chunk selections follow one group. The institutions here simulate court-source distributions.
- **EI, CaseHOLD and mini-BioASQ.** Reuse the v1 training-only topic classifiers. Pharmacological, criminal and molecular themes respectively favor Monash; procedural/behavioral, civil and clinical themes favor BharatGen. IITD receives a mixture. The two topic routes use 50/17/33 and 17/50/33 percentages in site order; CaseHOLD civil uses 20/47/33 after inspecting training counts. Shared/mixed/unknown groups have equal weights. The native PMCID, citing context or deduplicated question is the assignment unit. No change to candidate order or supervision.
- **JudicialMind.** All AU queries belong to Monash. Indian records are connected by query identity and every known positive’s parent-document identities. A connected component belongs to one site across the native splits. The configuration lists public-law domains that favor BharatGen; other Indian domains favor IITD. In each preference stratum assign training components largest first to approach a 70/30 target (50/50 for ties); use a stable seeded hash for held-out-only components. Dev/test recall still uses the shared original Top500 candidate lists. AU Tax remains an external evaluation shared across views, never a local training source.
- JudicialMind hard-negative sampling retains locally available original negatives, then refills from eligible documents in the local jurisdiction/domain by TF–IDF rank. A deterministic local training-document fallback is used only if the stratum is exhausted. This occurs for two Monash rows, zero Indian rows. Known positives and same-parent passages remain excluded. There are 13,233 IITD training relations for 13,232 distinct training queries.

## Evaluation

Route pretrained, each of the three local adapters, FedAvg and any permitted pooled adapter to each site test and full native test. Select checkpoints on dev. Use the original metric calculation on the union (pooled EI AUROC and CaseHOLD confusion matrix). External AU Tax contributes no examples to the native full JudicialMind test.

Cached pretrained baseline scores (dev / test, percent) on the frozen partitions:

| Metric | Monash | BharatGen | IITD | Full |
|---|---:|---:|---:|---:|
| IL LSR MAP | 17.55 / 17.15 | 19.49 / 16.59 | 15.81 / 15.53 | 17.51 / 16.36 |
| IL PCR MAP | 11.32 / 9.29 | 11.98 / 13.15 | 13.74 / 12.77 | 12.43 / 11.84 |
| EI alignable Top-1 | 32.40 / 29.09 | 30.13 / 28.80 | 31.84 / 34.62 | 31.51 / 30.78 |
| CaseHOLD five-class Macro-F1 | 63.75 / 61.18 | 66.09 / 62.56 | 64.38 / 62.24 | 64.86 / 62.05 |
| Mini-BioASQ MAP@10 | 61.36 / 63.89 | 64.02 / 63.86 | 61.75 / 62.42 | 62.30 / 63.41 |
| JudicialMind native Top500 nDCG@10 | 98.09 / 97.79 | 97.50 / 99.05 | 98.68 / 98.11 | 98.08 / 98.24 |

JudicialMind Indian site test sizes are 144 and 151 queries. Differences of a few queries can visibly move site metrics. Full metrics use the original evaluation across the union.
