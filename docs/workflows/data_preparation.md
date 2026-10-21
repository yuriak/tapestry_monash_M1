# Source acquisition and normalization

Use the [environment guide](../environments/README.md) first. All data remains outside Git, defaulting to ignored `data/`; `M1_DATA_ROOT` or `--data-root` selects an explicit asset root.

```bash
./scripts/run --environment cpu data inventory configs/data/sources/legal-rag-bench.json --output outputs/legal-rag-source.json
./scripts/run --environment cpu data acquire configs/data/sources/legal-rag-bench.json --max-bytes 33554432 --output outputs/legal-rag-download.json
./scripts/run --environment cpu data normalize configs/data/sources/legal-rag-bench.json
./scripts/run --environment cpu data validate data/normalized/au-legal-rag-bench/db0b31dc6d195ce9916897e1ac5e4e6209736c8a
```

`inventory` only fetches source metadata; it never downloads bodies or runs normalization. `acquire` pins exact source/file identity, bounds **each file** and validates trusted upstream LFS hashes when present; it also records local SHA256 and acquisition time. Sum the file allowlist separately for an aggregate transfer budget. Resume requires expected identity and correctly honored HTTP Range; mismatches fail rather than replace retained artifacts.

Raw files are under `raw/<dataset>/<revision>`; normalized streams/manifests under `normalized/<dataset>/<revision>`. Source documents, queries, qrels and training pairs have explicit roles. Normalization preserves native text/boundaries. SQLite-backed validation checks joins/IDs/splits; a missing gold fails rather than shrinking the denominator. A preview `normalize --max-rows N --output <new-root>` bounds conversion of already acquired data, not transfer.

```bash
./scripts/run --environment cpu data audit <normalized-tax-root> <normalized-corpus-root> --output outputs/overlap.json
```

The overlap audit is literal/hash/available-URL analysis, not proof of semantic non-overlap. Streamed large-source validation still consumes CPU/I/O and needs an appropriate budget.

Pinned initial sources: Tax (105 documents/112 queries), Victorian Legal RAG Bench (4,876 passages/100 queries), full AU corpus (232,560 documents), AU legal QA (2,124 observed train pairs), JudicialMind metadata-only. Source cards/proposal estimates can differ from observed pinned counts. Review each config's rights/access flags; don't resolve them by changing a source label or treating QA as qrels. JudicialMind body acquisition remains blocked by its config. Global/medical families need reviewed source adapters/configs and leakage/split policy before acquisition.
