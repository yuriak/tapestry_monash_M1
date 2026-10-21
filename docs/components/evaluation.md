# Evaluation

Available: strict retrieval evaluation, separately labelled scorer compatibility view and paired bootstrap. Other evaluation families are planned.

## Ownership

`evaluation/retrieval/`, `statistics/`, `cli.py`.

## Input/output boundary

Eligible corpus/query/qrels and complete ranking records -> per-query/grouped metrics and paired intervals.

## Extension direction

Future families: E3 heterogeneity/evidence-location audit; generation/grounding and jurisdictional violations; verifier/calibration; membership/attribute/expert-activation/update-inversion attacks. Keep task-specific definitions distinct; do not collapse character-span, ranked-document, answer or privacy metrics into one generic score.

See [architecture](../architecture/overview.md), [contracts](../contracts/data.md) and the [component index](README.md).
