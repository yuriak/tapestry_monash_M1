#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../environment.sh"
dataset="${1:?Choose il_pcsr, evidence_inference, casehold, bioasq or judicialmind}"
case "$dataset" in
    il_pcsr|evidence_inference|casehold|bioasq|judicialmind) ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac
if [[ $# -gt 2 || ( $# -eq 2 && "$2" != "--prepared" ) ]]; then
    echo "Usage: build.sh DATASET [--prepared]" >&2
    exit 2
fi
version="${SPLIT_VERSION:-v2}"
case "$version" in
    v1) rules=rules.json ;;
    v2) rules=v2.json ;;
    *) echo "Unknown SPLIT_VERSION: $version" >&2; exit 2 ;;
esac
if [[ "$version" == v1 && "$dataset" == judicialmind ]]; then
    echo "JudicialMind site views require v2" >&2
    exit 2
fi
root="${DATA_ROOT:-$M1_ROOT/data/two-site}/$dataset"
config="$M1_ROOT/configs/reranker/two_site"
mkdir -p "$root"
exec 9>"$root/.build.lock"
flock 9
if [[ "${2:-}" != "--prepared" ]]; then
    "$PYTHON" -m tapestry_m1.reranker.two_site.acquire "$dataset" --root "$root" --scripts "$M1_ROOT/scripts/reranker" --config "$config"
fi
"$PYTHON" -m tapestry_m1.reranker.two_site.build "$dataset" --root "$root" --rules "$config/$rules"
"$PYTHON" -m tapestry_m1.reranker.two_site.verify "$dataset" --version "$version" --root "$root" --config "$config"
