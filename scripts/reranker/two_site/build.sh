#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../environment.sh"
dataset="${1:?Choose il_pcsr, evidence_inference, casehold or bioasq}"
case "$dataset" in
    il_pcsr|evidence_inference|casehold|bioasq) ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac
if [[ $# -gt 2 || ( $# -eq 2 && "$2" != "--prepared" ) ]]; then
    echo "Usage: build.sh DATASET [--prepared]" >&2
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
"$PYTHON" -m tapestry_m1.reranker.two_site.build "$dataset" --root "$root" --rules "$config/rules.json"
"$PYTHON" -m tapestry_m1.reranker.two_site.verify "$dataset" --root "$root" --config "$config"
