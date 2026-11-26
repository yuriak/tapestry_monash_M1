#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../environment.sh"
exec "$PYTHON" -m tapestry_m1.reranker.recipes.bioasq.prepare_jsonl "$@"
