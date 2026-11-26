#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../environment.sh"
exec "$PYTHON" -m tapestry_m1.workflows.casehold_pilot --config "$M1_ROOT/configs/reranker/training/casehold.json" "$@"
