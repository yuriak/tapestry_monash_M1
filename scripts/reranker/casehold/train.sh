#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../environment.sh"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$GPUS" -m tapestry_m1.workflows.casehold_epoch --config "$M1_ROOT/configs/reranker/training/casehold-epoch.json" "$@"
