#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../environment.sh"
source "$M1_ROOT/configs/reranker/recipes/bioasq/train_basic.sh"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node="$GPUS" -m tapestry_m1.reranker.recipes.bioasq.train "${recipe[@]}" "$@"
