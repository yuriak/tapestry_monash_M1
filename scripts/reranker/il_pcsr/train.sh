#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../environment.sh"
source "$M1_ROOT/configs/reranker/recipes/il_pcsr/train.sh"
if [[ "$GPUS" == 4 ]]; then
    exec "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
        -m tapestry_m1.reranker.recipes.il_pcsr.screen "${recipe[@]}" "$@"
fi
for arm in 0 1 2 3; do
    "$PYTHON" -m tapestry_m1.reranker.recipes.il_pcsr.screen --rank "$arm" --local-rank 0 "${recipe[@]}" "$@"
done
