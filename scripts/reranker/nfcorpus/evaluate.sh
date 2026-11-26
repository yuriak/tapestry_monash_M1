#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../environment.sh"
exec "$PYTHON" -m tapestry_m1.workflows.nfcorpus_training --stage evaluate --config "$M1_ROOT/configs/experiments/nfcorpus-dense-trajectories.json" "$@"
