#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../environment.sh"
exec "$PYTHON" -m tapestry_m1.workflows.nfcorpus_pilot prepare --config "$M1_ROOT/configs/experiments/nfcorpus-pilot.json" "$@"
