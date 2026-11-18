# Sourced by task scripts. Use the active Python environment and local source checkout.
M1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$M1_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-1}"
export M1_ENVIRONMENT_LOCK="${M1_ENVIRONMENT_LOCK:-$M1_ROOT/uv.lock}"
