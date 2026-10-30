#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$PROJECT_ROOT/scripts/local_env.sh" ]; then
  source "$PROJECT_ROOT/scripts/local_env.sh"
fi
CONDA_BIN="${CONDA_BIN:-$HOME/miniforge3/bin/conda}"
eval "$("$CONDA_BIN" shell.bash hook)"
set +u
conda activate "${ENV_NAME:-vipe-scene}"
set -u
export CUDA_HOME="$CONDA_PREFIX"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
cd "$PROJECT_ROOT"
exec python -m video2scene "$@"
