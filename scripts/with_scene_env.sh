#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$ROOT/scripts/local_env.sh" ]; then source "$ROOT/scripts/local_env.sh"; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PATH="$HOME/miniforge3/envs/vipe-scene/bin:$PATH"
cd "$ROOT"
exec "$@"
