#!/usr/bin/env bash
# Dense RGB input prepared by the workbench. Model weights must already exist.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${1:?Usage: run_vipe_workflow.sh RUN_ROOT [inference]}"
STAGE="${2:-inference}"
if [[ "$STAGE" != inference ]]; then echo "Only inference is supported" >&2; exit 2; fi
CONDA_BIN="${CONDA_BIN:-$HOME/miniforge3/bin/conda}"
eval "$("$CONDA_BIN" shell.bash hook)"
set +u
conda activate "${ENV_NAME:-vipe-scene}"
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME="$CONDA_PREFIX" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
VIPE_BUFFER="${VIPE_BUFFER:-176}"
if ! [[ "$VIPE_BUFFER" =~ ^[0-9]+$ ]] || (( VIPE_BUFFER < 32 || VIPE_BUFFER > 4096 )); then
  echo "VIPE_BUFFER must be an integer in [32,4096]" >&2
  exit 2
fi
cd "$PROJECT_ROOT"
exec python -m video2scene infer --video "$RUN_ROOT/prepared/input.mp4" \
  --output "$RUN_ROOT/vipe" --depth-model unidepth-l --buffer "$VIPE_BUFFER"
