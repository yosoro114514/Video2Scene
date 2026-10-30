#!/usr/bin/env bash
# Stage-specific resume: inference and reconstruction outputs remain independent.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="$1"
STAGE="$2"
if [ -f "$PROJECT_ROOT/scripts/local_env.sh" ]; then source "$PROJECT_ROOT/scripts/local_env.sh"; fi
CONDA_BIN="${CONDA_BIN:-$HOME/miniforge3/bin/conda}"
eval "$("$CONDA_BIN" shell.bash hook)"
set +u
conda activate "${ENV_NAME:-vipe-scene}"
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME="$CONDA_PREFIX" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$PROJECT_ROOT"
if [ "$STAGE" = inference ]; then
  VIPE_BUFFER="${VIPE_BUFFER:-176}"
  if ! [[ "$VIPE_BUFFER" =~ ^[0-9]+$ ]] || (( VIPE_BUFFER < 32 || VIPE_BUFFER > 4096 )); then
    echo "VIPE_BUFFER must be an integer in [32,4096]" >&2
    exit 2
  fi
  exec python -m video2scene infer --video "$RUN_ROOT/prepared/input.mp4" --output "$RUN_ROOT/vipe" --depth-model unidepth-l --buffer "$VIPE_BUFFER"
elif [ "$STAGE" = reconstruction ]; then
  exec python -m video2scene reconstruct --vipe-dir "$RUN_ROOT/vipe" --sequence input --output "$RUN_ROOT/scene" --segment yolo --weights "$PROJECT_ROOT/yolo11s-seg.pt" --segment-device 0 --stride 2 --segment-every 3 --confidence 0.30 --voxel 0.035 --max-depth 8 --max-faces 120000
else
  echo "Unknown stage" >&2
  exit 2
fi
