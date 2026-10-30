#!/usr/bin/env bash
# Run inside Ubuntu/WSL. Keeps large environments and model caches on Linux.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-$HOME/miniforge3/bin/conda}"
ENV_NAME="${ENV_NAME:-vipe-scene}"
eval "$("$CONDA_BIN" shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" -c nvidia/label/cuda-12.8.0 -c conda-forge --override-channels \
    python=3.12 uv cuda-nvcc eigen zlib libcusparse-dev libcublas-dev libcusolver-dev
fi
set +u  # NVIDIA's conda activation script reads optional unset variables.
conda activate "$ENV_NAME"
set -u
export CUDA_HOME="$CONDA_PREFIX"
# Build for the detected GPU, or an explicit architecture supplied by the user.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  echo "TORCH_CUDA_ARCH_LIST is unset; PyTorch will detect the visible GPU."
fi
export MAX_JOBS="${MAX_JOBS:-4}"
export UV_SYSTEM_CERTS=true
export USE_SYSTEM_EIGEN=1
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$HOME/vipe-scene-runtime}"
mkdir -p "$RUNTIME_ROOT"
if [ ! -d "$PROJECT_ROOT/third_party/vipe/.git" ]; then
  git clone https://github.com/nv-tlabs/vipe.git "$PROJECT_ROOT/third_party/vipe"
  git -C "$PROJECT_ROOT/third_party/vipe" checkout 8c9f36144e08d8f8c8cf60d20ad70c100d9cff07
fi
# Native compilation is considerably faster on the Linux filesystem than /mnt/e.
if [ ! -d "$RUNTIME_ROOT/vipe/.git" ]; then
  git clone "$PROJECT_ROOT/third_party/vipe" "$RUNTIME_ROOT/vipe"
fi
uv pip install --python "$CONDA_PREFIX/bin/python" 'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128' \
  --index-url https://pypi.org/simple \
  --find-links https://download.pytorch.org/whl/cu128/torch/ \
  --find-links https://download.pytorch.org/whl/cu128/torchvision/
uv pip install --python "$CONDA_PREFIX/bin/python" setuptools wheel ninja
python "$PROJECT_ROOT/scripts/patch_vipe_metadata.py" "$RUNTIME_ROOT/vipe"
uv pip install --python "$CONDA_PREFIX/bin/python" --no-build-isolation -e "$RUNTIME_ROOT/vipe"
uv pip install --python "$CONDA_PREFIX/bin/python" -e "$PROJECT_ROOT"
python -c 'import torch, vipe, open3d, trimesh; print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name()); print("Ready")'
