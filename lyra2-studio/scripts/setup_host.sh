#!/usr/bin/env bash
# Set up a Lyra 2.0 inference environment on an NVIDIA GPU host.
#
# Mirrors the official Lyra-2/INSTALL.md (tested on Ubuntu 22.04 + CUDA 12.8 + H100).
# Requires: Linux, an NVIDIA GPU (Ampere/Hopper/Blackwell), and conda/miniconda.
#
# Usage:
#   bash scripts/setup_host.sh [/path/to/clone/dir]
# After it finishes, set:
#   export LYRA2_HOME=<clone_dir>/lyra/Lyra-2
# No 'nounset' (-u): conda's CUDA/compiler activate+deactivate scripts reference
# unbound vars (CONDA_BACKUP_GXX, NVCC_PREPEND_FLAGS, ...) and would abort the script.
set -eo pipefail

CLONE_DIR="${1:-$HOME/lyra-src}"
ENV_NAME="${LYRA2_ENV:-lyra2}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: Lyra 2.0 inference requires Linux + an NVIDIA GPU. Detected: $(uname -s)." >&2
  echo "Run this on a GPU host (cloud instance or workstation), not on macOS/Windows." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. Lyra 2.0 needs an NVIDIA GPU (H100-class recommended)." >&2
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Newer conda requires accepting channel Terms of Service before creating envs.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "== [1/8] Clone Lyra (recursive) into $CLONE_DIR =="
mkdir -p "$CLONE_DIR"
if [[ ! -d "$CLONE_DIR/lyra/.git" ]]; then
  git clone --recursive https://github.com/nv-tlabs/lyra.git "$CLONE_DIR/lyra"
else
  echo "  already cloned; updating submodules"
  git -C "$CLONE_DIR/lyra" submodule update --init --recursive
fi
cd "$CLONE_DIR/lyra/Lyra-2"

echo "== [2/8] Create conda env '$ENV_NAME' (python 3.10) =="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "  env '$ENV_NAME' already exists; reusing it"
else
  conda create -n "$ENV_NAME" python=3.10 pip cmake ninja libgl ffmpeg packaging -c conda-forge -y
fi
conda activate "$ENV_NAME"
CONDA_BACKUP_CXX="" conda install -y --override-channels -c conda-forge gcc=13.3.0 gxx=13.3.0 eigen zlib

echo "== [3/8] CUDA toolkit (12.8) inside the env =="
# --override-channels avoids the broken 'defaults' resolution that yields
# "nothing provides __win needed by cuda" (+ cascading gcc conflicts). conda-forge
# supplies the compiler deps the toolkit needs.
conda install -y --override-channels -c nvidia/label/cuda-12.8.0 -c conda-forge cuda
# transformer-engine's source build needs the NVTX dev headers (nvtx3/nvToolsExt.h),
# which the 'cuda' metapackage doesn't always pull in.
conda install -y --override-channels -c nvidia/label/cuda-12.8.0 -c conda-forge cuda-nvtx-dev || true
export CUDA_HOME="$CONDA_PREFIX"

echo "== [4/8] PyTorch 2.7.1 (cu128) =="
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128

echo "== [5/8] Build environment variables =="
SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
# Bake LD_LIBRARY_PATH into the env so `conda activate $ENV_NAME` sets it automatically --
# Lyra needs it at runtime to find CUDA libs (else "library not found" in every new shell).
conda env config vars set LD_LIBRARY_PATH="$LD_LIBRARY_PATH" >/dev/null 2>&1 || true

# Parallelise the source builds (flash-attn / transformer-engine / extensions) across
# all cores, but cap by RAM (~4GB per nvcc job) so the flash-attn compile doesn't OOM.
_CORES="$(nproc 2>/dev/null || echo 8)"
_RAMGB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"; _RAMGB="${_RAMGB:-8}"
_RAMCAP=$(( _RAMGB / 4 )); [[ "$_RAMCAP" -lt 1 ]] && _RAMCAP=1
export MAX_JOBS="${MAX_JOBS:-$(( _CORES < _RAMCAP ? _CORES : _RAMCAP ))}"
echo "   MAX_JOBS=$MAX_JOBS  (cores=$_CORES, ram=${_RAMGB}GB)"

echo "== [6/8] Python dependencies =="
pip install --no-deps -r requirements.txt
pip install "git+https://github.com/microsoft/MoGe.git"
pip install --no-build-isolation "transformer_engine[pytorch]"
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"

echo "== [7/8] Flash Attention (compiles; takes a while) =="
# flash-attn's compile uses ~8-10GB RAM PER job; too many jobs OOM-kill the build (and
# the tmux session) on a big box. Cap hard (slower but reliable). Override via FA_MAX_JOBS.
MAX_JOBS="${FA_MAX_JOBS:-4}" pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3

echo "== [8/8] Vendored CUDA extensions =="
# DA3 is required for generation (single-image depth), so it's always built.
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'
# vipe (video pose estimation) is only used by the 3D-reconstruction path, not by
# generation -- skip its compile with LYRA2_SKIP_RECON=1 to shorten a one-off setup.
if [[ "${LYRA2_SKIP_RECON:-0}" == "1" ]]; then
  echo "   LYRA2_SKIP_RECON=1 -> skipping vipe build (reconstruct.py will be unavailable)."
else
  USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
fi

cat <<EOF

Done. Lyra-2 is at: $CLONE_DIR/lyra/Lyra-2

Add to your shell profile (~/.bashrc) so LD_LIBRARY_PATH persists:
  SITE=\$CONDA_PREFIX/lib/python3.10/site-packages
  export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$SITE/torch/lib:\$SITE/nvidia/cuda_runtime/lib:\$SITE/nvidia/cudnn/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"

Then:
  conda activate $ENV_NAME
  export LYRA2_HOME=$CLONE_DIR/lyra/Lyra-2
  bash scripts/download_weights.sh
  bash scripts/smoke_test.sh
EOF
