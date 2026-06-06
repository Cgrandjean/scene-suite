#!/usr/bin/env bash
# Set up a Wan 2.2 inference environment on an NVIDIA GPU host.
#
# Mirrors the official Wan2.2 README. Requires: Linux, an NVIDIA GPU, and conda.
# Much lighter than typical research stacks -- Wan is Apache-2.0 with no exotic
# from-source CUDA extensions beyond flash_attn.
#
# Usage:
#   bash scripts/setup_host.sh [/path/to/clone/dir]   # default: ~/wan-src
# After it finishes:
#   export WAN22_HOME=<clone_dir>/Wan2.2
set -euo pipefail

CLONE_DIR="${1:-$HOME/wan-src}"
ENV_NAME="${WAN22_ENV:-wan22}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: Wan 2.2 inference requires Linux + an NVIDIA GPU. Detected: $(uname -s)." >&2
  exit 1
fi
command -v nvidia-smi >/dev/null 2>&1 || echo "WARNING: nvidia-smi not found -- no NVIDIA GPU?" >&2
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda first (or use scripts/bootstrap.sh)." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Newer conda requires accepting channel Terms of Service before creating envs.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "== [1/4] Clone Wan2.2 into $CLONE_DIR =="
mkdir -p "$CLONE_DIR"
if [[ ! -d "$CLONE_DIR/Wan2.2/.git" ]]; then
  git clone https://github.com/Wan-Video/Wan2.2.git "$CLONE_DIR/Wan2.2"
else
  echo "  already cloned"
fi
cd "$CLONE_DIR/Wan2.2"

echo "== [2/4] Create conda env '$ENV_NAME' (python 3.10) =="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "  env '$ENV_NAME' already exists; reusing it"
else
  conda create -n "$ENV_NAME" python=3.10 pip ffmpeg -c conda-forge -y
fi
conda activate "$ENV_NAME"

echo "== [3/4] PyTorch (>=2.4, CUDA 12.4) =="
pip install "torch>=2.4.0" torchvision --index-url https://download.pytorch.org/whl/cu124

echo "== [4/4] Wan 2.2 requirements =="
# Install everything EXCEPT flash_attn. flash_attn is OPTIONAL: Wan's attention module
# falls back to torch's scaled_dot_product_attention when it's missing (just a warning).
grep -viE 'flash[-_]attn' requirements.txt > /tmp/wan_req.txt
pip install -r /tmp/wan_req.txt
# Deps Wan's code imports but its requirements.txt omits.
pip install einops decord librosa peft

echo "== [4b/4] flash_attn (REQUIRED by Wan i2v; compiled from source) =="
# Wan's model calls flash_attention() directly (assert FLASH_ATTN_2_AVAILABLE), so it
# MUST be installed -- there is no SDPA fallback on that path. Needs nvcc + CUDA_HOME;
# a CUDA 'devel' base image provides nvcc. Locate it.
if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda
  else
    CUDA_HOME="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)"
  fi
fi
if [[ "${WAN22_SKIP_FLASH:-0}" == "1" ]]; then
  echo "   WAN22_SKIP_FLASH=1 -> skipping (NOTE: Wan i2v generation will then FAIL)."
elif [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
  echo "   building flash_attn with CUDA_HOME=$CUDA_HOME (10-20 min)..."
  MAX_JOBS="${MAX_JOBS:-8}" pip install flash_attn --no-build-isolation \
    && echo "   flash_attn installed." \
    || echo "   WARNING: flash_attn build FAILED -- Wan i2v will not run until it's installed."
else
  echo "   WARNING: nvcc not found -- cannot build flash_attn; use a CUDA 'devel' image."
fi

cat <<EOF

Done. Wan2.2 is at: $CLONE_DIR/Wan2.2
Then:
  conda activate $ENV_NAME
  export WAN22_HOME=$CLONE_DIR/Wan2.2
  bash scripts/download_weights.sh
  bash scripts/smoke_test.sh
EOF
