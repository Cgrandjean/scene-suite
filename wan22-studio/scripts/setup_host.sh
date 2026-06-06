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

echo "== [4b/4] flash_attn  (wheel -> conda-nvcc compile -> SDPA fallback) =="
# Wan's model.py calls flash_attention() directly. Get it working in 3 escalating ways,
# so a fresh deploy ends with real flash_attn whenever possible.
_fa_ok=0
if [[ "${WAN22_SKIP_FLASH:-0}" != "1" ]]; then
  _imp() { python -c "import flash_attn" 2>/dev/null; }
  FA_VER=2.8.3
  TVER="$(python -c 'import torch;print(".".join(torch.__version__.split("+")[0].split(".")[:2]))')"
  PYTAG="$(python -c 'import sys;print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
  ABI="$(python -c 'import torch;print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')"
  OTHER="$([[ "$ABI" == "TRUE" ]] && echo FALSE || echo TRUE)"
  _try_whl() { pip install --force-reinstall "https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/flash_attn-${FA_VER}+cu12torch${TVER}cxx11abi$1-${PYTAG}-${PYTAG}-linux_x86_64.whl" >/dev/null 2>&1 && _imp; }
  # (a) fast path: prebuilt wheel (try both ABIs, test the import)
  echo "   [a] prebuilt wheel (torch ${TVER}, ${PYTAG})..."
  if _try_whl "$ABI" || _try_whl "$OTHER"; then _fa_ok=1; echo "   flash_attn OK (wheel)."; fi
  # (b) reliable path: install nvcc via conda (matches torch's CUDA) and compile -- this
  #     avoids the wheel ABI mismatch because it builds against the torch in THIS env.
  if [[ "$_fa_ok" != 1 ]]; then
    echo "   [b] no usable wheel -> installing CUDA toolkit via conda + compiling (~15-25 min)..."
    CUVER="$(python -c 'import torch;print(torch.version.cuda or "12.4")')"
    conda install -y -c nvidia "cuda-toolkit=${CUVER}" >/dev/null 2>&1 \
      || conda install -y -c nvidia cuda-nvcc cuda-cudart-dev >/dev/null 2>&1 || true
    export CUDA_HOME="$CONDA_PREFIX"; export PATH="$CUDA_HOME/bin:$PATH"
    if command -v nvcc >/dev/null 2>&1; then
      pip uninstall -y flash-attn >/dev/null 2>&1 || true
      if MAX_JOBS="${MAX_JOBS:-8}" pip install flash_attn --no-build-isolation && _imp; then
        _fa_ok=1; echo "   flash_attn OK (compiled from source)."
      fi
    else
      echo "   nvcc still unavailable after conda install."
    fi
  fi
fi
# (c) last resort: route Wan through PyTorch SDPA -- always works, ~1.5x slower.
if [[ "$_fa_ok" != 1 ]]; then
  echo "   flash_attn unavailable -> patching Wan to use the PyTorch SDPA fallback."
  pip uninstall -y flash-attn >/dev/null 2>&1 || true
  sed -i 's/from \.attention import flash_attention/from .attention import attention as flash_attention/' wan/modules/model.py
fi

cat <<EOF

Done. Wan2.2 is at: $CLONE_DIR/Wan2.2
Then:
  conda activate $ENV_NAME
  export WAN22_HOME=$CLONE_DIR/Wan2.2
  bash scripts/download_weights.sh
  bash scripts/smoke_test.sh
EOF
