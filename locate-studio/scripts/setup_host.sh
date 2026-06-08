#!/usr/bin/env bash
# Set up a LocateAnything-3B inference environment on a Linux GPU host.
#
# Much lighter than Lyra/Wan: NO from-source CUDA extensions to compile -- just a
# recent PyTorch + transformers + a few wheels. ~6-8 GB VRAM at bf16.
#
# Usage:
#   bash scripts/setup_host.sh
set -euo pipefail

ENV_NAME="${LOCATE_ENV:-locate}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: LocateAnything inference needs Linux + an NVIDIA GPU. Detected: $(uname -s)." >&2
  exit 1
fi
command -v nvidia-smi >/dev/null 2>&1 || echo "WARNING: nvidia-smi not found -- no NVIDIA GPU?" >&2
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda first (or use scripts/bootstrap.sh)." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "== [1/3] Create conda env '$ENV_NAME' (python 3.10) =="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "  env '$ENV_NAME' already exists; reusing it"
else
  conda create -n "$ENV_NAME" python=3.10 pip ffmpeg -c conda-forge -y
fi
set +u; conda activate "$ENV_NAME"; set -u   # conda activate scripts may use unbound vars

echo "== [2/3] PyTorch 2.7.1 (cu128) =="
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128

echo "== [3/3] LocateAnything deps + the locate_studio package =="
# Versions from the model card; + accelerate/einops/sentencepiece/hf_hub that the custom
# remote code and the HF loader need. numpy pinned <2 for decord 0.6 ABI compatibility.
# NB: transformers 4.57.1 requires huggingface_hub <1.0 -- pin it, or the import fails
# with "huggingface-hub>=0.30,<1.0 is required ... found 1.x". (huggingface-cli still works.)
pip install transformers==4.57.1 "huggingface_hub<1.0" "numpy<2" Pillow==11.1.0 \
            opencv-python-headless==4.11.0.86 peft decord==0.6.0 lmdb==1.7.5 \
            accelerate einops sentencepiece imageio imageio-ffmpeg
pip install -e "$REPO_ROOT"

cat <<EOF

Done. LocateAnything env '$ENV_NAME' is ready.
  conda activate $ENV_NAME
  export HF_TOKEN=hf_...                 # the model is gated (non-commercial)
  bash $REPO_ROOT/scripts/download_weights.sh
  python -m locate_studio.detect --image my.jpg --classes "person,car"
EOF
