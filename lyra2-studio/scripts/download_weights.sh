#!/usr/bin/env bash
# Download Lyra 2.0 model weights from Hugging Face into $LYRA2_HOME/checkpoints.
#
# The weights are released under NVIDIA's Internal Scientific Research and
# Development Model License -- research/non-commercial use only. By downloading
# you accept that license. You may need to be logged in: `huggingface-cli login`.
#
# Usage:
#   export LYRA2_HOME=/path/to/lyra/Lyra-2
#   bash scripts/download_weights.sh
set -euo pipefail

: "${LYRA2_HOME:?Set LYRA2_HOME to your Lyra-2 checkout, e.g. export LYRA2_HOME=\$HOME/lyra-src/lyra/Lyra-2}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Installing huggingface_hub..." >&2
  pip install -U "huggingface_hub[cli]"
fi
# hf_transfer = parallel chunked downloads; saturates the box's bandwidth on the 97GB pull.
pip install -U hf_transfer >/dev/null 2>&1 || true
export HF_HUB_ENABLE_HF_TRANSFER=1

cd "$LYRA2_HOME"
echo "Downloading nvidia/Lyra-2.0 checkpoints into $LYRA2_HOME/checkpoints ..."
echo "  ~97 GB total (model ~68GB, text_encoder ~11GB, recon ~13GB, encoders/vae/lora ~4GB)."
echo "  Ensure you have >=200GB free here. DA3 (~6.8GB) + MoGe (~1GB) download later at runtime."
huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .

echo
echo "Downloaded. Top-level checkpoint contents:"
ls -1 checkpoints 2>/dev/null || echo "  (no checkpoints/ dir found -- check the download output above)"
