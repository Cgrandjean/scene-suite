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

# pip here may hit an externally-managed system Python (PEP 668) -- fall back to --break-system-packages.
_pip() { pip install -U "$@" 2>/dev/null || pip install -U "$@" --break-system-packages; }
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Installing huggingface_hub..." >&2
  _pip "huggingface_hub[cli]" || true
fi
# hf_transfer = faster parallel downloads, but only enable it if it actually installs.
if _pip hf_transfer >/dev/null 2>&1; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi

cd "$LYRA2_HOME"
echo "Downloading nvidia/Lyra-2.0 checkpoints into $LYRA2_HOME/checkpoints ..."
echo "  ~97 GB total (model ~68GB, text_encoder ~11GB, recon ~13GB, encoders/vae/lora ~4GB)."
echo "  Ensure you have >=200GB free here. DA3 (~6.8GB) + MoGe (~1GB) download later at runtime."
ok=0
for attempt in 1 2 3; do
  if huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .; then ok=1; break; fi
  echo "  download attempt $attempt failed; retrying in 5s..." >&2; sleep 5
done
[[ "$ok" == 1 ]] || { echo "ERROR: Lyra download failed. nvidia/Lyra-2.0 is GATED -- accept its license on the HF model page AND export HF_TOKEN=hf_..." >&2; exit 1; }

echo
echo "Downloaded. Top-level checkpoint contents:"
ls -1 checkpoints 2>/dev/null || echo "  (no checkpoints/ dir found -- check the download output above)"
