#!/usr/bin/env bash
# Pre-fetch the LocateAnything-3B weights + custom code into the Hugging Face cache.
#
# The model is GATED and released under NVIDIA's non-commercial license -- accept it on
# the model page and have an HF token ready:
#   https://huggingface.co/nvidia/LocateAnything-3B
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash scripts/download_weights.sh
set -euo pipefail

MODEL="${LOCATE_MODEL:-nvidia/LocateAnything-3B}"

# pip here may hit an externally-managed system Python (PEP 668) -- fall back gracefully.
_pip() { pip install -U "$@" 2>/dev/null || pip install -U "$@" --break-system-packages; }
# transformers 4.57.1 needs huggingface_hub <1.0, so keep the CLI on that line too.
command -v huggingface-cli >/dev/null 2>&1 || _pip "huggingface_hub<1.0" || true
# hf_transfer = faster parallel downloads, only if it actually installs.
if _pip hf_transfer >/dev/null 2>&1; then export HF_HUB_ENABLE_HF_TRANSFER=1; fi

echo "Downloading $MODEL (~6-7 GB: bf16 weights + remote-code .py) into the HF cache ..."
ok=0
for attempt in 1 2 3; do
  if huggingface-cli download "$MODEL"; then ok=1; break; fi
  echo "  attempt $attempt failed; retrying in 5s..." >&2; sleep 5
done
[[ "$ok" == 1 ]] || {
  echo "ERROR: download failed. nvidia/LocateAnything-3B is GATED -- accept its license on" >&2
  echo "  the HF model page AND export HF_TOKEN=hf_..." >&2
  exit 1
}
echo "Done. Cached -- load it in code with the id '$MODEL' (no path needed)."
