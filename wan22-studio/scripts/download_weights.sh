#!/usr/bin/env bash
# Download Wan2.2-I2V-A14B weights from Hugging Face into $WAN22_HOME.
#
# Wan 2.2 is Apache-2.0 -- no license to accept, no gating. A HF token is optional
# (it only helps with download rate limits): export HF_TOKEN=hf_... if you have one.
#
# Usage:
#   export WAN22_HOME=/path/to/Wan2.2
#   bash scripts/download_weights.sh
set -euo pipefail

: "${WAN22_HOME:?Set WAN22_HOME to your Wan2.2 checkout, e.g. export WAN22_HOME=\$HOME/wan-src/Wan2.2}"
CKPT_DIR="${WAN22_CKPT_DIR:-Wan2.2-I2V-A14B}"
MODEL="${WAN22_MODEL:-Wan-AI/Wan2.2-I2V-A14B}"

# pip here may run against an externally-managed system Python (PEP 668) -- this script
# can start before any conda env is active -- so fall back to --break-system-packages.
_pip() { pip install -U "$@" 2>/dev/null || pip install -U "$@" --break-system-packages; }
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Installing huggingface_hub..." >&2
  _pip "huggingface_hub[cli]" || true
fi
# hf_transfer = faster parallel downloads, but ONLY enable it if it actually installs
# (HF_HUB_ENABLE_HF_TRANSFER=1 without the package makes huggingface-cli error out).
if _pip hf_transfer >/dev/null 2>&1; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi

cd "$WAN22_HOME"
echo "Downloading $MODEL into $WAN22_HOME/$CKPT_DIR ..."
echo "  The I2V-A14B checkpoint is large (~70 GB, MoE 27B params). Ensure >=150GB free."
ok=0
for attempt in 1 2 3; do
  if huggingface-cli download "$MODEL" --local-dir "$CKPT_DIR"; then ok=1; break; fi
  echo "  download attempt $attempt failed; retrying in 5s..." >&2
  sleep 5
done
[[ "$ok" == 1 ]] || { echo "ERROR: weight download failed after 3 attempts" >&2; exit 1; }

echo
echo "Done -> $WAN22_HOME/$CKPT_DIR"
ls -1 "$CKPT_DIR" 2>/dev/null | head || echo "  (check the download output above)"
