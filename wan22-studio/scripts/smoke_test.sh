#!/usr/bin/env bash
# Quick end-to-end image-to-video check on Wan's bundled example image.
# Confirms the install + weights before you run your own jobs.
#
# Usage:
#   export WAN22_HOME=/path/to/Wan2.2
#   bash scripts/smoke_test.sh
set -euo pipefail

: "${WAN22_HOME:?Set WAN22_HOME first}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

IMG="$WAN22_HOME/examples/i2v_input.JPG"
PROMPT="A gentle breeze moves through the scene as it slowly comes to life, cinematic, photorealistic."
mkdir -p "$WAN22_HOME/outputs"

echo "== Sanity: print the command that would run (no GPU needed) =="
python -m wan22_studio.generate --dry-run --image "$IMG" --prompt "$PROMPT" --size "1280*720"

echo
echo "== Real run on GPU (needs the A100 + downloaded weights) =="
python -m wan22_studio.generate --image "$IMG" --prompt "$PROMPT" --size "1280*720" \
  --save-file "$WAN22_HOME/outputs/smoke.mp4"

echo
echo "If it worked, look for: $WAN22_HOME/outputs/smoke.mp4"
