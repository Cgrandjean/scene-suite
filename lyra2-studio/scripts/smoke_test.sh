#!/usr/bin/env bash
# Quick end-to-end check using a bundled sample image.
# Generates a short DMD (fast, 4-step) walkthrough so you can confirm the
# install + weights work before running full-quality jobs.
#
# Usage:
#   export LYRA2_HOME=/path/to/lyra/Lyra-2
#   bash scripts/smoke_test.sh
set -euo pipefail

: "${LYRA2_HOME:?Set LYRA2_HOME first}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "== Sanity: print the command that would run (no GPU needed) =="
python -m lyra2_studio.generate --dry-run --use-dmd preset \
  --image "$LYRA2_HOME/assets/samples/04.png" \
  --prompt "A cinematic drone shot glides forward through a sun-drenched ancient stone amphitheater." \
  --output-path outputs/smoke

echo
echo "== Running the real thing on GPU (this needs an NVIDIA GPU + downloaded weights) =="
python -m lyra2_studio.generate --use-dmd preset \
  --image "$LYRA2_HOME/assets/samples/04.png" \
  --prompt "A cinematic drone shot glides forward through a sun-drenched ancient stone amphitheater." \
  --output-path outputs/smoke

echo
echo "If it succeeded, look for: $LYRA2_HOME/outputs/smoke/videos/04.mp4"
