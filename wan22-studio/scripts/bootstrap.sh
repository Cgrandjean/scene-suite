#!/usr/bin/env bash
# One-shot deploy of Wan 2.2 image-to-video on a fresh A100 Linux box: installs
# Miniconda if missing, clones Wan2.2, downloads the weights IN PARALLEL with the
# env build, then runs the smoke test. One command, mostly unattended.
#
# Usage (on the box):
#   bash scripts/bootstrap.sh [/path/to/clone/dir]   # default: ~/wan-src
# A HF token is optional: export HF_TOKEN=hf_... (only helps download speed limits).
set -euo pipefail

CLONE_DIR="${1:-$HOME/wan-src}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${WAN22_ENV:-wan22}"

echo "== [0/5] Pre-flight =="
[[ "$(uname -s)" == "Linux" ]] || { echo "ERROR: needs a Linux GPU box." >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: no nvidia-smi -- no NVIDIA GPU?" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
PARENT="$(dirname "$CLONE_DIR")"; mkdir -p "$PARENT"
AVAIL_GB="$(df -BG "$PARENT" | awk 'NR==2{gsub(/[^0-9]/,"",$4); print $4}')"
echo "   disk free at $PARENT: ${AVAIL_GB:-?}GB (need >=150)"
[[ "${AVAIL_GB:-0}" -lt 150 ]] && echo "   WARNING: <150GB free; weights + env may not fit." >&2

echo "== [1/5] Miniconda =="
if ! command -v conda >/dev/null 2>&1; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "== [2/5] Clone Wan2.2 (so the weight download has a target dir) =="
mkdir -p "$CLONE_DIR"
[[ -d "$CLONE_DIR/Wan2.2/.git" ]] || git clone https://github.com/Wan-Video/Wan2.2.git "$CLONE_DIR/Wan2.2"
export WAN22_HOME="$CLONE_DIR/Wan2.2"

echo "== [3/5] Weight download in BACKGROUND (overlaps the env build) =="
echo "   progress -> $REPO_ROOT/weights_download.log"
( bash "$REPO_ROOT/scripts/download_weights.sh" ) > "$REPO_ROOT/weights_download.log" 2>&1 &
DL_PID=$!

echo "== [4/5] Build the conda env =="
bash "$REPO_ROOT/scripts/setup_host.sh" "$CLONE_DIR"

echo "== Waiting for the weight download to finish... =="
if wait "$DL_PID"; then
  echo "   weights OK"
else
  echo "ERROR: weight download failed -- see $REPO_ROOT/weights_download.log" >&2
  exit 1
fi

echo "== [5/5] Smoke test =="
conda activate "$ENV_NAME"
bash "$REPO_ROOT/scripts/smoke_test.sh"

cat <<EOF

DONE. To make your own:
  conda activate $ENV_NAME
  export WAN22_HOME=$WAN22_HOME
  python -m wan22_studio.generate --image my_maps_frame.jpg \\
    --prompt "the scene slowly comes to life: ..." --size "1280*720"
EOF
