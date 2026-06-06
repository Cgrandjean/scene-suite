#!/usr/bin/env bash
# One-shot deploy on a fresh A100 Linux box: installs Miniconda if missing,
# downloads the ~97GB weights IN PARALLEL with the (slow, compile-heavy) env
# build, then runs the smoke test. One command, mostly unattended.
#
# Do these BEFORE you start the paid instance (they're free, from your laptop):
#   1. Get this repo onto the box (rsync/scp or git clone) -- see README.
#   2. Accept the Lyra-2.0 license at https://huggingface.co/nvidia/Lyra-2.0
#   3. Have a Hugging Face token ready.
#
# Usage (on the box):
#   export HF_TOKEN=hf_xxx
#   bash scripts/bootstrap.sh [/path/to/clone/dir]   # default: ~/lyra-src
set -euo pipefail

CLONE_DIR="${1:-$HOME/lyra-src}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${LYRA2_ENV:-lyra2}"

echo "== [0/5] Pre-flight =="
[[ "$(uname -s)" == "Linux" ]] || { echo "ERROR: needs a Linux GPU box." >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: no nvidia-smi -- no NVIDIA GPU?" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# A HF token (or a prior `huggingface-cli login`) is required, else the download 403s.
if [[ -z "${HF_TOKEN:-}" ]] && [[ ! -f "$HOME/.cache/huggingface/token" ]]; then
  echo "ERROR: no Hugging Face credentials." >&2
  echo "  Run 'huggingface-cli login' or 'export HF_TOKEN=hf_...', and accept the" >&2
  echo "  license at https://huggingface.co/nvidia/Lyra-2.0" >&2
  exit 1
fi
PARENT="$(dirname "$CLONE_DIR")"; mkdir -p "$PARENT"
AVAIL_GB="$(df -BG "$PARENT" | awk 'NR==2{gsub(/[^0-9]/,"",$4); print $4}')"
echo "   disk free at $PARENT: ${AVAIL_GB:-?}GB (need >=200)"
[[ "${AVAIL_GB:-0}" -lt 200 ]] && echo "   WARNING: <200GB free; weights + env may not fit." >&2

echo "== [1/5] Miniconda =="
if ! command -v conda >/dev/null 2>&1; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "== [2/5] Clone Lyra (so the weight download has a target dir) =="
mkdir -p "$CLONE_DIR"
[[ -d "$CLONE_DIR/lyra/.git" ]] || git clone --recursive https://github.com/nv-tlabs/lyra.git "$CLONE_DIR/lyra"
export LYRA2_HOME="$CLONE_DIR/lyra/Lyra-2"

echo "== [3/5] Weight download in BACKGROUND (overlaps the env build) =="
echo "   progress -> $REPO_ROOT/weights_download.log"
( bash "$REPO_ROOT/scripts/download_weights.sh" ) > "$REPO_ROOT/weights_download.log" 2>&1 &
DL_PID=$!

echo "== [4/5] Build the conda env (compile-heavy; runs while weights download) =="
# One-off generation test: skip the vipe build (video-pose / 3D-recon only; generation
# does not import it). DA3 is still built -- generation needs it. Set LYRA2_SKIP_RECON=0
# if you also want the reconstruct.py path.
export LYRA2_SKIP_RECON="${LYRA2_SKIP_RECON:-1}"
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
# Re-export the runtime library path (setup_host.sh set this in its own process only).
SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash "$REPO_ROOT/scripts/smoke_test.sh"

cat <<EOF

DONE. To generate your own:
  conda activate $ENV_NAME
  export LYRA2_HOME=$LYRA2_HOME
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
  python -m lyra2_studio.generate --use-dmd preset --image my.jpg --prompt "..."
EOF
