#!/usr/bin/env bash
# One-shot deploy of LocateAnything-3B on a fresh Linux GPU box: installs Miniconda if
# missing, builds the env, downloads the gated weights, then loads the model to verify.
#
# Do this BEFORE starting (free, from your laptop):
#   1. Accept the license at https://huggingface.co/nvidia/LocateAnything-3B
#   2. Have a Hugging Face token ready.
#
# Usage (on the box):
#   export HF_TOKEN=hf_...
#   bash scripts/bootstrap.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${LOCATE_ENV:-locate}"

echo "== [0/4] Pre-flight =="
[[ "$(uname -s)" == "Linux" ]] || { echo "ERROR: needs a Linux GPU box." >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: no nvidia-smi -- no NVIDIA GPU?" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if [[ -z "${HF_TOKEN:-}" ]] && [[ ! -f "$HOME/.cache/huggingface/token" ]]; then
  echo "ERROR: no Hugging Face credentials." >&2
  echo "  export HF_TOKEN=hf_... and accept the license at" >&2
  echo "  https://huggingface.co/nvidia/LocateAnything-3B" >&2
  exit 1
fi

echo "== [1/4] Miniconda =="
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
else
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
fi

echo "== [2/4] Build the conda env =="
bash "$REPO_ROOT/scripts/setup_host.sh"

echo "== [3/4] Download weights (in-env) =="
set +u; conda activate "$ENV_NAME"; set -u
bash "$REPO_ROOT/scripts/download_weights.sh"

echo "== [4/4] Smoke test (load the model) =="
bash "$REPO_ROOT/scripts/smoke_test.sh"

cat <<EOF

DONE. To detect:
  conda activate $ENV_NAME
  python -m locate_studio.detect --image my.jpg  --classes "person,car,dog"
  python -m locate_studio.detect --video clip.mp4 --classes "person" --stride 5
  python -m locate_studio.detect --image room.jpg --query "the lamp on the left"
EOF
