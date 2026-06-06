#!/usr/bin/env bash
# Turnkey deploy on a fresh A100 pod -- so you can "just launch a GPU and try".
# Sets up the backend(s), installs the scene-studio orchestrator, wires everything,
# and confirms it works. One command.
#
# Expects the repos as SIBLINGS on the pod (rsync them together):
#   ~/scene-studio  ~/wan22-studio  [~/lyra2-studio]
#
# Usage (on the pod):
#   bash scene-studio/scripts/deploy.sh           # Wan only  -> `animate` (fast,  ~70 GB)
#   bash scene-studio/scripts/deploy.sh --all     # + Lyra     -> `travel`/`chain` (~+97 GB, slower)
set -euo pipefail

WITH_LYRA=0
[[ "${1:-}" == "--all" || "${SCENE_WITH_LYRA:-0}" == "1" ]] && WITH_LYRA=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(dirname "$HERE")"
WAN_REPO="$PARENT/wan22-studio"
LYRA_REPO="$PARENT/lyra2-studio"

echo "############################################################"
echo "  scene-studio turnkey deploy   (lyra: $([[ $WITH_LYRA == 1 ]] && echo yes || echo no))"
echo "############################################################"
[[ "$(uname -s)" == "Linux" ]] || { echo "ERROR: run this on the Linux GPU pod." >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: no nvidia-smi / no GPU." >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[[ -d "$WAN_REPO" ]] || { echo "ERROR: $WAN_REPO not found -- rsync all repos as siblings." >&2; exit 1; }
if [[ $WITH_LYRA == 1 && ! -d "$LYRA_REPO" ]]; then
  echo "ERROR: --all needs $LYRA_REPO (rsync it too)." >&2; exit 1
fi

echo; echo "######## [1/4] Wan backend (mode: animate) ########"
bash "$WAN_REPO/scripts/bootstrap.sh"

if [[ $WITH_LYRA == 1 ]]; then
  echo; echo "######## [2/4] Lyra backend (modes: travel, chain) ########"
  bash "$LYRA_REPO/scripts/bootstrap.sh"
else
  echo; echo "######## [2/4] Lyra skipped (re-run with --all for travel/chain) ########"
fi

echo; echo "######## [3/4] scene-studio orchestrator ########"
# conda was installed by the backend bootstrap; make it available in this shell too.
if ! command -v conda >/dev/null 2>&1; then
  [[ -x "$HOME/miniconda3/bin/conda" ]] && eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate wan22; set -u          # reuse the Wan env (already has numpy + ffmpeg)
pip install -e "$HERE"

# Write the env you source on each login (activate env + export backend paths).
ENVFILE="$HOME/scene_env.sh"
{
  echo "source \"\$(conda info --base)/etc/profile.d/conda.sh\""
  echo "conda activate wan22"
  bash "$HERE/scripts/print_env.sh"
} > "$ENVFILE"
echo "wrote $ENVFILE"
set +u; source "$ENVFILE"; set -u

echo; echo "######## [4/4] verify scene-studio wiring (dry-run) ########"
( cd "$HERE" && python -m scene_studio animate \
    --image "$WAN22_HOME/examples/i2v_input.JPG" --prompt "test" --dry-run )

cat <<EOF

############################################################
  DONE.
############################################################
Real test clip (from the Wan smoke test): $WAN22_HOME/outputs/smoke.mp4

On each new login:   source ~/scene_env.sh
Generate (animate):  scene-studio animate --image my.jpg --prompt "..." --save-file out.mp4
EOF
if [[ $WITH_LYRA == 1 ]]; then
cat <<EOF
Travel (Lyra):       scene-studio travel --image my.jpg --motion dolly --use-dmd
Chain (travel->live):scene-studio chain  --image my.jpg --spec $HERE/examples/chain_travel_then_animate.json --out out.mp4
EOF
fi
