#!/usr/bin/env bash
# Print the env vars scene-studio needs, derived from your conda envs + checkouts.
# scene-studio dispatches each mode to the right backend's conda-env python, so it
# needs to know each backend's HOME and that env's python interpreter.
#
# Usage:
#   bash scripts/print_env.sh [wan_env] [wan_home] [lyra_env] [lyra_home]
#   eval "$(bash scripts/print_env.sh)"      # to apply directly
set -euo pipefail

WAN_ENV="${1:-wan22}";  WAN_HOME="${2:-$HOME/wan-src/Wan2.2}"
LYRA_ENV="${3:-lyra2}"; LYRA_HOME="${4:-$HOME/lyra-src/lyra/Lyra-2}"

_py() { conda run -n "$1" python -c 'import sys; print(sys.executable)' 2>/dev/null || true; }

echo "# scene-studio environment"
if [[ -d "$WAN_HOME" ]]; then
  echo "export WAN22_HOME=$WAN_HOME"
  P="$(_py "$WAN_ENV")"; [[ -n "$P" ]] && echo "export WAN22_PYTHON=$P"
else
  echo "# (Wan backend not found at $WAN_HOME -- 'animate' will be unavailable)"
fi
if [[ -d "$LYRA_HOME" ]]; then
  echo "export LYRA2_HOME=$LYRA_HOME"
  P="$(_py "$LYRA_ENV")"; [[ -n "$P" ]] && echo "export LYRA2_PYTHON=$P"
else
  echo "# (Lyra backend not found at $LYRA_HOME -- 'travel' will be unavailable)"
fi
echo 'export SCENE_OUTPUT_DIR=$PWD/outputs'
