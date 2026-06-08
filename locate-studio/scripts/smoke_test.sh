#!/usr/bin/env bash
# Verify the install + weights. With no argument: just load the model (proves the env,
# the gated download and the trust_remote_code path all work). With an image path:
# run a real detection on it.
#
# Usage:
#   bash scripts/smoke_test.sh                 # load-only check
#   bash scripts/smoke_test.sh path/to/img.jpg # full detection check
set -euo pipefail

IMG="${1:-}"
if [[ -n "$IMG" ]]; then
  echo "== Detection smoke test on $IMG =="
  python -m locate_studio.detect --image "$IMG" --classes "person,car,sign,door" --out outputs/smoke
else
  echo "== Load-only smoke test (pass an image path for a full detection) =="
  python - <<'PY'
from locate_studio.worker import LocateAnythingWorker, DEFAULT_MODEL
print(f"loading {DEFAULT_MODEL} ...")
LocateAnythingWorker(DEFAULT_MODEL)
print("OK -- LocateAnything-3B loaded and ready.")
PY
fi
