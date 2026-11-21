#!/usr/bin/env bash
# Use the existing Linux environment and the shared project compute lock.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$HOME/XGO-Duck/workspace/xgoduck_rl"
PY="$WORK/.venv/bin/python"
ENTRY="$ROOT/scripts/validation/run_roll_post_getup_training.py"
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Another XGO-Duck compute job is active.'; exit 3; }
source "$ROOT/scripts/setup/project_environment.sh"
export PYTHONPATH="$ROOT/scripts/training:$ROOT/scripts/validation"
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA TZ=Asia/Shanghai
cd "$WORK"
case "${1:?Use qualify, zero, match-zero or train}" in
  qualify)
    "$PY" -B "$ENTRY" freeze
    "$PY" -B "$ENTRY" zero official-standing-reset
    "$PY" -B "$ENTRY" zero post-getup-start-curriculum
    exec "$PY" -B "$ENTRY" match-zero
    ;;
  train)
    if [[ $# -gt 1 ]]; then exec "$PY" -B "$ENTRY" "$@"; fi
    "$PY" -B "$ENTRY" train official-standing-reset
    exec "$PY" -B "$ENTRY" train post-getup-start-curriculum
    ;;
  zero|match-zero) exec "$PY" -B "$ENTRY" "$@" ;;
  *) echo 'Unknown post-get-up roll training action.'; exit 2 ;;
esac
