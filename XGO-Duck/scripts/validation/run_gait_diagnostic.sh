#!/usr/bin/env bash
# Visible fixed-weight diagnosis after stopping the degenerate training run.
# Compare one factor at a time; no training or hardware operation is performed.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl TZ=Asia/Shanghai PYTHONDONTWRITEBYTECODE=1
cd "$HOME/XGO-Duck/workspace/xgoduck_rl"
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Another compute task is active.'; exit 3; }
exec .venv/bin/python -B "$project_root/scripts/validation/run_gait_diagnostic.py" "$@"
