#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
record=records/reports/continuous-forward-yaw-replication
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
runner=scripts/validation/run_continuous_forward_yaw_replication.py
test -f "$record/windows-entry.json" && test ! -e "$record/zero"
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Project CUDA lock is busy'; exit 3; }
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

for phase in zero sample verify; do
    printf '%q %q %q %q\n' "$python" -B "$runner" "$phase" > "$record/$phase-command.txt"
    code=0
    "$python" -B "$runner" "$phase" > "$record/$phase-execution.log" 2>&1 || code=$?
    printf '%s\n' "$code" > "$record/$phase-exit.txt"
    (( code == 0 )) || exit "$code"
done
