#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
record=records/reports/continuous-forward-yaw-pair
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
runner=scripts/validation/run_continuous_forward_yaw_pair.py
verifier=scripts/validation/verify_continuous_forward_yaw_pair.py
test -f "$record/windows-entry.json" && test ! -e "$record/zero"
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Project CUDA lock is busy'; exit 3; }
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

run() {
    local phase=$1
    shift
    printf '%q ' "$@" > "$record/$phase-command.txt"
    printf '\n' >> "$record/$phase-command.txt"
    local code=0
    "$@" > "$record/$phase-execution.log" 2>&1 || code=$?
    printf '%s\n' "$code" > "$record/$phase-exit.txt"
    return "$code"
}

run zero "$python" -B "$runner" zero
run sample "$python" -B "$runner" sample
run verification "$python" -B "$verifier"
