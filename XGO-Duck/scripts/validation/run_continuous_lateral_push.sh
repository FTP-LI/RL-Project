#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
record=records/reports/continuous-forward-lateral-push
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
test -f "$record/windows-entry.json" && test -f "$record/cuda-zero.json"
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Project CUDA lock is busy'; exit 3; }
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

run() {
    local phase=$1 code=0
    shift
    printf '%q ' "$@" > "$record/$phase-command.txt"
    printf '\n' >> "$record/$phase-command.txt"
    "$@" > "$record/$phase-execution.log" 2>&1 || code=$?
    printf '%s\n' "$code" > "$record/$phase-exit.txt"
    return "$code"
}

run sample "$python" -B scripts/validation/run_continuous_lateral_push.py sample
run verify "$python" -B scripts/validation/verify_continuous_lateral_push.py
