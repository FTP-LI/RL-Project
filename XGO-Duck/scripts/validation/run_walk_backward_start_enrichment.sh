#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
report=records/reports/common-walk-backward-start-enrichment/recovery
entry=scripts/validation/run_walk_backward_start_enrichment.py
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
test -f "$report/windows-entry.json"
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Project CUDA lock is busy'; exit 3; }
export PYTHONPATH=scripts/validation:scripts/training MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
for arm in control candidate; do
    directory="$report/$arm"
    test ! -e "$directory/sampling-exit.txt"
    printf -v command '%q ' "$python" -B "$entry" --sample "$arm"
    printf '%s\n' "$command" > "$directory/command.txt"
    code=0
    script -qefc "$command" "$directory/sampling.log" || code=$?
    printf '%s\n' "$code" > "$directory/sampling-exit.txt"
    if (( code != 0 )); then
        exit "$code"
    fi
    "$python" -B "$entry" --verify "$arm" > "$directory/verification.log" 2>&1 || code=$?
    printf '%s\n' "$code" > "$directory/verification-exit.txt"
    cat "$directory/verification.log"
    if (( code != 0 )); then
        exit "$code"
    fi
done
