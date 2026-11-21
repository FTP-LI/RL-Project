#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
report=records/reports/common-walk-backward-load-clock
entry=scripts/validation/run_walk_backward_load_clock.py
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
test -f "$report/windows-entry.json"
test ! -e "$report/sampling-exit.txt"
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Project CUDA lock is busy'; exit 3; }
export PYTHONPATH=scripts/validation:scripts/training MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
printf -v command '%q ' "$python" -B "$entry" --sample
printf '%s\n' "$command" > "$report/command.txt"
code=0
script -qefc "$command" "$report/sampling.log" || code=$?
printf '%s\n' "$code" > "$report/sampling-exit.txt"
if (( code != 0 )); then
    exit "$code"
fi
"$python" -B "$entry" --verify > "$report/verification.log" 2>&1 || code=$?
printf '%s\n' "$code" > "$report/verification-exit.txt"
cat "$report/verification.log"
exit "$code"
