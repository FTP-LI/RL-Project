#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
seed=${1:?Specify a frozen development seed}
case $seed in 1103|1104) ;; *) exit 2 ;; esac
record=records/reports/common-walk-backward-roll-pair
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
entry=scripts/validation/run_walk_backward_roll_pair.py
test -f "$record/windows-entry.json"
test ! -e "$record/seed$seed-sampling-exit.txt"
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Project CUDA lock is busy'; exit 3; }
export PYTHONPATH=scripts/validation:scripts/training MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
printf -v command '%q ' "$python" -B "$entry" --sample "$seed"
printf '%s\n' "$command" > "$record/seed$seed-command.txt"
code=0
script -qefc "$command" "$record/seed$seed-sampling.log" || code=$?
printf '%s\n' "$code" > "$record/seed$seed-sampling-exit.txt"
if (( code != 0 )); then
    exit "$code"
fi
"$python" -B "$entry" --verify "$seed" > "$record/seed$seed-verification.log" 2>&1 || code=$?
printf '%s\n' "$code" > "$record/seed$seed-verification-exit.txt"
cat "$record/seed$seed-verification.log"
exit "$code"
