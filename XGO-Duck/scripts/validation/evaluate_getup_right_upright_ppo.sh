#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
record=records/reports/getup-right-upright-ppo/evaluation
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
test -f "$record/preflight.json"
test ! -e "$record/sample-exit.txt"
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Project CUDA lock is busy'; exit 3; }
export PYTHONPATH=scripts/validation:scripts/training MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
printf -v command '%q ' "$python" -B scripts/validation/evaluate_getup_right_upright_ppo.py
printf '%s\n' "$command" > "$record/sample-command.txt"
code=0
script -qefc "$command" "$record/sample-execution.log" || code=$?
printf '%s\n' "$code" > "$record/sample-exit.txt"
if (( code != 0 )); then
    exit "$code"
fi
"$python" -B scripts/validation/evaluate_getup_right_upright_ppo.py --verify > "$record/verify-execution.log" 2>&1 || code=$?
printf '%s\n' "$code" > "$record/verify-exit.txt"
cat "$record/verify-execution.log"
exit "$code"
