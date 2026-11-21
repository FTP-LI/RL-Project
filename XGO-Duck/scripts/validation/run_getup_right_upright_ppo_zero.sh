#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
record=records/reports/getup-right-upright-ppo-zero
entry=run_getup_right_upright_ppo_zero.py
case ${1-} in
    readback|probe)
        record+="/$1"
        entry="run_getup_right_upright_ppo_zero_$1.py"
        ;;
    '') ;;
    *) exit 2 ;;
esac
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
test -f "$record/preflight.json"
test ! -e "$record/pair-exit.txt"
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Project CUDA lock is busy'; exit 3; }
export PYTHONPATH=scripts/validation:scripts/training MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
command=("$python" -B "scripts/validation/$entry" pair)
printf '%q ' "${command[@]}" > "$record/pair-command.txt"
printf '\n' >> "$record/pair-command.txt"
code=0
"${command[@]}" > "$record/pair-execution.log" 2>&1 || code=$?
printf '%s\n' "$code" > "$record/pair-exit.txt"
cat "$record/pair-execution.log"
exit "$code"
