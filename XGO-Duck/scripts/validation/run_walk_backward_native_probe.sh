#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
export PYTHONPATH=scripts/validation:scripts/training MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Project CUDA lock is busy'; exit 3; }
entry=scripts/validation/run_walk_backward_native_probe.py
plan=configs/experiments/common_walk_backward_native_probe.json
record=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["record"])' "$plan")
test ! -e "$record/execution.log"
printf -v command '%q ' "$python" -B "$entry" --sample
printf '%s\n' "$command" > "$record/command.txt"
code=0
script -qefc "$command" "$record/execution.log" || code=$?
printf '%s\n' "$code" > "$record/execution-exit.txt"
exit "$code"
