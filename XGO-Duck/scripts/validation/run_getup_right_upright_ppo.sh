#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
record=records/reports/getup-right-upright-ppo
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
test -f "$record/windows-entry.json"
test ! -e "$record/coordinator.json"
test ! -e "$record/training-exit.txt"
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Project CUDA lock is busy'; exit 3; }
export PYTHONPATH=scripts/validation:scripts/training MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
printf -v command '%q ' "$python" -B scripts/validation/run_getup_right_upright_ppo.py
printf '%s\n' "$command" > "$record/training-command.txt"
# The inherited coordinator expects a PTY. This records it without opening a window.
code=0
script -qefc "$command" "$record/coordinator-transcript.txt" || code=$?
printf '%s\n' "$code" > "$record/training-exit.txt"
exit "$code"
