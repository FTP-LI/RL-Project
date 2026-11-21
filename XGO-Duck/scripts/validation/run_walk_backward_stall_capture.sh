#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
out=records/reports/common-walk-backward-stall-capture/capacity
entry=scripts/validation/run_walk_backward_stall_capture.py
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
test -f "$out/windows-entry.json"
test ! -e "$out/sampling-exit.txt"
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Project CUDA lock is busy'; exit 3; }
export PYTHONPATH=scripts/validation:scripts/training MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
printf -v command '%q ' "$python" -B "$entry" --sample
printf '%s\n' "$command" > "$out/command.txt"
code=0
script -qefc "$command" "$out/sampling.log" || code=$?
printf '%s\n' "$code" > "$out/sampling-exit.txt"
if (( code != 0 )); then
    exit "$code"
fi
CUDA_VISIBLE_DEVICES=-1 "$python" -B "$entry" --verify > "$out/verification.log" 2>&1 || code=$?
printf '%s\n' "$code" > "$out/verification-exit.txt"
cat "$out/verification.log"
exit "$code"
