#!/usr/bin/env bash
# Foreground simulation only. The whole batch holds the shared compute lock.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl TZ=Asia/Shanghai PYTHONDONTWRITEBYTECODE=1
mkdir -p "$project_root/records/evaluations"
log="$project_root/records/evaluations/diagnostic-console-$(date +%Y%m%dT%H%M%S).log"
cd "$HOME/XGO-Duck/workspace/xgoduck_rl"
printf 'Single-factor ONNX diagnostics; no training.\nTranscript: %s\n' "$log"
flock -n "$HOME/XGO-Duck/training.lock" .venv/bin/python -B "$project_root/scripts/validation/run_adaptation_diagnostics.py" "$@" 2>&1 | tee "$log"
