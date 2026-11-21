#!/usr/bin/env bash
# Foreground-only evaluation in the shared VS Code terminal. This does not invoke
# train or load an optimizer. Preflight is 2 x 2 s; formal is 32 x 20 s per command.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export TZ=Asia/Shanghai
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$project_root/records/evaluations"
log="$project_root/records/evaluations/console-$(date +%Y%m%dT%H%M%S).log"
cd "$HOME/XGO-Duck/workspace/xgoduck_rl"
printf 'Fixed ONNX evaluation; no training.\nTranscript: %s\n' "$log"
flock -n "$HOME/XGO-Duck/training.lock" .venv/bin/python "$project_root/scripts/validation/evaluate_fixed_policy.py" "$@" 2>&1 | tee "$log"
