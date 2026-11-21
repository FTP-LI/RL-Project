#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export TZ=Asia/Shanghai
cd "$HOME/XGO-Duck/workspace/xgoduck_rl"
uv run --locked --offline python "$project_root/scripts/training/visible_training.py" "$@"
