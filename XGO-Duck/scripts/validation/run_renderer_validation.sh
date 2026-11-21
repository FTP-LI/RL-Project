#!/usr/bin/env bash
# Full-mesh EGL/GLFW capability checks, sequential and visible. No training.
# The Python entry holds the project lock and waits for pilot completion.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Shanghai
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
cd "$project_root"
if [[ "${1:-}" == --check-only ]]; then
  exec "$python" -B "$project_root/scripts/validation/validate_renderer.py"
fi
[[ -z "${1:-}" ]] || exit 2
exec "$python" -B "$project_root/scripts/validation/validate_renderer.py" --execute
