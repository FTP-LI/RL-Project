#!/usr/bin/env bash
# One visible foreground batch; all six workers inherit the same terminal.
set -euo pipefail
export TZ=Asia/Shanghai
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
[[ "${1:-}" == '' || "${1:-}" == --check-only ]] || exit 2
if [[ "${1:-}" == --check-only ]]; then
  exec "$python" -B "$project_root/scripts/training/benchmark_parallel_envs.py" --check-only
fi
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Another compute task is active; no benchmark started.'; exit 3; }
"$python" -B "$project_root/scripts/training/benchmark_parallel_envs.py" --check-only
exec "$python" -u -B "$project_root/scripts/training/benchmark_parallel_envs.py" --execute
