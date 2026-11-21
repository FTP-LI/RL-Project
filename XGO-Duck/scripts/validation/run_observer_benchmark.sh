#!/usr/bin/env bash
# Six sequential diagnostics in the existing visible shared task terminal.
# Each restores the same parent; no diagnostic weights become a gait candidate.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Shanghai
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
if [[ "${1:-}" == --check-only ]]; then
  exec "$python" -B "$project_root/scripts/training/benchmark_observer.py" --check-only
fi
[[ -z "${1:-}" ]] || exit 2
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
"$python" -B "$project_root/scripts/training/benchmark_observer.py" --check-only
exec "$python" -u -B "$project_root/scripts/training/benchmark_observer.py" --execute
