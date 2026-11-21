#!/usr/bin/env bash
# Reproduce the proposed reachable-head configuration check in the shared task
# terminal. This samples commands on CPU fixtures; it does not run a simulator,
# policy, optimizer, native viewer or hardware program.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
[[ -z "${1:-}" ]] || exit 2
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'A project compute task is active; finish its timing first.'; exit 3; }
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
record_dir="$project_root/records/reports/head-command-bounds"
mkdir -p "$record_dir"
printf 'HEAD_COMMAND_CPU_CHECK %s\nCOMMAND: %s -B %s\n' "$(date -Iseconds)" "$python" "$project_root/scripts/validation/check_head_command_bounds.py" | tee -a "$record_dir/console.log"
set +e
"$python" -B "$project_root/scripts/validation/check_head_command_bounds.py" 2>&1 | tee -a "$record_dir/console.log"
status=${PIPESTATUS[0]}
set -e
printf 'HEAD_COMMAND_CHECK_EXIT_CODE=%s\n' "$status" | tee -a "$record_dir/console.log"
exit "$status"
