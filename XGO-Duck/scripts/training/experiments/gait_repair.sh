#!/usr/bin/env bash
# Separate matched repair pilot: one added target-excess reward, 250 updates/arm.
# Preserve original baseline, native live observer, recorded clips and shared
# visible terminal. A pilot result is not stable-gait or hardware acceptance.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl TZ=Asia/Shanghai
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
cd "$HOME/XGO-Duck/workspace/xgoduck_rl"
if [[ "${1:-}" == --check-only ]]; then
  exec "$python" -B "$project_root/scripts/validation/check_gait_repair.py"
fi
[[ -z "${1:-}" ]] || exit 2
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
"$python" -B "$project_root/scripts/validation/check_gait_repair.py"
"$python" -B "$project_root/scripts/training/gait_repair.py" --execute
flock -u 9
exec 9>&-
batch="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["batch"])' "$project_root/workspace/console/current-gait-repair.json")"
exec "$python" -B "$project_root/scripts/validation/finish_gait_repair.py" "$batch"
