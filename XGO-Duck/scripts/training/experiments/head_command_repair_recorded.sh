#!/usr/bin/env bash
# Same saved control parent for both arms, same 250-update budget. Only the
# declared neck-command ranges differ. Native viewer is OFF in both arms to
# remove its measured lag; retain real 200-frame clips and the visible terminal.
# This creates a new matched experiment, not an exact interrupted trajectory.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl TZ=Asia/Shanghai
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
export GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
cd "$HOME/XGO-Duck/workspace/xgoduck_rl"
if [[ "${1:-}" == --check-only ]]; then
  exec "$python" -B "$project_root/scripts/training/head_command_repair.py" --recorded-recovery --check-only
fi
[[ -z "${1:-}" ]] || exit 2
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
"$python" -B "$project_root/scripts/training/head_command_repair.py" --recorded-recovery --check-only
"$python" -B "$project_root/scripts/training/head_command_repair.py" --recorded-recovery --execute
flock -u 9
exec 9>&-
batch="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["batch"])' "$project_root/workspace/console/current-head-command-repair.json")"
exec "$python" -B "$project_root/scripts/validation/finish_head_command_repair.py" "$batch"
