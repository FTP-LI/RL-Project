#!/usr/bin/env bash
# Visible, sequential diagnostic pair. No install, formal-model replacement,
# gait selection, or host-speedup claim. The ordinary run measures profiler cost.
set -euo pipefail
export TZ=Asia/Shanghai
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
work="$HOME/XGO-Duck/workspace/xgoduck_rl"
python="$work/.venv/bin/python"
mode="${1:-}"
[[ -z "$mode" || "$mode" == --check-only ]] || exit 2
if [[ "$mode" == --check-only ]]; then
  exec "$python" -B "$project_root/scripts/training/profile_components.py" --check-only
fi
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Compute task already active; profiling not started.'; exit 3; }
"$python" -B "$project_root/scripts/training/profile_components.py" --check-only
python3 - "$project_root/project.json" <<'PY'
import json, sys
project=json.load(open(sys.argv[1],encoding='utf-8-sig'))
assert not project['formal_training']['status'].startswith('paused'), 'Training paused by user'
PY
pair="$project_root/records/profiling/profile_$(date +%Y%m%dT%H%M%S)_s42"
mkdir -p "$project_root/records/profiling"
mkdir "$pair"
cp "${BASH_SOURCE[0]}" "$pair/entry.sh"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=egl
cd "$work"
for trial in reference traced; do
  mkdir "$pair/$trial"
  set +e
  "$python" -u -B "$project_root/scripts/training/profile_components.py" --execute \
    --mode "$trial" --record-dir "$pair/$trial" 2>&1 | tee "$pair/$trial/console.log"
  result=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$result" > "$pair/$trial/exit-code.txt"
  printf 'PROFILE_MODE=%s EXIT_CODE=%s\n' "$trial" "$result"
  [[ "$result" == 0 ]] || exit "$result"
done
"$python" -B "$project_root/scripts/validation/verify_profile_pair.py" "$pair" 2>&1 | tee "$pair/verification-console.log"
echo "PROFILE_PAIR_COMPLETED: $pair"
