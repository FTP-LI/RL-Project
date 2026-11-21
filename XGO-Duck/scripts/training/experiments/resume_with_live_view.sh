#!/usr/bin/env bash
# Change: actual env-0 live view and joint/feet diagnostics; no reward or physics edit.
# Why: expose the reported leg-adduction behavior during learning, not old replay.
# Restore the audited checkpoint with Adam, rate, normalizers and curriculum;
# remaining updates end at cumulative 7500. Episodes/RNG restart is explicit.
# Rendering overhead is measured. A visible robot is not proof of stable gait.
set -euo pipefail
export TZ=Asia/Shanghai
export XGODUCK_EXPERIMENT_ENTRY="$(realpath "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
[[ "${1:-}" == '' || "${1:-}" == --check-only ]] || exit 2
if [[ "${1:-}" == --check-only ]]; then
  exec "$python" -B "$project_root/scripts/training/live_resume.py" --check-only
fi
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Another compute task is active; no live resume started.'; exit 3; }
"$python" -B "$project_root/scripts/training/live_resume.py" --check-only
bash "$project_root/scripts/training/run_visible_training.sh" --mode baseline --live-resume
run_id="$(python3 - "$project_root/workspace/console/current-training.json" <<'PY'
import json,re,sys
request=json.load(open(sys.argv[1]))
assert request['phase']=='extension' and request['visualization']['kind']=='native-live-training-state'
run=request['run_id']
assert re.fullmatch(r'baseline_extension_\d{8}T\d{6}_s42',run)
print(run)
PY
)"
flock -u 9
exec 9>&-
exec "$python" -u -B "$project_root/scripts/validation/finish_baseline_extension.py" "$run_id"
