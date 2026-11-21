#!/usr/bin/env bash
# Continue only the formal 5000-update checkpoint, never a scaling-test model.
# Changes: measured parallelism and a bounded 2500-update learning stage.
# Why: improve sampling efficiency and evaluate learning again at 7500 updates.
# Restore Adam/rate/curriculum/normalizers, reset fresh environments; keep the
# official reward and fixed gait thresholds. More throughput is not better gait.
# Run visibly in the existing shared VS Code terminal. Preserve all old records.
set -euo pipefail
export TZ=Asia/Shanghai
export XGODUCK_EXPERIMENT_ENTRY="$(realpath "${BASH_SOURCE[0]}")"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
python="$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python"
[[ "${1:-}" == '' || "${1:-}" == --check-only ]] || exit 2
if [[ "${1:-}" == --check-only ]]; then
  exec "$python" -B "$project_root/scripts/training/extend_official.py" --check-only
fi
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Another compute task is active; no extension started.'; exit 3; }
"$python" -B "$project_root/scripts/training/extend_official.py" --check-only
bash "$project_root/scripts/training/run_visible_training.sh" --mode baseline --extend
# Reuse the same visible task for final checks and fixed development evaluation.
# Each simulation/export helper acquires its own compute lock after training.
run_id="$(python3 - "$project_root/workspace/console/current-training.json" <<'PY'
import json,re,sys
request=json.load(open(sys.argv[1]))
assert request['phase']=='extension'
run=request['run_id']
assert re.fullmatch(r'baseline_extension_\d{8}T\d{6}_s42',run)
print(run)
PY
)"
flock -u 9
exec 9>&-
exec "$python" -u -B "$project_root/scripts/validation/finish_baseline_extension.py" "$run_id"
