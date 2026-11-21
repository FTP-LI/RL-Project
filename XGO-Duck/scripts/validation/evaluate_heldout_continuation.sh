#!/usr/bin/env bash
# Run only after tasks 19/20 finish and development metrics pass. Reuse the visible
# VS Code terminal. The final model is fixed before using the reserved seed 2001;
# commands, simulator, metrics and episode budget remain unchanged. One attempt
# per training family; a CPU verification retry must read the saved evaluation.
set -euo pipefail
mode="${1:-}"
[[ -z "$mode" || "$mode" == --check-only ]] || exit 2
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$HOME/XGO-Duck/workspace/xgoduck_rl"
helper="$project_root/scripts/validation/heldout_evaluation.py"
if [[ "$mode" == --check-only ]]; then
  exec "$work/.venv/bin/python" -B "$helper" check
fi
plan="$("$work/.venv/bin/python" -B "$helper" prepare)"
model="$("$work/.venv/bin/python" -B - "$project_root" "$plan" <<'PY'
import json,sys
from pathlib import Path
root,plan=map(Path,sys.argv[1:])
print(root/json.loads(plan.read_text())['model'])
PY
)"
record="$(dirname "$plan")"
source "$project_root/scripts/setup/project_environment.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl TZ=Asia/Shanghai
log_file="$record/heldout-evaluation-$(date +%Y%m%dT%H%M%S).log"
cp "${BASH_SOURCE[0]}" "$record/heldout-evaluation-entry.sh"
printf 'Held-out final-policy evaluation. Selection: %s\nTranscript: %s\n' "$plan" "$log_file"
cd "$work"
# The claim is made inside the evaluator, after the shared compute lock is held.
flock -n "$HOME/XGO-Duck/training.lock" "$work/.venv/bin/python" -B \
  "$project_root/scripts/validation/evaluate_fixed_policy.py" \
  --onnx "$model" --heldout-plan "$plan" 2>&1 | tee "$log_file"
evaluation="$("$work/.venv/bin/python" -B - "$log_file" <<'PY'
import re,sys
from pathlib import Path
paths=re.findall(r'^EVALUATION completed: (.+)$',Path(sys.argv[1]).read_text(),re.M)
assert len(paths)==1, 'Expected one completed held-out evaluation in this transcript'
print(paths[0])
PY
)"
"$work/.venv/bin/python" -B "$project_root/scripts/validation/verify_fixed_evaluation.py" "$evaluation" 2>&1 | tee -a "$log_file"
"$work/.venv/bin/python" -B "$helper" finish "$evaluation" 2>&1 | tee -a "$log_file"
printf 'HELDOUT_EVALUATION_EXIT_CODE=0\n' | tee -a "$log_file"
