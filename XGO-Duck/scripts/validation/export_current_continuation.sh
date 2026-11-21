#!/usr/bin/env bash
# Run after the visible continuation task has finished, in the same terminal.
# Select the registered final checkpoint, audit its total budget on CPU, then
# export and compare 1000 simulator observations. No training or robot I/O.
# --check-only reads current input records; it does not run the checkpoint audit,
# export, simulation, or install anything. Incomplete training is rejected.
set -euo pipefail
export TZ=Asia/Shanghai
mode="${1:-}"
[[ -z "$mode" || "$mode" == --check-only ]] || exit 2
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run_id="$(python3 - "$project_root/project.json" <<'PY'
import json, re, sys
p=json.load(open(sys.argv[1], encoding='utf-8-sig'))
run=p['formal_training']['continuation_run_id']
assert re.fullmatch(r'baseline_continuation_\d{8}T\d{6}_s(?:42|43|44)',run)
print(run)
PY
)"
record="$project_root/records/experiments/$run_id"
if [[ ! -f "$record/exit-code.txt" ]] || [[ "$(cat "$record/exit-code.txt")" != 0 ]]; then
  echo 'EXPORT_NOT_STARTED: Training has not completed successfully. Keep the existing training task running.'
  exit 2
fi
checkpoint="$(python3 - "$record" <<'PY'
import json, sys
from pathlib import Path
r=Path(sys.argv[1])
result=json.loads((r/'training-result.json').read_text())
plan=json.loads((r/'resume-plan.json').read_text())
assert result['status']=='training-finished' and result['exit_code']==0
print(Path(result['log_dir'])/f"model_{plan['final_iteration_label']}.pt")
PY
)"
if [[ "$mode" == --check-only ]]; then
  exec bash "$project_root/scripts/validation/export_and_validate.sh" "$run_id" "$checkpoint" --check-only
fi
"$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" -B \
  "$project_root/scripts/validation/verify_continuation_checkpoint.py" "$run_id" "$checkpoint" --final
exec bash "$project_root/scripts/validation/export_and_validate.sh" "$run_id" "$checkpoint"
