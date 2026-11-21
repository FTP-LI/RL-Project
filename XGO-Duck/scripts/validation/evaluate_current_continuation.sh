#!/usr/bin/env bash
# Evaluate the completed baseline's exported final policy, after task 19.
# Uses the frozen development conditions, records real videos, then independently
# recomputes metrics. This performs no training and does not enable a robot.
# A successful command means evidence collection/verification completed; gait
# passes only if every condition meets the fixed criteria. No held-out seed here.
set -euo pipefail
mode="${1:-}"
[[ -z "$mode" || "$mode" == --check-only ]] || exit 2
export TZ=Asia/Shanghai
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work="$HOME/XGO-Duck/workspace/xgoduck_rl"
run_id="$(python3 - "$project_root" "$work" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
root,work=map(Path,sys.argv[1:])
p=json.loads((root/'project.json').read_text(encoding='utf-8-sig'))
run=p['formal_training']['continuation_run_id']
assert re.fullmatch(r'baseline_continuation_\d{8}T\d{6}_s(?:42|43|44)',run)
record=root/'records/experiments'/run
for name in ('exit-code.txt','export-exit-code.txt'):
    if not (record/name).is_file() or (record/name).read_text().strip()!='0':
        raise SystemExit('EVALUATION_NOT_STARTED: Completed training and successful export are required.')
manifest=json.loads((root/'artifacts'/run/'manifest.json').read_text())
assert manifest['run_id']==run and manifest['status']=='continuation-export-validated'
model=root/'artifacts'/run/'policy.onnx'
assert hashlib.sha256(model.read_bytes()).hexdigest()==manifest['onnx_sha256']
request=json.loads((record/'training-request.json').read_text())
assert hashlib.sha256((work/'src/mjlab_microduck/tasks/__init__.py').read_bytes()).hexdigest()==request['task_source_sha256']
print(run)
PY
)"
if [[ "$mode" == --check-only ]]; then
  printf 'EVALUATION_INPUTS_VERIFIED: %s; no simulation started.\n' "$run_id"
  exit 0
fi
record="$project_root/records/experiments/$run_id"
model="$project_root/artifacts/$run_id/policy.onnx"
log_file="$record/fixed-evaluation-$(date +%Y%m%dT%H%M%S).log"
cp "${BASH_SOURCE[0]}" "$record/fixed-evaluation-entry.sh"
# run_fixed_evaluation.sh holds the shared compute lock for the simulation.
bash "$project_root/scripts/validation/run_fixed_evaluation.sh" --onnx "$model" 2>&1 | tee "$log_file"
evaluation="$(python3 - "$project_root" "$log_file" "$model" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
root,log,model=map(Path,sys.argv[1:])
paths=re.findall(r'^EVALUATION completed: (.+)$',log.read_text(),re.M)
assert len(paths)==1, 'Expected one completed evaluation in this exact transcript'
folder=Path(paths[0]).resolve()
assert folder.parent==root/'records/evaluations'
report=json.loads((folder/'evaluation.json').read_text())
protocol=json.loads((root/'configs/experiments/formal_velocity_tracking.json').read_text())
assert report['status']=='completed' and report['phase']=='reference'
assert report['seed']==protocol['evaluation']['development_seed']
assert report['num_envs']==protocol['evaluation']['episodes_per_condition']
assert report['episode_seconds']==protocol['evaluation']['episode_seconds']
assert report['model_sha256']==hashlib.sha256(model.read_bytes()).hexdigest()
assert report['protocol_sha256']==hashlib.sha256((root/'configs/experiments/formal_velocity_tracking.json').read_bytes()).hexdigest()
assert {c['condition'] for c in report['conditions']}==set(protocol['evaluation']['commands'])
print(folder)
PY
)"
"$work/.venv/bin/python" -B "$project_root/scripts/validation/verify_fixed_evaluation.py" "$evaluation" 2>&1 | tee -a "$log_file"
python3 - "$record" "$evaluation" "$log_file" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
record,evaluation,log=map(Path,sys.argv[1:])
report=json.loads((evaluation/'evaluation.json').read_text())
check=json.loads((evaluation/'verification.json').read_text())
assert check['status']=='passed' and check['run_id']==report['run_id']
result=dict(status='fixed-evaluation-completed',checked_at=datetime.now(timezone.utc).isoformat(),
            evaluation=str(evaluation),run_id=report['run_id'],model_sha256=report['model_sha256'],
            all_numeric_criteria_passed=report['meets_all_numeric_criteria'],
            independent_verification_passed=True,heldout_evaluation_completed=False,
            hardware_verified=False,transcript=str(log))
(record/'fixed-evaluation-link.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
