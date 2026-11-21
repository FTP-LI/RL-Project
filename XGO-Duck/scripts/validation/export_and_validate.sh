#!/usr/bin/env bash
# Execute in the visible project console after checking the training result.
# Support completed continuations as well as the original short pipeline runs.
# The continuation path requires an independently audited final update budget.
# --check-only reads provenance and exits; it never exports or runs simulation.
set -euo pipefail
run_id="${1:?Pass the recorded training run id}"
checkpoint="${2:?Pass the exact checkpoint path}"
mode="${3:-}"
[[ -z "$mode" || "$mode" == --check-only ]] || exit 2
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
work="$HOME/XGO-Duck/workspace/xgoduck_rl"
plan_json="$(python3 -B "$project_root/scripts/validation/prepare_policy_export.py" "$run_id" "$checkpoint")"
if [[ "$mode" == --check-only ]]; then
  printf '%s\n' "$plan_json"
  exit 0
fi
# Hold the same lock as training for export and numerical-parity simulation.
exec 9>"$HOME/XGO-Duck/training.lock"
flock -n 9 || { echo 'Project compute task is active; no export started.'; exit 3; }
checkpoint="$(realpath -e "$checkpoint")"
record_dir="$project_root/records/experiments/$run_id"
seed="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["seed"])' <<<"$plan_json")"
run_dir="$(dirname "$checkpoint")"
model="$run_dir/policy.onnx"
validation_dir="$run_dir/export_validation"
[[ ! -e "$model" && ! -e "$validation_dir" && ! -e "$project_root/artifacts/$run_id" ]] || { echo 'Export output already exists; preserve prior evidence.'; exit 4; }
exec > >(tee -a "$record_dir/export-$(date +%H%M%S).log") 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$record_dir/export-exit-code.txt"; printf "EXPORT_CHECK_EXIT_CODE=%s\n" "$result"' EXIT
printf '%s\n' "$plan_json" | tee "$record_dir/export-inputs.json"
cp "${BASH_SOURCE[0]}" "$record_dir/export-entry.sh"
cp "$project_root/scripts/validation/prepare_policy_export.py" "$record_dir/export-input-check.py"
cd "$work"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=egl
set -x
uv run --locked --offline python scripts/export.py Mjlab-Velocity-Flat-XgoDuck \
  --checkpoint-file "$checkpoint" --num-envs 1 --onnx-file "$model"
uv run --locked --offline python "$project_root/scripts/validation/validate_export.py" \
  --checkpoint "$checkpoint" --onnx "$model" --output-dir "$validation_dir" --steps 1000 --seed "$seed"
set +x
python3 - "$project_root" "$run_id" "$checkpoint" "$model" "$validation_dir" <<'PY'
import hashlib, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
project, run_id, checkpoint, model, validation = sys.argv[1:]
record = Path(project) / 'records/experiments' / run_id
request = json.loads((record / 'training-request.json').read_text())
inputs = json.loads((record / 'export-inputs.json').read_text())
report = json.loads((Path(validation) / 'export-validation.json').read_text())
assert report['status'] == 'passed'
assert request.get('run_id', request.get('run_name')) == run_id
assert report['checkpoint_sha256'] == hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
assert report['checkpoint_sha256'] == inputs['checkpoint_sha256']
artifact = Path(project) / 'artifacts' / run_id
artifact.mkdir(parents=True, exist_ok=False)
shutil.copy2(model, artifact / 'policy.onnx')
shutil.copy2(Path(validation) / 'export-validation.json', record / 'export-validation.json')
manifest = {
    'created_at': datetime.now(timezone.utc).isoformat(), 'run_id': run_id,
    'training_request': request, 'checkpoint': checkpoint,
    'checkpoint_sha256': report['checkpoint_sha256'],
    'onnx_sha256': report['onnx_sha256'], 'validation': validation,
    'artifact': str(artifact), 'status': inputs['intended_artifact_status'],
    'cumulative_updates': inputs['cumulative_updates'],
    'runtime_1_22_1_compatibility_verified': False,
    'gait_quality_verified': False, 'hardware_deployment_ready': False,
}
assert hashlib.sha256((artifact / 'policy.onnx').read_bytes()).hexdigest() == report['onnx_sha256']
for path in [record / 'artifact-manifest.json', artifact / 'manifest.json']:
    path.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
PY
