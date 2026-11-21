#!/usr/bin/env bash
# Run visibly after bootstrap and cache relocation have passed.
set -euo pipefail
run_kind="${1:?Pass baseline or candidate}"
case "$run_kind" in baseline|candidate) ;; *) echo 'Expected baseline or candidate'; exit 2 ;; esac
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
working_copy="$HOME/XGO-Duck/workspace/xgoduck_rl"
run_name="${run_kind}_smoke_$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)"
record_dir="$project_root/records/experiments/$run_name"
mkdir -p "$record_dir" "$CUDA_CACHE_PATH" "$TORCH_HOME" "$TRITON_CACHE_DIR"
exec > >(tee -a "$record_dir/training.log") 2>&1
trap 'result=$?; printf "%s\n" "$result" > "$record_dir/exit-code.txt"; printf "SMOKE_EXIT_CODE=%s\n" "$result"' EXIT

read -r num_envs iterations seed < <(python3 - "$project_root" "$run_kind" "$working_copy" "$record_dir" "$run_name" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
project, kind, work, record, run_name = sys.argv[1:]
definition = json.loads((Path(project) / 'configs/experiments/velocity_tracking.json').read_text())
source = Path(work) / definition['change']['file']
actual = hashlib.sha256(source.read_bytes()).hexdigest()
key = 'source_sha256' if kind == 'baseline' else 'candidate_sha256'
if actual != definition['change'][key]:
    raise SystemExit(f'{kind} source hash mismatch: {actual}. Apply or restore the documented patch before training.')
smoke = definition['smoke']
request = {
    'started_at': datetime.now(timezone.utc).isoformat(),
    'run_kind': kind, 'run_name': run_name, 'task': definition['task'],
    'smoke': smoke, 'upstream_commit': definition['upstream_commit'],
    'working_copy': work, 'task_source_sha256': actual,
    'uv_lock_sha256': hashlib.sha256((Path(work) / 'uv.lock').read_bytes()).hexdigest(),
    'scope': 'Short pipeline validation, not gait-quality or hardware acceptance',
}
(Path(record) / 'training-request.json').write_text(json.dumps(request, indent=2) + '\n')
(Path(record) / 'experiment-definition.json').write_text(json.dumps(definition, indent=2) + '\n')
print(int(smoke['num_envs']), int(smoke['max_iterations']), int(smoke['seed']))
PY
)
cd "$working_copy"
export PYTHONUNBUFFERED=1
set -x
uv run --locked --offline train Mjlab-Velocity-Flat-XgoDuck \
    --env.scene.num-envs "$num_envs" \
    --agent.max-iterations "$iterations" \
    --agent.seed "$seed" \
    --agent.run-name "$run_name" \
    --agent.logger tensorboard
set +x
printf 'Training command finished. Verify checkpoints for run name: %s\n' "$run_name"
