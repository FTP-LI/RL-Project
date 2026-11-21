#!/usr/bin/env bash
# Fixed weights, same dynamics and acceptance thresholds. Only command magnitude
# changes, using archived UI defaults and limits. All output stays in this shared
# foreground terminal. This is not training and does not unlock paused training.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$project_root/scripts/setup/project_environment.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl TZ=Asia/Shanghai
cd "$HOME/XGO-Duck/workspace/xgoduck_rl"
log="$project_root/records/evaluations/amplitude-console-$(date +%Y%m%dT%H%M%S).log"
run_diagnostic() {
  printf 'COMMAND: evaluate_fixed_policy.py --command-amplitude\n'
  .venv/bin/python -B "$project_root/scripts/validation/evaluate_fixed_policy.py" --command-amplitude || return $?
  run_dir="$(.venv/bin/python -c 'import json,sys; from pathlib import Path; p=json.load(open(sys.argv[1])); r=json.load(open(Path(p["path"])/"evaluation.json")); assert r["command_amplitude_diagnostic"] and r["status"]=="completed"; print(p["path"])' "$project_root/records/evaluations/current-evaluation.json")" || return $?
  printf 'COMMAND: verify_fixed_evaluation.py %s\n' "$run_dir"
  .venv/bin/python -B "$project_root/scripts/validation/verify_fixed_evaluation.py" "$run_dir" || return $?
  printf 'COMMAND: analyze_command_amplitude.py %s\n' "$run_dir"
  .venv/bin/python -B "$project_root/scripts/validation/analyze_command_amplitude.py" "$run_dir" || return $?
}
set +e
run_diagnostic 2>&1 | tee "$log"
result=${PIPESTATUS[0]}
printf '\nAMPLITUDE_DIAGNOSTIC_EXIT_CODE=%s\n' "$result" | tee -a "$log"
exit "$result"
