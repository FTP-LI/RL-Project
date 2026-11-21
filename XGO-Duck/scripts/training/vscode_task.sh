#!/usr/bin/env bash
# One foreground VS Code task entry for WSL and native Ubuntu.
# Source/experiment comments live beside the actual executable entry scripts.
# Training remains blocked while project.json records the user's pause request.
set -euo pipefail
task="${1:-overview}"
seed="${2:-42}"
board_url="${3:-}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "$seed" in 42|43|44) ;; *) echo 'Seed must be 42, 43 or 44'; exit 2;; esac
case "$task" in overview|report|servo_mapping|board_status|eval_preflight|eval_reference|diagnostic_preflight|diagnostic_comparison|diagnostic_verify|command_amplitude|offline_efficiency|interface_audit|host_profile|resume_audit|resume_baseline|baseline|candidate|video_preflight) ;; *) echo 'Unknown task'; exit 2;; esac
python3 - "$project_root" "$task" <<'PY'
import json,sys
from pathlib import Path
root,task=Path(sys.argv[1]),sys.argv[2]
p=json.loads((root/'project.json').read_text(encoding='utf-8-sig'))
print('XGO-Duck |',p['current_phase'],flush=True)
print('Training:',p['formal_training']['status'],flush=True)
print('Recovery checkpoint:',p['formal_training'].get('resume_checkpoint','none'),flush=True)
if task in ('baseline','candidate','video_preflight','resume_baseline'):
    if p['formal_training']['status'].startswith('paused'):
        raise SystemExit('Training is paused by user. Record explicit resume instruction before launching. No training started.')
    if task=='candidate' and not p['formal_training']['stable_gait_verified']:
        raise SystemExit('Stable baseline gait must be verified before candidate comparison.')
PY
if [[ "$task" == overview ]]; then
  echo 'Read docs/current_goal.md, docs/reports/development_report.md and scripts/training/experiments/*.sh'
  echo 'No training or installation was started.'
  exit 0
fi
if [[ "$task" == report ]]; then
  "$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" "$project_root/scripts/validation/render_development_report.py"
  "$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" -B "$project_root/scripts/validation/verify_report_artifacts.py"
  exit $?
fi
if [[ "$task" == servo_mapping ]]; then
  "$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" "$project_root/scripts/validation/inspect_servo_mapping.py"
  exit $?
fi
if [[ "$task" == host_profile ]]; then
  python3 -B "$project_root/scripts/setup/inspect_linux_host.py"
  exit $?
fi
if [[ "$task" == board_status ]]; then
  mkdir -p "$project_root/records/hardware/status-tool"
  log_file="$project_root/records/hardware/status-tool/console-$(date +%Y%m%dT%H%M%S).log"
  set +e
  {
    if [[ -z "$board_url" ]]; then
      echo 'No board URL supplied: offline software checks only; no board connection.'
      "$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" -u -B "$project_root/scripts/validation/check_board_status_capture.py"
    else
      python3 -u -B "$project_root/scripts/validation/capture_board_status.py" --base-url "$board_url"
    fi
  } 2>&1 | tee "$log_file"
  result=${PIPESTATUS[0]}
  printf '\nSTATUS_CAPTURE_EXIT_CODE=%s\n' "$result" | tee -a "$log_file"
  exit "$result"
fi
if [[ "$task" == resume_audit ]]; then
  run_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["formal_training"]["run_id"])' "$project_root/project.json")"
  log_file="$project_root/records/experiments/$run_id/resume-analysis/console-$(date +%Y%m%dT%H%M%S).log"
  set +e
  {
    "$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" -u -B "$project_root/scripts/validation/inspect_resume_checkpoint.py" &&
    "$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" -u -B "$project_root/scripts/validation/verify_resume_preparation.py"
  } 2>&1 | tee "$log_file"
  result=${PIPESTATUS[0]}
  printf '\nRESUME_AUDIT_EXIT_CODE=%s\n' "$result" | tee -a "$log_file"
  exit "$result"
fi
if [[ "$task" == diagnostic_verify ]]; then
  batch_dir="$(python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(p.parent/json.loads(p.read_text())["batch_id"])' "$project_root/records/evaluations/current-adaptation-diagnostics.json")"
  log_file="$batch_dir/reverification-console-$(date +%Y%m%dT%H%M%S).log"
  set +e
  flock -n "$HOME/XGO-Duck/training.lock" "$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" -u -B "$project_root/scripts/validation/verify_adaptation_diagnostics.py" "$batch_dir" --resume-verification 2>&1 | tee "$log_file"
  result=${PIPESTATUS[0]}
  printf '\nVERIFICATION_EXIT_CODE=%s\n' "$result" | tee -a "$log_file"
  exit "$result"
fi
if [[ "$task" == offline_efficiency ]]; then
  run_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["formal_training"]["run_id"])' "$project_root/project.json")"
  "$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" -B "$project_root/scripts/validation/analyze_paused_training.py" "$run_id"
  "$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" -B "$project_root/scripts/validation/plot_training_efficiency.py" "$project_root/records/experiments/$run_id/efficiency-analysis"
  exit $?
fi
if [[ "$task" == interface_audit ]]; then
  reference="$(python3 -c 'import json,sys; from pathlib import Path; print(Path(json.load(open(sys.argv[1]))["official_pretrained_reference"]["evidence"]).parent)' "$project_root/project.json")"
  "$HOME/XGO-Duck/workspace/xgoduck_rl/.venv/bin/python" -B "$project_root/scripts/validation/audit_reference_interface.py" "$project_root/$reference"
  exit $?
fi
if [[ "$task" == eval_preflight ]]; then
  exec bash "$project_root/scripts/validation/run_fixed_evaluation.sh" --preflight
fi
if [[ "$task" == command_amplitude ]]; then
  exec flock -n "$HOME/XGO-Duck/training.lock" bash "$project_root/scripts/validation/run_command_amplitude.sh"
fi
if [[ "$task" == eval_reference ]]; then
  exec bash "$project_root/scripts/validation/run_fixed_evaluation.sh"
fi
if [[ "$task" == diagnostic_preflight ]]; then
  exec bash "$project_root/scripts/validation/run_adaptation_diagnostics.sh" --preflight
fi
if [[ "$task" == diagnostic_comparison ]]; then
  exec bash "$project_root/scripts/validation/run_adaptation_diagnostics.sh"
fi
case "$task" in
  baseline) entry=official_baseline.sh; extra=() ;;
  candidate) entry=tracking_reward.sh; extra=() ;;
  video_preflight) entry=official_baseline.sh; extra=(--preflight) ;;
  resume_baseline) entry=resume_baseline.sh; extra=() ;;
esac
# A shared lock prevents different VS Code training tasks from using the GPU
# concurrently. It is held only for this visible foreground command.
flock -n "$HOME/XGO-Duck/training.lock" bash "$project_root/scripts/training/experiments/$entry" --seed "$seed" "${extra[@]}"
