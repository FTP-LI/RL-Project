"""Close the numeric zero comparison and continue the dynamic review."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
evidence = json.loads((ROOT/'records/reports/roll-matched-factor-preflight-20261116/friction-raw-zero-comparison.json').read_text(encoding='utf-8'))
path = ROOT/'project.json'
project = json.loads(path.read_text(encoding='utf-8'))
task, goal = project['staged_training_request'], project['goal_execution']
assert evidence['status'] == 'raw-comparison-completed-dynamic-review-required'
assert not task['learning_running'] and not task['simulation_running']
assert task['current_protocol'] == 'configs/experiments/roll_factor_fp_raw_zero_friction-low_s1242_20261116.json'
task.update(current_protocol_frozen=False, current_stage='roll-parameter-dynamic-admission-review',
    next_gate='Assess constraint ordering and numeric differences, then establish an observed bounded dynamic comparison.')
goal['roll_parameter_resume_preflight'].update(status='raw-friction-comparison-reviewed-dynamics-pending',
    raw_comparison='records/reports/roll-matched-factor-preflight-20261116/friction-raw-zero-comparison.json',
    next_gate='Assess constraint ordering and numeric differences before an observed bounded dynamic comparison.')
goal['native_file_observer_evaluation']['zero_coordinator_status'] = 'raw-friction-zero-independently-reviewed'
temp = path.with_name('project.raw-friction-close-tmp.json')
assert not temp.exists()
temp.write_text(json.dumps(project, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
os.replace(temp, path)
print(json.dumps({'status': 'raw-friction-zero-closed', 'stage': task['current_stage']}))
