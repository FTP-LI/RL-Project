"""Close the stopped diagnostic while keeping the overall goal active."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
analysis = json.loads((ROOT / 'records/reports/roll-matched-factor-preflight-20261116/logged-zero-stop-analysis.json').read_text(encoding='utf-8'))
project_path = ROOT / 'project.json'
project = json.loads(project_path.read_text(encoding='utf-8'))
task, goal = project['staged_training_request'], project['goal_execution']
assert analysis['status'] == 'saved-zero-stop-and-source-reviewed'
assert not task['learning_running'] and not task['simulation_running']
assert task['current_protocol'] == 'configs/experiments/roll_factor_fp_logged_zero_nominal_s1242_20261116.json'
task.update(current_protocol_frozen=False, current_stage='roll-parameter-entity-review',
    next_gate='Review terrain entity state and adapt fingerprint inventory; no dynamic parameter rollouts yet.')
goal['roll_parameter_resume_preflight'].update(status='logged-zero-stopped-entity-review-required',
    logged_stop='records/reports/duck_roll-factor_nominal_s1242_20261116T134504/native-zero-run/result.json',
    logged_analysis='records/reports/roll-matched-factor-preflight-20261116/logged-zero-stop-analysis.json',
    next_gate='Review terrain entity state and adapt fingerprint inventory; no dynamic parameter rollouts yet.')
goal['native_file_observer_evaluation']['zero_coordinator_status'] = 'logged-diagnostic-stopped-review-required'
temp = project_path.with_name('project.roll-zero-close-tmp.json')
assert not temp.exists()
temp.write_text(json.dumps(project, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
os.replace(temp, project_path)
print(json.dumps({'status': 'diagnostic-closed-goal-active', 'stage': task['current_stage'],
                  'training_running': task['learning_running'], 'simulation_running': task['simulation_running']}))
