"""Close the independently checked nominal numeric zero run."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
evidence = json.loads((ROOT/'records/reports/roll-matched-factor-preflight-20261116/raw-nominal-zero-verification.json').read_text(encoding='utf-8'))
path = ROOT/'project.json'
project = json.loads(path.read_text(encoding='utf-8'))
task, goal = project['staged_training_request'], project['goal_execution']
assert evidence['status'] == 'passed-nominal-raw-zero-readback'
assert not task['learning_running'] and not task['simulation_running']
assert task['current_protocol'] == 'configs/experiments/roll_factor_fp_raw_zero_nominal_s1242_20261116.json'
task.update(current_protocol_frozen=False, current_stage='roll-parameter-raw-compare-preparation',
    next_gate='Prepare one low-friction zero-only numeric capture and compare saved arrays.')
goal['roll_parameter_resume_preflight'].update(status='nominal-raw-zero-verified',
    nominal_raw_zero='records/reports/roll-matched-factor-preflight-20261116/raw-nominal-zero-verification.json',
    next_gate='Prepare one low-friction zero-only numeric capture and compare saved arrays.')
goal['native_file_observer_evaluation']['zero_coordinator_status'] = 'nominal-raw-zero-independently-verified'
temp = path.with_name('project.raw-nominal-close-tmp.json')
assert not temp.exists()
temp.write_text(json.dumps(project, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
os.replace(temp, path)
print(json.dumps({'status': 'nominal-raw-zero-closed', 'stage': task['current_stage']}))
