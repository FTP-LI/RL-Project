"""Close the verified nominal zero check and prepare matched factor review."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
evidence = json.loads((ROOT / 'records/reports/roll-matched-factor-preflight-20261116/scene-nominal-zero-verification.json').read_text(encoding='utf-8'))
path = ROOT / 'project.json'
project = json.loads(path.read_text(encoding='utf-8'))
task, goal = project['staged_training_request'], project['goal_execution']
assert evidence['status'] == 'passed-independent-nominal-scene-zero-readback'
assert not task['learning_running'] and not task['simulation_running']
assert task['current_protocol'] == 'configs/experiments/roll_factor_fp_scene_zero_nominal_s1242_20261116.json'
task.update(current_protocol_frozen=False, current_stage='roll-parameter-matched-scene-zero-preparation',
    next_gate='Prepare distinct low-friction and high-voltage zero checks; compare initial states before dynamics.')
goal['roll_parameter_resume_preflight'].update(status='nominal-scene-zero-verified',
    nominal_scene_zero='records/reports/roll-matched-factor-preflight-20261116/scene-nominal-zero-verification.json',
    next_gate='Prepare distinct low-friction and high-voltage zero checks; compare initial states before dynamics.')
goal['native_file_observer_evaluation']['zero_coordinator_status'] = 'scene-nominal-zero-independently-verified'
temp = path.with_name('project.scene-zero-close-tmp.json')
assert not temp.exists()
temp.write_text(json.dumps(project, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
os.replace(temp, path)
print(json.dumps({'status': 'nominal-zero-closed-goal-active', 'stage': task['current_stage']}))
