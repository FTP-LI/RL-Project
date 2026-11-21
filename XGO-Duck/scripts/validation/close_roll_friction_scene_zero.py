"""Close the low-friction zero check and retain its unresolved differences."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
evidence = json.loads((ROOT/'records/reports/roll-matched-factor-preflight-20261116/friction-low-scene-zero-comparison.json').read_text(encoding='utf-8'))
path = ROOT / 'project.json'
project = json.loads(path.read_text(encoding='utf-8'))
task, goal = project['staged_training_request'], project['goal_execution']
assert evidence['status'] == 'review-required-internal-state-differences'
assert not task['learning_running'] and not task['simulation_running']
assert task['current_protocol'] == 'configs/experiments/roll_factor_fp_scene_zero_friction-low_s1242_20261116.json'
task.update(current_protocol_frozen=False, current_stage='roll-parameter-friction-difference-review',
    next_gate='Investigate saved nominal/low-friction internal state differences before further factor runs or dynamics.')
goal['roll_parameter_resume_preflight'].update(status='friction-zero-internal-differences-review-required',
    friction_scene_zero='records/reports/roll-matched-factor-preflight-20261116/friction-low-scene-zero-comparison.json',
    next_gate='Investigate internal differences before further factor runs or dynamics.')
goal['native_file_observer_evaluation']['zero_coordinator_status'] = 'friction-zero-closed-review-required'
temporary = path.with_name('project.friction-zero-close-tmp.json')
assert not temporary.exists()
temporary.write_text(json.dumps(project, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
os.replace(temporary, path)
print(json.dumps({'status': 'friction-zero-closed-review-required', 'stage': task['current_stage']}))
