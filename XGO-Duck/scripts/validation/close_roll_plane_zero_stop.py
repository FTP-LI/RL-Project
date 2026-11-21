"""Record and close the stopped plane-aware zero-step diagnostic."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'records/reports/duck_roll-factor_nominal_s1242_20261116T135313'
OUTPUT = ROOT / 'records/reports/roll-matched-factor-preflight-20261116/plane-zero-stop-analysis.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert not OUTPUT.exists()
result = read(RUN / 'native-zero-run/result.json')
runtime = read(RUN / 'zero/runtime-contract.json')
receipt = read(RUN / 'native-zero-execution.json')
log_path = RUN / 'native-zero-execution.log'
log = log_path.read_text(encoding='utf-8')
assert receipt['log_sha256'] == sha(log_path) and receipt['exit_code'] == 2
assert result['status'] == 'stopped-review-required' and result['error_type'] == 'AssertionError'
assert "assert type(sensor).__name__ in {'ContactSensor','TerrainHeightSensor'}" in log
assert runtime['control_steps'] == runtime['physics_substeps'] == 0
assert not (RUN / 'zero/start-fingerprints.json').exists()
observation = read(RUN / 'native-zero-run/ui-proof.json')
assert observation['screenshot_sha256'] == sha(RUN / 'native-zero-run/ui-check.jpg')
analysis = {'status': 'saved-plane-zero-stop-reviewed', 'checked_at': datetime.now(timezone.utc).isoformat(),
    'run_id': observation['run_id'], 'error': 'existing sensor-type inventory assertion',
    'actual_rejected_sensor_type_saved': False, 'control_steps': 0, 'physics_substeps': 0,
    'learning_updates': 0, 'hardware_io': False, 'fingerprint_complete': False,
    'source_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in
        [Path(__file__), log_path, RUN / 'zero/runtime-contract.json', RUN / 'native-zero-run/ui-proof.json']}}
OUTPUT.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

project_path = ROOT / 'project.json'
project = read(project_path)
task, goal = project['staged_training_request'], project['goal_execution']
assert task['current_protocol'] == 'configs/experiments/roll_factor_fp_plane_zero_nominal_s1242_20261116.json'
assert not task['learning_running'] and not task['simulation_running']
task.update(current_protocol_frozen=False, current_stage='roll-parameter-sensor-review',
    next_gate='Inspect the actual sensor types and state inventory before any new zero-step diagnosis.')
goal['roll_parameter_resume_preflight'].update(status='plane-zero-stopped-sensor-review-required',
    plane_stop='records/reports/duck_roll-factor_nominal_s1242_20261116T135313/native-zero-run/result.json',
    plane_stop_analysis=OUTPUT.relative_to(ROOT).as_posix(),
    next_gate='Inspect actual sensor types and state inventory; no dynamic parameter rollouts yet.')
goal['native_file_observer_evaluation']['zero_coordinator_status'] = 'plane-diagnostic-stopped-review-required'
temporary = project_path.with_name('project.plane-stop-tmp.json')
assert not temporary.exists()
temporary.write_text(json.dumps(project, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
os.replace(temporary, project_path)
print(json.dumps(analysis, ensure_ascii=False))
