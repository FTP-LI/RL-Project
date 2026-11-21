"""Freeze one new plane-aware nominal zero-step diagnostic."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import roll_fingerprint_plane_zero as plane

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'records/reports/roll-matched-factor-preflight-20261116'
BASE = ROOT / 'configs/experiments/roll_factor_fp_logged_zero_nominal_s1242_20261116.json'
BASE_WRAPPER = ROOT / 'configs/experiments/roll_factor_fp_logged_native_nominal_s1242_20261116.json'
PLAN = ROOT / 'configs/experiments/roll_factor_fp_plane_zero_nominal_s1242_20261116.json'
WRAPPER = ROOT / 'configs/experiments/roll_factor_fp_plane_native_nominal_s1242_20261116.json'
LIB = Path('/home/atlas/XGO-Duck/workspace/xgoduck_rl/.venv/lib/python3.12/site-packages')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def write_project(project):
    path = ROOT / 'project.json'
    temporary = path.with_name('project.plane-zero-tmp.json')
    assert not temporary.exists()
    temporary.write_bytes(encode(project))
    os.replace(temporary, path)


def main():
    index = REPORT / 'plane-diagnostic-plan.json'
    assert not any(path.exists() for path in (PLAN, WRAPPER, index))
    project = read(ROOT / 'project.json')
    task, goal = project['staged_training_request'], project['goal_execution']
    assert goal['status'] == 'active' and goal['pause_requested_by_user'] is False
    assert not task['pause_requested'] and not task['learning_running'] and not task['simulation_running']
    assert not task['current_protocol_frozen'] and task['current_stage'] == 'roll-parameter-entity-review'
    task.update(current_stage='roll-parameter-plane-diagnostic-preparation',
                next_gate='Prepare one separately frozen zero-only run using the plane-aware fingerprint inventory.')
    goal['roll_parameter_resume_preflight']['status'] = 'plane-zero-preparation'
    write_project(project)

    analysis = read(REPORT / 'logged-zero-stop-analysis.json')
    scene = read(REPORT / 'terrain-scene-cpu.json')
    assert analysis['status'] == 'saved-zero-stop-and-source-reviewed' and analysis['physics_substeps'] == 0
    assert scene['entities'] == ['robot', 'terrain'] and scene['terrain_configuration'] == 'plane'
    for name, expected_status in [('terrain-inventory-fixture.json', 'passed-cpu-plane-inventory-fixture-only'),
                                  ('terrain-hook-fixture.json', 'passed-cpu-plane-hook-fixture-only')]:
        fixture = read(REPORT / name)
        assert fixture['status'] == expected_status
        for source, digest in fixture['source_sha256'].items():
            assert sha(ROOT / source) == digest, source
    for source, digest in analysis['source_sha256'].items():
        assert sha(ROOT / source) == digest, source
    base, previous_wrapper = read(BASE), read(BASE_WRAPPER)
    assert previous_wrapper['upstream_plan_sha256'] == sha(BASE)
    assert base['seed'] == 1242 and base['factor_arm'] == 'nominal'
    run_id = 'duck_roll-factor_nominal_s1242_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    output = ROOT / 'records/reports' / run_id
    assert not output.exists()
    plan = deepcopy(base)
    plan.update(run_id=run_id, guarded_output=output.relative_to(ROOT).as_posix(),
                page=f'records/{run_id}-live.html', admission_record=f'records/reports/{run_id}/admission.json')
    previous = ROOT / 'records/reports/duck_roll-factor_nominal_s1242_20261116T134504'
    extras = plane.REQUIRED | {'scripts/validation/prepare_roll_fingerprint_plane_zero.py',
        BASE.relative_to(ROOT).as_posix(), BASE_WRAPPER.relative_to(ROOT).as_posix(),
        (previous / 'native-zero-run/result.json').relative_to(ROOT).as_posix(),
        (previous / 'zero/actuator-observed-start.npz').relative_to(ROOT).as_posix()}
    plan['frozen_sha256'].update({name: sha(ROOT / name) for name in sorted(extras)})
    new_libraries = [LIB / 'mjlab/terrains/terrain_entity.py', LIB / 'mjlab/entity/data.py']
    plan['library_source_sha256'].update({str(path): sha(path) for path in new_libraries})
    plan_bytes = encode(plan)
    plan_name, wrapper_name = (path.relative_to(ROOT).as_posix() for path in (PLAN, WRAPPER))
    wrapper = deepcopy(previous_wrapper)
    wrapper.update(created_at=datetime.now(timezone.utc).isoformat(), upstream_plan=plan_name,
        upstream_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        source_sha256={name: sha(ROOT / name) for name in sorted(plane.REQUIRED)},
        scope='One separate plane-aware zero-only fingerprint check; no dynamics or learning.')
    PLAN.write_bytes(plan_bytes)
    WRAPPER.write_bytes(encode(wrapper))
    output.mkdir()
    (output / 'windows-entry.json').write_bytes(encode({'status': 'native-diagnostic-prepared',
        'protocol_sha256': sha(PLAN), 'browser_visual_verified': False, 'real_observer_verified': False}))
    fake = deepcopy(project)
    fake['staged_training_request'].update(current_stage=plan['stage'], current_protocol=plan_name,
        current_protocol_frozen=True, learning_running=False, simulation_running=False)
    fake['goal_execution']['native_file_observer_evaluation'].update(native_zero_plan=wrapper_name,
        native_zero_plan_sha256=sha(WRAPPER))
    original_read = plane.old.read
    def fake_read(path):
        return deepcopy(fake) if Path(path).resolve() == (ROOT / 'project.json').resolve() else original_read(path)
    with patch.object(plane.old, 'read', side_effect=fake_read), plane.plane_bound(plan_name, wrapper_name):
        with plane.repair.repair_context(wrapper):
            assert plane.logged.diagnostic.coordinator.verify_wrapper(WRAPPER, wrapper, fake) == sha(WRAPPER)
            assert plane.logged.diagnostic.coordinator.provenance.audit()['status'] == 'source-audit-passed-native-zero-only'
    result = {'status': 'frozen-separate-plane-diagnostic-awaiting-observer-and-registration',
        'created_at': datetime.now(timezone.utc).isoformat(), 'run_id': run_id, 'seed': 1242,
        'plan': plan_name, 'plan_sha256': sha(PLAN), 'wrapper': wrapper_name, 'wrapper_sha256': sha(WRAPPER),
        'output': output.relative_to(ROOT).as_posix(), 'previous_stop': analysis['run_id'],
        'new_library_sha256': {str(path): sha(path) for path in new_libraries},
        'physical_steps': 0, 'learning_updates': 0, 'hardware_io': False,
        'automatic_retry': False, 'dynamic_execution_authorized': False}
    index.write_bytes(encode(result))
    task.update(current_stage=plan['stage'], current_protocol=plan_name, current_protocol_frozen=True,
        next_gate='Observe and run the separately frozen plane-aware zero-only diagnostic.')
    goal['native_file_observer_evaluation'].update(native_zero_plan=wrapper_name, native_zero_plan_sha256=sha(WRAPPER),
        zero_coordinator_status='plane-diagnostic-admitted-awaiting-view')
    goal['roll_parameter_resume_preflight'].update(status='plane-zero-admitted-awaiting-view',
        plane_diagnostic_plan=index.relative_to(ROOT).as_posix())
    write_project(project)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
