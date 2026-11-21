"""Freeze one zero-step numeric capture for the friction-difference review."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import roll_fingerprint_raw_zero as entry

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'records/reports/roll-matched-factor-preflight-20261116'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def save_project(project):
    path = ROOT/'project.json'
    temp = path.with_name('project.raw-zero-tmp.json')
    assert not temp.exists()
    temp.write_bytes(encode(project))
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=('nominal', 'friction-low'))
    arm = parser.parse_args().arm
    base = ROOT/f'configs/experiments/roll_factor_fp_scene_zero_{arm}_s1242_20261116.json'
    base_wrapper = ROOT/f'configs/experiments/roll_factor_fp_scene_native_{arm}_s1242_20261116.json'
    plan_path = ROOT/f'configs/experiments/roll_factor_fp_raw_zero_{arm}_s1242_20261116.json'
    wrapper_path = ROOT/f'configs/experiments/roll_factor_fp_raw_native_{arm}_s1242_20261116.json'
    index = REPORT/f'raw-zero-{arm}-plan.json'
    assert not any(path.exists() for path in (plan_path, wrapper_path, index))
    project = read(ROOT/'project.json')
    task, goal = project['staged_training_request'], project['goal_execution']
    assert goal['status'] == 'active' and not goal['pause_requested_by_user']
    assert not task['pause_requested'] and not task['learning_running'] and not task['simulation_running']
    assert not task['current_protocol_frozen']
    allowed_stage = 'roll-parameter-friction-difference-review' if arm == 'nominal' else 'roll-parameter-raw-compare-preparation'
    assert task['current_stage'] == allowed_stage
    cpu = read(REPORT/'raw-capture-cpu/verification.json')
    identity = read(REPORT/'friction-cache-identity-review.json')
    assert cpu['status'] == 'passed-initialized-cpu-numeric-capture' and cpu['fields'] == 34
    assert cpu['arrays_sha256'] == sha(REPORT/'raw-capture-cpu/selected-zero-numeric.npz')
    assert identity['status'] == 'saved-cache-identity-and-cpu-repeat-reviewed'
    assert identity['gpu_other_changed_field_count'] == 25
    for evidence in (cpu, identity):
        for name, digest in evidence['source_sha256'].items():
            assert sha(ROOT/name) == digest, name
    nominal_raw = None
    if arm == 'friction-low':
        nominal_raw = read(REPORT/'raw-nominal-zero-verification.json')
        assert nominal_raw['status'] == 'passed-nominal-raw-zero-readback'
    task['next_gate'] = f'Freeze one {arm} zero-only numeric capture; no physical steps.'
    goal['roll_parameter_resume_preflight']['status'] = f'raw-{arm}-zero-preparation'
    save_project(project)

    p, old_wrapper = read(base), read(base_wrapper)
    assert old_wrapper['upstream_plan_sha256'] == sha(base) and p['factor_arm'] == arm and p['seed'] == 1242
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    # The existing safety gate requires the exact arm-based run-ID pattern.
    # The raw-capture purpose is recorded in the plan name and index instead.
    run_id = f'duck_roll-factor_{arm}_s1242_{stamp}'
    output = ROOT/'records/reports'/run_id
    assert not output.exists()
    p = deepcopy(p)
    p.update(run_id=run_id, guarded_output=output.relative_to(ROOT).as_posix(),
        page=f'records/{run_id}-live.html', admission_record=f'records/reports/{run_id}/admission.json')
    extras = entry.REQUIRED | {'scripts/validation/prepare_roll_friction_raw_zero.py',
        base.relative_to(ROOT).as_posix(), base_wrapper.relative_to(ROOT).as_posix()}
    if nominal_raw is not None:
        extras.add('records/reports/roll-matched-factor-preflight-20261116/raw-nominal-zero-verification.json')
    p['frozen_sha256'].update({name: sha(ROOT/name) for name in sorted(extras)})
    plan_bytes = encode(p)
    plan_name, wrapper_name = (path.relative_to(ROOT).as_posix() for path in (plan_path, wrapper_path))
    wrapper = deepcopy(old_wrapper)
    wrapper.update(created_at=datetime.now(timezone.utc).isoformat(), upstream_plan=plan_name,
        upstream_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        source_sha256={name: sha(ROOT/name) for name in sorted(entry.REQUIRED)},
        scope='One zero-only run records selected numeric arrays; no dynamics or learning.')
    plan_path.write_bytes(plan_bytes)
    wrapper_path.write_bytes(encode(wrapper))
    output.mkdir()
    (output/'windows-entry.json').write_bytes(encode({'status': 'native-zero-prepared',
        'protocol_sha256': sha(plan_path), 'browser_visual_verified': False, 'real_observer_verified': False}))
    fake = deepcopy(project)
    fake['staged_training_request'].update(current_stage=p['stage'], current_protocol=plan_name,
        current_protocol_frozen=True, learning_running=False, simulation_running=False)
    fake['goal_execution']['native_file_observer_evaluation'].update(native_zero_plan=wrapper_name,
        native_zero_plan_sha256=sha(wrapper_path))
    original_read = entry.old.read
    def fake_read(path):
        return deepcopy(fake) if Path(path).resolve() == (ROOT/'project.json').resolve() else original_read(path)
    with patch.object(entry.old, 'read', side_effect=fake_read), entry.raw_bound(plan_name, wrapper_name):
        with entry.repair.repair_context(wrapper):
            coordinator = entry.logged.diagnostic.coordinator
            assert coordinator.verify_wrapper(wrapper_path, wrapper, fake) == sha(wrapper_path)
            assert coordinator.provenance.audit()['status'] == 'source-audit-passed-native-zero-only'
    result = {'status': 'frozen-raw-zero-awaiting-observation', 'created_at': datetime.now(timezone.utc).isoformat(),
        'arm': arm, 'run_id': run_id, 'seed': 1242, 'plan': plan_name, 'plan_sha256': sha(plan_path),
        'wrapper': wrapper_name, 'wrapper_sha256': sha(wrapper_path), 'output': output.relative_to(ROOT).as_posix(),
        'physical_steps': 0, 'learning_updates': 0, 'hardware_io': False,
        'automatic_retry': False, 'dynamic_execution_authorized': False}
    index.write_bytes(encode(result))
    task.update(current_stage=p['stage'], current_protocol=plan_name, current_protocol_frozen=True,
        next_gate=f'Observe and run the {arm} zero-only numeric capture.')
    goal['native_file_observer_evaluation'].update(native_zero_plan=wrapper_name,
        native_zero_plan_sha256=sha(wrapper_path), zero_coordinator_status=f'raw-{arm}-admitted-awaiting-view')
    goal['roll_parameter_resume_preflight'].update(status=f'raw-{arm}-zero-admitted-awaiting-view',
        current_raw_zero_plan=index.relative_to(ROOT).as_posix())
    save_project(project)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
