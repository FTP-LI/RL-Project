"""Prepare one separate recorded zero-step diagnosis after the saved stops."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import roll_fingerprint_logged_zero as logged

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'records/reports/roll-matched-factor-preflight-20261116'
OLD_PLAN = ROOT / 'configs/experiments/roll_factor_fp_diagnostic_zero_nominal_s1242_20261116.json'
OLD_WRAPPER = ROOT / 'configs/experiments/roll_factor_fp_diagnostic_native_nominal_s1242_20261116.json'
PLAN = ROOT / 'configs/experiments/roll_factor_fp_logged_zero_nominal_s1242_20261116.json'
WRAPPER = ROOT / 'configs/experiments/roll_factor_fp_logged_native_nominal_s1242_20261116.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def write_project(project):
    path = ROOT / 'project.json'
    temp = path.with_name('project.roll-zero-tmp.json')
    assert not temp.exists()
    temp.write_bytes(encode(project))
    os.replace(temp, path)


def main():
    index = REPORT / 'logged-diagnostic-plan.json'
    assert not any(path.exists() for path in (PLAN, WRAPPER, index))
    project = read(ROOT / 'project.json')
    task, goal = project['staged_training_request'], project['goal_execution']
    assert goal['status'] == 'active' and goal['pause_requested_by_user'] is False
    assert task['pause_requested'] is False and not task['learning_running']
    assert not task['simulation_running'] and not task['current_protocol_frozen']
    task.update(current_stage='roll-parameter-logged-diagnostic-preparation',
                next_gate='Prepare one separately frozen zero-only diagnostic recording observer refusal and actuator values.')
    goal['roll_parameter_resume_preflight'].update(status='logged-zero-preparation',
        next_gate='One separately frozen zero-only diagnostic; preserve prior stops, no dynamics or learning.')
    write_project(project)

    stop = read(REPORT / 'observer-stop-readback.json')
    fixture = read(REPORT / 'logged-wrapper-fixture.json')
    observer = read(REPORT / 'observer-health-repro/result.json')
    assert stop['historical_rejection_cause_known'] is False and stop['physics_substeps'] == 0
    assert fixture['status'] == 'passed-cpu-logged-wrapper-fixture-only'
    assert observer['status'] == 'passed-no-refusals' and observer['checks'] == 50
    for name, digest in fixture['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    logged.diagnostic.diagnostic_evidence()
    base, old_wrapper = read(OLD_PLAN), read(OLD_WRAPPER)
    assert base['seed'] == 1242 and base['factor_arm'] == 'nominal'
    assert old_wrapper['upstream_plan_sha256'] == sha(OLD_PLAN)
    failed = ROOT / 'records/reports/duck_roll-factor_nominal_s1242_20261116T133748'
    previous_result = read(failed / 'native-zero-run/result.json')
    assert previous_result['physics_substeps'] == 0 and previous_result['learning_updates'] == 0
    assert sha(failed / 'native-zero-execution.log') == stop['log_sha256']

    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = 'duck_roll-factor_nominal_s1242_' + stamp
    output = ROOT / 'records/reports' / run_id
    assert not output.exists()
    plan = deepcopy(base)
    plan.update(run_id=run_id, guarded_output=output.relative_to(ROOT).as_posix(),
                page=f'records/{run_id}-live.html',
                admission_record=f'records/reports/{run_id}/admission.json')
    extras = logged.REQUIRED | {
        'scripts/validation/prepare_roll_fingerprint_logged_zero.py',
        'scripts/validation/verify_roll_fingerprint_logged_zero.py',
        'records/reports/roll-matched-factor-preflight-20261116/logged-wrapper-fixture.json',
        OLD_PLAN.relative_to(ROOT).as_posix(), OLD_WRAPPER.relative_to(ROOT).as_posix(),
        (failed / 'native-zero-execution.log').relative_to(ROOT).as_posix(),
        (failed / 'native-zero-run/result.json').relative_to(ROOT).as_posix(),
    }
    plan['frozen_sha256'].update({name: sha(ROOT / name) for name in sorted(extras)})
    plan_bytes = encode(plan)
    plan_name, wrapper_name = (p.relative_to(ROOT).as_posix() for p in (PLAN, WRAPPER))
    wrapper = deepcopy(old_wrapper)
    wrapper.update(created_at=datetime.now(timezone.utc).isoformat(), upstream_plan=plan_name,
                   upstream_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                   source_sha256={name: sha(ROOT / name) for name in sorted(logged.REQUIRED)},
                   scope='One separate zero-only diagnostic records guard refusals and actuator state; no dynamics or learning.')
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
    original_read = logged.old.read
    def fake_read(path):
        return deepcopy(fake) if Path(path).resolve() == (ROOT / 'project.json').resolve() else original_read(path)
    with patch.object(logged.old, 'read', side_effect=fake_read), logged.logged_bound(plan_name, wrapper_name):
        with logged.repair.repair_context(wrapper):
            assert logged.diagnostic.coordinator.verify_wrapper(WRAPPER, wrapper, fake) == sha(WRAPPER)
            assert logged.diagnostic.coordinator.provenance.audit()['status'] == 'source-audit-passed-native-zero-only'
    result = {'status': 'frozen-separate-diagnostic-awaiting-observer-and-registration',
              'created_at': datetime.now(timezone.utc).isoformat(), 'run_id': run_id, 'seed': 1242,
              'seed_rationale': 'Same seed isolates recording changes; new plan and process, no restart.',
              'plan': plan_name, 'plan_sha256': sha(PLAN), 'wrapper': wrapper_name,
              'wrapper_sha256': sha(WRAPPER), 'output': output.relative_to(ROOT).as_posix(),
              'previous_stop': stop['run_id'], 'physical_steps': 0, 'learning_updates': 0,
              'hardware_io': False, 'automatic_retry': False, 'dynamic_execution_authorized': False}
    index.write_bytes(encode(result))
    project['staged_training_request'].update(current_stage=plan['stage'], current_protocol=plan_name,
        current_protocol_frozen=True, next_gate='Observe and run the one separately frozen zero-only diagnostic.')
    goal['native_file_observer_evaluation'].update(native_zero_plan=wrapper_name,
        native_zero_plan_sha256=sha(WRAPPER), zero_coordinator_status='logged-diagnostic-admitted-awaiting-view')
    goal['roll_parameter_resume_preflight'].update(status='logged-zero-admitted-awaiting-view',
        logged_diagnostic_plan=index.relative_to(ROOT).as_posix())
    write_project(project)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
