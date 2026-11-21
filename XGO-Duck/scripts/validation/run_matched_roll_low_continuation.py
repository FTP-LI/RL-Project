"""Collect the one outstanding friction arm without replaying completed nominal data.

Only identity, output paths and provenance change. Keep every dynamics,
scoring, observation, resource and stop setting from the original frozen arm.
"""
from datetime import datetime
from pathlib import Path
import argparse
import shutil
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_matched_roll_factor_report_capture as original

matched = original.matched
ROOT = original.ROOT
ENTRY = Path(__file__).resolve()
MASTER = ROOT / 'configs/experiments/roll_factor_low_continuation_s1242/master.json'
OUT = ROOT / 'records/reports/roll-matched-factor-low-continuation-s1242'
STAGE = 'roll-factor-low-continuation-s1242'
ROLE = 'friction-low'


def freeze():
    project_path = ROOT / 'project.json'
    project = matched.read(project_path)
    task, goal = project['staged_training_request'], project['goal_execution']
    prior = goal['roll_factor_dynamic']
    if (not matched.idle(project) or task['current_stage'] != 'roll-factor-low-continuation-preflight'
            or prior['next_arm'] != ROLE or prior['protocol'] != matched.rel(original.MASTER)
            or prior['protocol_sha256'] != matched.sha(original.MASTER)):
        raise ValueError('Prior comparison or current idle authorization differs')
    master = matched.read(original.MASTER)
    prior_plan = ROOT / master['plans'][ROLE]
    plan = matched.read(prior_plan)
    prior_out = original.OUT / ROLE
    stopped, receipt, log, health = (prior_out / name for name in (
        'result.json', 'execution-receipt.json', 'execution.log', 'native-view/native-health.json'))
    result, execution, observation = map(matched.read, (stopped, receipt, health))
    digest = matched.sha(prior_plan)
    if (digest != master['plan_sha256'][ROLE] or result['protocol_sha256'] != digest
            or result['status'] != 'stopped-review-required' or result['reason'] != 'RuntimeError: Current observer unavailable: None'
            or result['learning_updates'] != 0 or result['hardware_io'] is not False
            or execution['status'] != 'child-failed' or execution['exit_code'] != 2
            or execution['log_sha256'] != matched.sha(log) or observation['health'] != 'stopped'
            or observation['run_id'] != plan['run_id'] or (prior_out / 'viewing-confirmation.json').exists()
            or (ROOT / 'records/evaluations' / plan['run_id']).exists()
            or 'WAITING_FOR_CURRENT_VIEW' not in log.read_text(encoding='utf-8')):
        raise ValueError('The prior stop must be before observer admission and sampling')
    evidence = [ENTRY, original.MASTER, prior_plan, stopped, receipt, log, health]
    for role in ('nominal-a', 'nominal-b'):
        registered = prior['arms'][role]
        path = ROOT / registered['analysis']
        arm = matched.read(path)
        if (registered['status'] != 'independently-verified' or arm['run_id'] != registered['run_id']
                or arm['role'] != role or arm['protocol_sha256'] != master['plan_sha256'][role]
                or arm['status'] != 'completed-baseline-independently-verified' or not arm['passed_development_gate']
                or arm['controls'] != 1550 or arm['control_transitions'] != 198400
                or arm['learning_updates'] != 0 or arm['hardware_io'] is not False):
            raise ValueError('Nominal reference is incomplete: ' + role)
        evidence.extend((path, ROOT / master['plans'][role]))
    out = OUT / ROLE
    if MASTER.parent.exists() or OUT.exists():
        raise FileExistsError('Continuation already exists; preserve it for review')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_roll-friction-low-continuation_s1242_{stamp}'
    if (ROOT / 'records/evaluations' / run_id).exists():
        raise FileExistsError('New evaluation path exists')
    plan.update(stage=STAGE + '-' + ROLE, run_id=run_id, guarded_output=matched.rel(out),
                page=matched.rel(out / 'run-view.html'), admission_record=matched.rel(out / 'admission.json'))
    plan['frozen_sha256'].update({matched.rel(p): matched.sha(p) for p in evidence})
    matched.baseline.verify_sources(plan)
    for name in ('startup.npz', 'endpoint-contact.npz'):
        source = ROOT / plan['reference_zero'] / name
        if matched.sha(source) != plan['reference_zero_files'][name]:
            raise ValueError('Source zero-step evidence differs: ' + name)
    MASTER.parent.mkdir()
    (out / 'zero').mkdir(parents=True)
    for name in ('startup.npz', 'endpoint-contact.npz'):
        shutil.copyfile(ROOT / plan['reference_zero'] / name, out / 'zero' / name)
        if matched.sha(out / 'zero' / name) != plan['reference_zero_files'][name]:
            raise ValueError('Copied zero-step evidence differs: ' + name)
    path = MASTER.with_name(f'{ROLE}.json')
    matched.baseline.put_new(path, plan)
    plan_sha = matched.sha(path)
    matched.baseline.put_new(out / 'windows-entry.json', dict(status='native-observer-prepared', protocol_sha256=plan_sha))
    matched.baseline.put_new(out / 'freeze.json', dict(status='frozen-before-dynamics', run_id=run_id,
        protocol_sha256=plan_sha, source_plan=matched.rel(prior_plan), stopped_result=matched.rel(stopped),
        stopped_receipt=matched.rel(receipt), prior_physical_substeps=0, reference_zero=plan['reference_zero'],
        zero_files=plan['reference_zero_files'], controls=1550, worlds=128, learning_updates=0, hardware_io=False))
    master.update(stage=STAGE, arms=[ROLE], plans={ROLE: matched.rel(path)}, plan_sha256={ROLE: plan_sha},
                  previous_stopped_protocol=matched.rel(original.MASTER), maximum_total_transitions=198400,
                  sequence_rule='Collect only the outstanding low-friction arm; independently verify it against the two dated nominal references.')
    matched.baseline.put_new(MASTER, master)
    if matched.read(project_path) != project:
        raise RuntimeError('Project changed during freeze; no run admitted')
    prior['arms'][ROLE].update(status='stopped-before-observation', result=matched.rel(stopped), receipt=matched.rel(receipt))
    goal['roll_factor_report_capture_history'] = prior
    goal['roll_factor_dynamic'] = dict(status='frozen-ready', protocol=matched.rel(MASTER), protocol_sha256=matched.sha(MASTER),
        next_arm=ROLE, arms={}, learning_updates=0, hardware_io=False)
    goal['resume_event_20261120']['status'] = 'continuation-frozen-awaiting-native-view'
    task.update(current_stage=STAGE + '-ready', current_protocol=matched.rel(MASTER), current_protocol_frozen=True,
                next_gate='Verify VS Code live observation, then collect and independently score the low-friction arm.')
    matched.write(project_path, project)
    print('ROLL_LOW_CONTINUATION_FROZEN', run_id, plan_sha, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'child', 'verify'))
    parser.add_argument('--arm', choices=(ROLE,), default=ROLE)
    action = parser.parse_args().action
    matched.MASTER, matched.OUT, matched.STAGE, matched.__file__ = MASTER, OUT, STAGE, str(ENTRY)
    matched.ROLES = (ROLE,)
    if action == 'freeze':
        freeze()
    elif action == 'run':
        raise SystemExit(matched.run(ROLE))
    elif action == 'child':
        _, _, plan, out = matched.checked(ROLE)
        import roll_factor_zero_recorded_hook as hook
        with patch.object(hook, 'capture', original.report_capture(ROOT, out, plan['run_id'], hook.capture)):
            raise SystemExit(matched.child(ROLE))
    else:
        matched.verify(ROLE)
