"""Create a separate observed comparison after the pre-sampling path error.

The original stopped evidence and controller remain unchanged.  The new
plans keep the same seed, factors, budgets, original guards, and independent
review.  The small wrapper only gives the new attempt distinct paths and binds
the tested confirmation helper into the frozen source inventory.
"""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import argparse
import shutil
from zoneinfo import ZoneInfo

import run_matched_roll_factor as matched

ROOT = matched.ROOT
ORIGINAL = ROOT / 'configs/experiments/roll_factor_matched_dynamic_s1242.json'
MASTER = ROOT / 'configs/experiments/roll_factor_matched_observed_s1242.json'
OUT = ROOT / 'records/reports/roll-matched-factor-observed-s1242'
STAGE = 'roll-factor-matched-observed-s1242'
FIXTURE = ROOT / 'records/reports/roll-matched-factor-dynamic-s1242/view-confirmation-path-check.json'
HELPER = ROOT / 'scripts/validation/native_view_confirmation.py'
ENTRY = Path(__file__).resolve()

# Reuse the previously frozen evaluator and guards without changing their files.
matched.MASTER, matched.OUT, matched.STAGE = MASTER, OUT, STAGE
matched.__file__ = str(ENTRY)


def freeze():
    if MASTER.exists() or OUT.exists():
        raise FileExistsError('The new comparison already exists')
    project, original, fixture = map(matched.read, (ROOT / 'project.json', ORIGINAL, FIXTURE))
    failure = project['goal_execution']['roll_factor_observer_failure']
    old_result = matched.read(ROOT / failure['result'])
    old_receipt = matched.read(ROOT / 'records/reports/roll-matched-factor-dynamic-s1242/nominal-a/execution-receipt.json')
    if (not matched.idle(project) or project['staged_training_request']['current_stage'] != 'roll-factor-observation-path-review'
            or failure['protocol_sha256'] != matched.sha(ORIGINAL)
            or failure['physical_substeps'] != 0 or failure['learning_updates'] != 0
            or old_result['status'] != 'stopped-review-required' or old_receipt['exit_code'] != 2
            or 'Viewing confirmation belongs to a different run or process' not in old_result['reason']
            or fixture['status'] != 'passed-path-and-view-confirmation-fixture'
            or not all(fixture['cases'].values()) or fixture['helper_sha256'] != matched.sha(HELPER)):
        raise ValueError('Prior stop or path fixture has not been verified')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plans, hashes = {}, {}
    OUT.mkdir()
    for role in matched.ROLES:
        prior_path = ROOT / original['plans'][role]
        if matched.sha(prior_path) != original['plan_sha256'][role]:
            raise ValueError('Original frozen arm changed: ' + role)
        plan = deepcopy(matched.read(prior_path))
        matched.baseline.verify_sources(plan)
        arm_out = OUT / role
        (arm_out / 'zero').mkdir(parents=True)
        for name in ('startup.npz', 'endpoint-contact.npz'):
            source = ROOT / plan['reference_zero'] / name
            shutil.copyfile(source, arm_out / 'zero' / name)
            if matched.sha(arm_out / 'zero' / name) != plan['reference_zero_files'][name]:
                raise ValueError('Copied zero-step evidence differs: ' + name)
        run_id = f'duck_roll-factor-observed_{role}_s1242_{stamp}'
        path = MASTER.with_name(f'roll_factor_matched_observed_s1242_{role}.json')
        if path.exists() or (ROOT / 'records/evaluations' / run_id).exists():
            raise FileExistsError('New plan or evaluation path exists')
        plan.update(stage=STAGE + '-' + role, run_id=run_id, guarded_output=matched.rel(arm_out),
                    page=matched.rel(arm_out / 'run-view.html'),
                    admission_record=matched.rel(arm_out / 'admission.json'))
        for source in (ENTRY, HELPER, FIXTURE, ROOT / failure['result']):
            plan['frozen_sha256'][matched.rel(source)] = matched.sha(source)
        matched.baseline.verify_sources(plan)
        matched.baseline.put_new(path, plan)
        digest = matched.sha(path)
        matched.baseline.put_new(arm_out / 'windows-entry.json',
                                 dict(status='native-observer-prepared', protocol_sha256=digest))
        matched.baseline.put_new(arm_out / 'freeze.json', dict(status='frozen-before-dynamics',
            run_id=run_id, protocol_sha256=digest, reference_zero=plan['reference_zero'],
            zero_files=plan['reference_zero_files'], controls=1550, worlds=128,
            learning_updates=0, hardware_io=False))
        plans[role], hashes[role] = matched.rel(path), digest
    master = dict(original, status='frozen-before-dynamics', stage=STAGE, plans=plans, plan_sha256=hashes,
        previous_stopped_protocol=matched.rel(ORIGINAL), path_fixture=matched.rel(FIXTURE),
        automatic_retry=False, automatic_extension=False)
    matched.baseline.put_new(MASTER, master)
    project = matched.read(ROOT / 'project.json')
    if not matched.idle(project) or project['staged_training_request']['current_stage'] != 'roll-factor-observation-path-review':
        raise RuntimeError('Project changed while freezing the new plan')
    project['staged_training_request'].update(current_stage=STAGE + '-ready', current_protocol=matched.rel(MASTER),
        current_protocol_frozen=True, next_gate='Observe and collect the first nominal arm; independently verify before the next.')
    project['goal_execution']['roll_factor_dynamic'] = dict(status='frozen-ready', protocol=matched.rel(MASTER),
        protocol_sha256=matched.sha(MASTER), next_arm=matched.ROLES[0], arms={}, learning_updates=0, hardware_io=False)
    matched.write(ROOT / 'project.json', project)
    (OUT / 'README.md').write_text('# 滚翻摩擦动态对照\n\n已冻结三组新记录，等待首组采集。\n', encoding='utf-8')
    print('OBSERVED_ROLL_FROZEN', matched.sha(MASTER), plans, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'child', 'verify'))
    parser.add_argument('--arm', choices=matched.ROLES)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    elif not args.arm:
        parser.error('--arm is required')
    elif args.action == 'run':
        raise SystemExit(matched.run(args.arm))
    elif args.action == 'child':
        raise SystemExit(matched.child(args.arm))
    else:
        matched.verify(args.arm)
