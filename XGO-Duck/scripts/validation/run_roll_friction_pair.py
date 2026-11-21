"""Freeze and run the equal-sample rolling experiment after capacity selection."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_roll_ppo_capacity_live as live

trial, bounded = live.trial, live.bounded
ROOT, P, sha, read, put, rel, require = trial.ROOT, trial.P, trial.sha, trial.read, trial.put, trial.rel, trial.require
ENTRY = Path(__file__).resolve()
PLAN = ROOT / 'configs/experiments/roll_friction_paired_training.json'
REPORT = ROOT / 'records/reports/roll-friction-paired-training'
STAGE = 'roll-friction-paired-training'
ROLES = {'uniform': 'official-distribution', 'low-friction': 'low-friction-enriched'}


def command(role, action):
    return [sys.executable, '-B', '-u', str(ENTRY), action, role]


def setup_path(role):
    return REPORT / 'reset-admission' / role


def freeze():
    project = read(P)
    task, goal = project['staged_training_request'], project['goal_execution']
    capacity = goal['roll_ppo_capacity']
    require(trial.allowed(project, trial.STAGE+'-complete') and capacity['status'] == 'verified-selection'
        and not task['simulation_running'] and not task['learning_running'], 'Capacity selection is pending')
    capacity_plan, selection = ROOT / capacity['protocol'], ROOT / capacity['selection']
    inherited = read(capacity_plan)
    n = read(selection)['provisional_worlds']
    require(sha(capacity_plan) == capacity['sha256'] and n == capacity['chosen_environments'], 'Capacity selection differs')
    draft = read(trial.DRAFT)
    settings, stop = draft['common_training'], draft['stop']
    updates = settings['learning_updates_per_arm_limit']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    arms = {role: dict(run_id=f'duck_roll_friction_{role}_e{n}_s{draft["training_seed"]}_{stamp}',
                environments=n, friction_role=friction, updates=updates,
                transitions=n*updates*settings['rollout_controls'],
                extra_checkpoints=settings['checkpoints_at_complete_updates'][:-1]) for role, friction in ROLES.items()}
    fixed_trace = ROOT / 'records/evaluations/duck_roll_action-memory_control_s1003_20261103T004639/forward-roll-trace.npz'
    reference = read(ROOT / 'configs/experiments/roll_top_contact/control.json')['drift_probe_trace_sha256']
    require(sha(fixed_trace) == reference, 'Historical fixed actor probe differs')
    check_steps = [336, 686, 1036]
    clip_starts = [0, 700]
    require(all(s % 350 == 336 for s in check_steps) and clip_starts[-1]+350 <= settings['rollout_controls']*updates,
            'Action clocks exceed the fixed training budget')
    dependencies = [ENTRY, ENTRY.with_name('check_roll_friction_paired_reset.py'),
                    ENTRY.with_name('export_roll_friction_pair.py'), ROOT / 'scripts/training/roll_friction_paired.py',
                    ROOT / 'scripts/training/walk_backward_export.py', ROOT / 'scripts/training/walk_backward_ppo.py',
                    capacity_plan, selection, fixed_trace]
    sources = inherited['source_sha256'] | {rel(file): sha(file) for file in dependencies}
    plan = dict(status='frozen-paired-reset-admission', scope='formal-paired-training', stage=STAGE,
                arms=arms, supervision=inherited['supervision'], installed_sha256=inherited['installed_sha256'],
                source_sha256=sources, hardware_io=False, automatic_retry=False, automatic_extension=False,
                supervisor_sha256=inherited['supervisor_sha256'],
                parent_exit_guard_sha256=inherited['parent_exit_guard_sha256'],
                supervisor_workers={arm['run_id']: command(role, 'worker') for role, arm in arms.items()},
                recipe=inherited['recipe'] | {'seed': draft['training_seed'], 'late_monitor_updates': []},
                training_checks=dict(drift_trace=rel(fixed_trace), drift_trace_sha256=reference,
                    drift_frames=[1,16,31,46,61,76,151,301,450], max_probe_rms_rad=stop['fixed_probe_max_rms_rad'],
                    max_probe_abs_rad=stop['fixed_probe_max_abs_rad'], late_control_steps=check_steps,
                    clip_start_controls=clip_starts),
                evaluation=dict(development_seeds=draft['development_evaluation_seeds'], conditions=draft['development_evaluation'],
                                reserved_seed=draft['reserved_evaluation_seed'], reserved_seed_used=False))
    require(not PLAN.exists() and not REPORT.exists(), 'Preserve the frozen plan and existing reports')
    (REPORT / 'reset-admission').mkdir(parents=True)
    put(PLAN, plan)
    goal['roll_friction_paired'] = dict(status='reset-pending', protocol=rel(PLAN), sha256=sha(PLAN),
                                    report=rel(REPORT), next_role='uniform', reviews={})
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
                next_gate='Read back both full learners at zero physics, then run the uniform and enriched arms.')
    trial.write(P, project)
    print('ROLL_FRICTION_PAIR_FROZEN', n, sha(PLAN), flush=True)


def checked(states, *, role=None, idle=True):
    project, plan = read(P), read(PLAN)
    goal, task = project['goal_execution']['roll_friction_paired'], project['staged_training_request']
    require(trial.allowed(project, STAGE) and task['current_protocol'] == rel(PLAN)
        and task['current_protocol_frozen'] and goal['sha256'] == sha(PLAN) and goal['status'] in states
        and (role is None or goal['next_role'] == role)
        and (not idle or not task['learning_running'] and not task['simulation_running']), 'Pair is not ready for this command')
    for base, name in ((ROOT, 'source_sha256'), (Path('/'), 'installed_sha256')):
        require(all(sha(base/file) == digest for file, digest in plan[name].items()), 'Paired source changed')
    return project, plan


def probe(role):
    from check_roll_friction_paired_reset import capture

    with trial.supervisor.TRAIN_LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, plan = checked({'reset-pending'})
        destination = setup_path(role)
        destination.mkdir()
        peak = capture(plan, plan['arms'][role], destination)
        put(destination / 'result.json', dict(peak_gpu_used_mib=peak, role=role, plan_sha256=sha(PLAN),
            evidence_sha256=sha(destination / 'zero-step.json'), physics_steps=0, learning_updates=0))


def admit():
    from check_roll_friction_paired_reset import compare

    project, plan = checked({'reset-pending'})
    paths = {}
    reset_hashes = {}
    for role in ROLES:
        directory = setup_path(role)
        outcome, evidence = (read(directory/name) for name in ('result.json', 'zero-step.json'))
        arr = directory / 'startup-inputs.npz'
        require(outcome['role'] == role and outcome['plan_sha256'] == sha(PLAN)
            and outcome['evidence_sha256'] == sha(directory/'zero-step.json') and evidence['startup_inputs_sha256']==sha(arr)
            and outcome['physics_steps']==outcome['learning_updates']==evidence['controls']==evidence['physics_substeps']==0
            and evidence['matched_actor_critic_and_adam'], 'Paired learner reset is incomplete')
        paths[role] = arr
        reset_hashes[role] = sha(arr)
    n = plan['arms']['uniform']['environments']
    proof = compare(paths['uniform'], paths['low-friction'], n)
    approval = REPORT / 'reset-admission/paired-check.json'
    put(approval, dict(proof, status='passed-paired-reset-only', input_sha256=reset_hashes, physical_steps=0,
                        learning_updates=0, plan_sha256=sha(PLAN)))
    project['goal_execution']['roll_friction_paired'].update(status='ready', reset=rel(approval), reset_sha256=sha(approval))
    trial.write(P, project)
    print('ROLL_FRICTION_RESET_VERIFIED', n, proof, flush=True)


def checked_reset(project, role):
    pair = project['goal_execution']['roll_friction_paired']
    receipt = ROOT / pair['reset']
    saved = read(receipt)
    path = setup_path(role) / 'startup-inputs.npz'
    require(sha(receipt) == pair['reset_sha256'] and saved['status'] == 'passed-paired-reset-only'
            and saved['plan_sha256'] == sha(PLAN) and saved['input_sha256'][role] == sha(path),
            'Reset admission proof changed')
    return path


def confirm(role, path):
    project, plan = checked({'ready'}, role=role)
    checked_reset(project, role)
    # Use the same observer identity, source, screenshot and freshness check
    # that was already used for the isolated capacity runs.
    with (patch.object(trial, 'PLAN', PLAN), patch.object(trial, 'REPORT', REPORT),
          patch.object(trial, 'checked', lambda _: (project, plan, plan['arms'][role]))):
        trial.confirmed(role, path)


def supervise(role):
    project, plan = checked({'ready'}, role=role)
    checked_reset(project, role)
    arm = plan['arms'][role]
    report = REPORT / role
    report.mkdir()
    os.environ.update(MUJOCO_GL='egl', GALLIUM_DRIVER='d3d12', MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
    timing = {key: value for key, value in plan['supervision'].items() if key != 'viewing_seconds'}
    receipt = trial.supervisor.run_supervised(command(role, 'worker'), cwd=ROOT,
        log_path=report/'execution.log', receipt_path=report/'supervisor-result.json', source=trial.source_for(arm),
        run_id=arm['run_id'], protocol=PLAN, protocol_sha256=sha(PLAN), stage=STAGE, **timing)
    passed = receipt['status'] == 'child-exited' and receipt['exit_code'] == 0
    if not passed:
        print('ROLL_FRICTION_PAIR_STOP', role, receipt['status'], receipt['exit_code'], flush=True)
        return 2
    final = trial.finalize_view(report, arm['run_id'], read(report/'entry-result.json'))
    print('ROLL_FRICTION_PAIR_COLLECTED', role, final['status'], flush=True)
    return 0 if final['status'] == 'saved' else 2


def worker(role):
    project, plan = checked({'ready'}, role=role)
    reset = checked_reset(project, role)
    arm, report = plan['arms'][role], REPORT / role
    pid, digest = os.getpid(), sha(PLAN)
    argv = Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0')
    require(len(argv) >= 5 and Path(os.fsdecode(argv[-4])).resolve() == ENTRY
            and argv[-3:-1] == [b'supervise', os.fsencode(role)], 'Use the authorized paired supervisor')
    log = (report/'execution.log').stat()
    require(all((log.st_dev, log.st_ino) == (os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in (1, 2)),
            'Paired worker requires the command transcript')
    active, monitor = False, None
    record = ROOT / 'records/experiments' / arm['run_id']
    outcome = dict(run_id=arm['run_id'], plan_sha256=digest, status='stopped-review-required', hardware_io=False)

    def authorized():
        p = bounded.read_status(P, reader=read)
        task, pair = p['staged_training_request'], p['goal_execution']['roll_friction_paired']
        return (trial.allowed(p, STAGE) and task['learning_running'] is active and task['simulation_running'] is active
                and pair['next_role'] == role and (not active or pair.get('worker_pid') == pid)
                and trial.supervisor._authorized(PLAN, digest, STAGE, arm['run_id'], command(role, 'worker'),
                                                 before_start=False))

    def update_project(**values):
        p = bounded.read_status(P, reader=read)
        p['staged_training_request'].update(simulation_running=active, learning_running=active)
        p['goal_execution']['roll_friction_paired'].update(worker_pid=pid, **values)
        trial.write(P, p)

    label = {'uniform': '均匀摩擦训练对照组', 'low-friction': '低摩擦加权训练候选组'}[role]
    view = bounded.BoundedNativePPO(arm['run_id'], report, lambda: authorized() if monitor is None else monitor.check(),
        environments=arm['environments'], planned_updates=arm['updates'], purpose=label)

    def terminate(_number, _frame):
        raise KeyboardInterrupt('Supervised paired worker terminated')

    try:
        signal.signal(signal.SIGTERM, terminate)
        view.start()
        print('WAITING_FOR_CURRENT_VIEW', view.view, flush=True)
        until = time.monotonic()+plan['supervision']['viewing_seconds']
        while not trial.viewing_confirmed(report/'viewing-confirmation.json', view, digest):
            view.require_active()
            require(time.monotonic() < until, 'The current VS Code view was not confirmed')
            time.sleep(.2)
        require(authorized(), 'Approval changed before training')
        active = True
        update_project(status='running')
        monitor = live.SynchronousAuthorization(authorized).start()
        view.require_active()
        record.mkdir()
        put(record/'protocol-arm.json', dict(arm, plan_sha256=digest, role=role, scope=plan['scope']))
        from roll_friction_paired import collect
        result = collect(plan, arm, record, view, reset)
        outcome.update(status=result['status'], reason=None, record=rel(record), online_result_sha256=sha(record/'online-result.json'))
    except BaseException as error:
        outcome['reason'] = type(error).__name__+': '+str(error)
        traceback.print_exc()
    finally:
        if monitor is not None:
            monitor.close()
            outcome['authorization_monitor'] = monitor.summary()
        try:
            view.close(outcome)
        finally:
            if active:
                active = False
                update_project(status='review-required', last_result=rel(report/'entry-result.json'))
            put(report/'entry-result.json', outcome)
    return 0 if outcome['status'] == 'collected-review-required' else 2


def verify(role):
    from verify_roll_ppo_capacity import independent

    project, plan = checked({'review-required'}, role=role)
    reset = checked_reset(project, role)
    arm, report = plan['arms'][role], REPORT / role
    result = independent(plan, PLAN, arm, report)
    raw = ROOT / 'records/experiments' / arm['run_id']
    checks = [read(raw/f'late-roll-control-{n}.json') for n in plan['training_checks']['late_control_steps']]
    drifts = [json.loads(line) for line in (raw/'policy-drift.jsonl').read_text(encoding='utf-8').splitlines()]
    wanted = plan['training_checks']['clip_start_controls']
    recorded = [int(Path(clip['path']).stem.removeprefix('clip-step-')) for clip in result['clips'] if clip['complete_episode']]
    require(recorded == wanted and all(x['passed'] for x in checks)
        and len(drifts) == arm['updates'] and all(x['update']==n and x['within_limits'] for n,x in enumerate(drifts, 1)),
        'Paired roll checks or full-action recordings are incomplete')
    for update in arm['extra_checkpoints']:
        require((raw/f'checkpoint-update-{update}.pt').is_file(), 'Intermediate checkpoint missing')
    result.update(status='passed-paired-training-readback', role=role, stabilized_roll_checks=checks,
        fixed_probe_max_rms_rad=max(row['rms_rad'] for row in drifts),
        fixed_probe_max_abs_rad=max(row['maximum_rad'] for row in drifts),
        reset_input_sha256=sha(reset), action_evaluation_done=False)
    path = report / 'independent.json'
    put(path, result)
    pair = project['goal_execution']['roll_friction_paired']
    pair['reviews'][role] = rel(path)
    pair.update(status='ready' if role=='uniform' else 'evaluation-pending', next_role='low-friction' if role=='uniform' else None)
    trial.write(P, project)
    print('ROLL_FRICTION_PAIR_VERIFIED', role, result['transitions'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','probe','admit','supervise','worker','confirm','verify'))
    parser.add_argument('role', nargs='?', choices=ROLES)
    parser.add_argument('--screenshot', type=Path)
    args = parser.parse_args()
    if args.action in ('freeze','admit'):
        return {'freeze': freeze, 'admit': admit}[args.action]()
    if args.role is None:
        parser.error('Training arm is required')
    if args.action == 'confirm':
        if args.screenshot is None:
            parser.error('Screenshot path is required')
        return confirm(args.role, args.screenshot)
    return {'probe': probe, 'supervise': supervise, 'worker': worker, 'verify': verify}[args.action](args.role)


if __name__ == '__main__':
    with patch.object(trial.supervisor, '_read', bounded.read_status):
        sys.exit(main())
