"""Run one frozen, observed arm of the official roll parameter comparison.

The previously tested official controller and guards remain byte-identical.
Three arms use the same seed and recorded zero-step inputs: two nominal runs
and one low-friction run. Solver internals are recorded and may differ; this
is matched-input comparison, not a claim of identical full physical state.
Each arm must pass independent review before the next is admitted. There is
no learning, hardware access, automatic retry, or automatic extension.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
sys.path.insert(0, str(ROOT / 'scripts/validation'))

import guarded_factor_zero_entry as zero_entry
import native_nominal_baseline as baseline
from native_roll_observation_bounded import NativeRollObservationBounded
from observed_getup_entry import viewing_confirmed
from record_io import write
from recorded_process import run_recorded

MASTER = ROOT / 'configs/experiments/roll_factor_matched_dynamic_s1242.json'
OUT = ROOT / 'records/reports/roll-matched-factor-dynamic-s1242'
PROOF = ROOT / 'records/reports/roll-matched-factor-preflight-20261116/friction-raw-zero-comparison.json'
NOMINAL = ROOT / 'configs/experiments/roll_factor_fp_raw_zero_nominal_s1242_20261116.json'
LOW = ROOT / 'configs/experiments/roll_factor_fp_raw_zero_friction-low_s1242_20261116.json'
ROLES = ('nominal-a', 'nominal-b', 'friction-low')
STAGE = 'roll-factor-matched-dynamic-s1242'
EXTRA = (
    'scripts/validation/run_matched_roll_factor.py',
    'scripts/validation/native_nominal_baseline.py',
    'scripts/validation/verify_native_nominal_baseline.py',
    'scripts/validation/guarded_roll_collector.py',
    'scripts/validation/guarded_roll_loop.py',
    'scripts/validation/roll_screen_stop_guard.py',
    'scripts/validation/factor_scene_guard_adapter.py',
    'scripts/validation/roll_factor_zero_recorded_hook.py',
    'scripts/validation/roll_scene_fingerprint_hook.py',
    'scripts/validation/run_local_getup_ppo.py',
    'scripts/validation/local_getup_scale.py',
    'scripts/training/native_roll_observation_bounded.py',
    'scripts/training/native_run_observer_bounded.py',
    'scripts/training/native_run_observer.py',
    'scripts/training/native_observer_guard.py',
    'scripts/training/recorded_process.py',
    'scripts/training/parent_exit_exec.py',
    'records/reports/roll-matched-factor-preflight-20261116/raw-nominal-zero-verification.json',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return zero_entry.sha(path)


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def idle(project):
    task, goal = project['staged_training_request'], project['goal_execution']
    return (goal.get('status') == 'active' and not goal.get('pause_requested_by_user', False)
            and not task['pause_requested'] and not task['simulation_running'] and not task['learning_running'])


def freeze():
    if MASTER.exists() or OUT.exists():
        raise FileExistsError('The existing plan or report must not be replaced')
    project, proof = read(ROOT / 'project.json'), read(PROOF)
    if (not idle(project) or project['staged_training_request']['current_stage'] != 'roll-parameter-dynamic-admission-review'
            or proof['status'] != 'raw-comparison-completed-dynamic-review-required'
            or proof['seed'] != 1242 or proof['worlds'] != 128 or proof['rng_equal'] is not True
            or not all(proof['startup_and_interface_arrays_equal'].values())
            or proof['same_factor_repeat_showed_same_prior_fingerprint_difference_names'] is not True):
        raise ValueError('The existing matched-input evidence is not ready')
    for name, expected in proof['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Existing zero-step source differs: ' + name)
    io = read(ROOT / 'records/reports/getup-monitored-pair-holdout-s4267/native-io-diagnosis.json')
    life = read(ROOT / 'records/reports/getup-monitored-pair-holdout-s4267/native-observer-lifecycle.json')
    if (io['status'] != 'passed-bounded-observer-tests' or life['status'] != 'passed-actual-observer-lifecycle'
            or io['adapter_sha256'] != sha(ROOT / 'scripts/training/native_run_observer_bounded.py')
            or life['bridge_sha256'] != sha(ROOT / 'scripts/training/native_roll_observation_bounded.py')):
        raise ValueError('The native observer checks do not match current sources')
    previous_plans = {'nominal': read(NOMINAL), 'friction-low': read(LOW)}
    for kind, prior in previous_plans.items():
        if (prior['seed'] != 1242 or prior['worlds'] != 128 or prior['factor_arm'] != kind
                or prior['maximum_controls'] != 1550 or prior['dynamics_authorized'] is not False
                or prior['heldout_used'] is not False or prior['learning_updates'] != 0):
            raise ValueError('Prior zero-step protocol scope differs: ' + kind)
        baseline.verify_sources(prior)
    import numpy as np
    from selective_friction_guard import compare_reset_fields, compare_static
    nominal_zero = ROOT / 'records/reports' / proof['nominal_run'] / 'zero'
    low_zero = ROOT / 'records/reports' / proof['low_friction_run'] / 'zero'
    matched = {}
    for name in ('startup.npz', 'reset-0-interface.npz', 'endpoint-contact.npz', 'actuator-observed-start.npz'):
        with np.load(nominal_zero / name, allow_pickle=False) as a, np.load(low_zero / name, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError('Prior zero-step field sets differ: ' + name)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=name + ':' + key)
            matched[name] = sorted(a.files)
    with np.load(nominal_zero / 'scene-static-fields.npz', allow_pickle=False) as a, \
         np.load(low_zero / 'scene-static-fields.npz', allow_pickle=False) as b:
        static_check = compare_static({key: b[key] for key in b.files}, {key: a[key] for key in a.files})
    feet = read(low_zero / 'factor-zero-wiring.json')['foot_geom_ids']
    with np.load(nominal_zero / 'expanded-fields-before-physics.npz', allow_pickle=False) as a, \
         np.load(low_zero / 'expanded-fields-before-physics.npz', allow_pickle=False) as b:
        friction_check = compare_reset_fields({key: b[key] for key in b.files}, {key: a[key] for key in a.files},
                                              foot_geom_ids=feet, friction=.8, worlds=128)
    nominal_fp, low_fp = (read(path / 'start-fingerprints.json') for path in (nominal_zero, low_zero))
    if nominal_fp['rng_digest'] != low_fp['rng_digest']:
        raise ValueError('Prior zero-step RNG digests differ')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plans, hashes, references = {}, {}, {}
    OUT.mkdir()
    write(OUT / 'zero-input-readback.json', dict(status='passed-independent-zero-input-readback',
        source_comparison_sha256=sha(PROOF), exact_input_fields=matched, static_model=static_check,
        expanded_friction=friction_check, rng_exact=True, physical_substeps=0, learning_updates=0,
        scope='Read-only evidence; does not imply full internal solver state is identical.'))
    for role in ROLES:
        kind = 'friction-low' if role == 'friction-low' else 'nominal'
        prior = previous_plans[kind]
        source_run = proof['low_friction_run'] if kind == 'friction-low' else proof['nominal_run']
        zero = ROOT / 'records/reports' / source_run / 'zero'
        arm_out = OUT / role
        arm_out.mkdir()
        (arm_out / 'zero').mkdir()
        for name in ('startup.npz', 'endpoint-contact.npz'):
            shutil.copyfile(zero / name, arm_out / 'zero' / name)
            if sha(arm_out / 'zero' / name) != sha(zero / name):
                raise RuntimeError('Zero-step copy differs: ' + name)
        run_id = f'duck_roll-factor-matched_{role}_s1242_{stamp}'
        plan_path = MASTER.with_name(f'roll_factor_matched_dynamic_s1242_{role}.json')
        if plan_path.exists() or (ROOT / 'records/evaluations' / run_id).exists():
            raise FileExistsError('A new run must not overwrite an existing record')
        p = deepcopy(prior)
        p.update(stage=STAGE + '-' + role, run_id=run_id, guarded_output=rel(arm_out),
                 page=rel(arm_out / 'run-view.html'), admission_record=rel(arm_out / 'admission.json'),
                 dynamics_authorized=True, comparison_role=role,
                 comparison_scope='Same seed and checked reset inputs; solver internals need not be bitwise identical.',
                 reference_zero=rel(zero), reference_zero_files={name: sha(zero / name) for name in
                   ('startup.npz', 'reset-0-interface.npz', 'endpoint-contact.npz', 'actuator-observed-start.npz',
                    'expanded-fields-before-physics.npz', 'scene-static-fields.npz', 'start-fingerprints.json')},
                 observer='bounded-source-read; existing child and heartbeat stop gate retained',
                 max_gpu_fraction=.85, no_automatic_retry=True, heldout_used=False)
        p['frozen_sha256'].update({name: sha(ROOT / name) for name in EXTRA})
        p['frozen_sha256'][rel(PROOF)] = sha(PROOF)
        p['frozen_sha256'][rel(OUT / 'zero-input-readback.json')] = sha(OUT / 'zero-input-readback.json')
        p['frozen_sha256'][rel(NOMINAL if kind == 'nominal' else LOW)] = sha(NOMINAL if kind == 'nominal' else LOW)
        for name, expected in p['reference_zero_files'].items():
            p['frozen_sha256'][rel(zero / name)] = expected
        baseline.verify_sources(p)
        baseline.put_new(plan_path, p)
        digest = sha(plan_path)
        baseline.put_new(arm_out / 'windows-entry.json', dict(status='native-observer-prepared', protocol_sha256=digest))
        baseline.put_new(arm_out / 'freeze.json', dict(status='frozen-before-dynamics', run_id=run_id,
            protocol_sha256=digest, reference_zero=rel(zero), zero_files=p['reference_zero_files'],
            controls=1550, worlds=128, learning_updates=0, hardware_io=False))
        plans[role], hashes[role], references[role] = rel(plan_path), digest, rel(zero)
    master = dict(status='frozen-before-dynamics', stage=STAGE, seed=1242, worlds=128, arms=list(ROLES),
        plans=plans, plan_sha256=hashes, reference_zero=references, prior_zero_comparison=rel(PROOF),
        prior_zero_comparison_sha256=sha(PROOF), maximum_controls_per_arm=1550,
        maximum_total_transitions=3*1550*128, minimum_roll_completion=116,
        minimum_continuous_successes=116, learning_updates=0, hardware_io=False, heldout_used=False,
        automatic_retry=False, automatic_extension=False,
        sequence_rule='Independently verify each completed arm and original gate before admitting the next arm.',
        matched_scope='Reset inputs and RNG are checked; unaligned solver-internal arrays may differ.')
    baseline.put_new(MASTER, master)
    project = read(ROOT / 'project.json')
    if not idle(project):
        raise RuntimeError('Project changed while plans were being frozen')
    project['staged_training_request'].update(current_stage=STAGE + '-ready', current_protocol=rel(MASTER),
        current_protocol_frozen=True, next_gate='Observe and collect the first nominal arm, then independently verify before proceeding.')
    project['goal_execution']['roll_factor_dynamic'] = dict(status='frozen-ready', protocol=rel(MASTER),
        protocol_sha256=sha(MASTER), next_arm=ROLES[0], arms={}, learning_updates=0, hardware_io=False)
    write(ROOT / 'project.json', project)
    (OUT / 'README.md').write_text('# 滚翻摩擦动态对照\n\n已冻结三组方案，等待第一组观察与采集。\n', encoding='utf-8')
    print('MATCHED_ROLL_FROZEN', sha(MASTER), plans, flush=True)


def checked(role):
    master, project = read(MASTER), read(ROOT / 'project.json')
    trial, task = project['goal_execution']['roll_factor_dynamic'], project['staged_training_request']
    if (role not in ROLES or master['arms'] != list(ROLES) or trial['protocol_sha256'] != sha(MASTER)
            or trial['next_arm'] != role or task['pause_requested'] or task['learning_running']
            or project['goal_execution']['status'] != 'active' or project['goal_execution'].get('pause_requested_by_user', False)):
        raise ValueError('This arm is not the currently admitted arm')
    path = ROOT / master['plans'][role]
    p = read(path)
    if (sha(path) != master['plan_sha256'][role] or p['comparison_role'] != role or p['seed'] != 1242
            or p['worlds'] != 128 or p['maximum_controls'] != 1550 or p['learning_updates'] != 0
            or p['hardware_io'] is not False or p['heldout_used'] is not False or p['dynamics_authorized'] is not True):
        raise ValueError('Frozen arm scope changed')
    out = ROOT / p['guarded_output']
    if out.resolve() != (OUT / role).resolve():
        raise ValueError('Arm output escaped the comparison report')
    baseline.verify_sources(p)
    return master, path, p, out


def register(role):
    master, path, p, out = checked(role)
    project = read(ROOT / 'project.json')
    if not idle(project) or (out / 'execution.log').exists() or (ROOT / 'records/evaluations' / p['run_id']).exists():
        raise RuntimeError('Arm is running, paused, or has already been attempted')
    project['staged_training_request'].update(current_stage=p['stage'], current_protocol=rel(path), current_protocol_frozen=True)
    project['goal_execution']['roll_factor_dynamic']['arms'][role] = dict(status='ready-for-native-view', run_id=p['run_id'])
    write(ROOT / 'project.json', project)
    return master, path, p, out


def child(role):
    master, path, plan, out = checked(role)
    task = read(ROOT / 'project.json')['staged_training_request']
    if task['current_stage'] != plan['stage'] or task['current_protocol'] != rel(path) or not task['current_protocol_frozen']:
        raise RuntimeError('Project does not point to this frozen arm')
    log_stat = os.stat(out / 'execution.log')
    if any((os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (log_stat.st_dev, log_stat.st_ino) for fd in (1, 2)):
        raise RuntimeError('Arm must run through the recorded supervisor')
    digest, owner, running = sha(path), os.getpid(), False
    result = dict(status='refused-before-runtime', reason='not-started', role=role,
                  protocol_sha256=digest, learning_updates=0, hardware_io=False)

    def allowed():
        try:
            project = read(ROOT / 'project.json')
            task, trial = project['staged_training_request'], project['goal_execution']['roll_factor_dynamic']
            return (project['goal_execution']['status'] == 'active' and not project['goal_execution'].get('pause_requested_by_user', False)
                    and not task['pause_requested'] and not task['learning_running'] and task['simulation_running'] is running
                    and task['current_stage'] == plan['stage'] and task['current_protocol'] == rel(path)
                    and task['current_protocol_frozen'] and trial['protocol_sha256'] == sha(MASTER)
                    and trial['next_arm'] == role and sha(path) == digest
                    and (not running or trial['arms'][role].get('owner_pid') == owner))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    bridge = NativeRollObservationBounded(plan['run_id'], out, allowed)
    bridge.state.update(purpose=f'滚翻参数对照：{role}，等待查看，尚未采样', maximum_controls=1550)
    code = 2
    try:
        bridge.start()
        print('WAITING_FOR_CURRENT_VIEW', str(bridge.view), flush=True)
        deadline = time.monotonic() + 240
        while not viewing_confirmed(out / 'viewing-confirmation.json', bridge, digest):
            if not bridge.check(force=True) or time.monotonic() >= deadline:
                raise RuntimeError('Current observer unavailable: ' + str(bridge.refusal))
            time.sleep(.2)
        project = read(ROOT / 'project.json')
        if not idle(project) or not allowed():
            raise RuntimeError('Project changed before runtime')
        project['staged_training_request']['simulation_running'] = True
        project['goal_execution']['roll_factor_dynamic']['arms'][role].update(status='running', owner_pid=owner)
        write(ROOT / 'project.json', project)
        running = True
        bridge.last_check = 0
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Supervisor terminated')))

        import numpy as np
        import torch
        import mjlab.envs as env_module
        import common_scene_contract as scene
        import run_continuous_roll_stand_walk as base
        import factor_scene_guard_adapter as bindings
        import model_force_limit_guard_bool as force_guard
        from guarded_roll_collector import make_guarded_collector
        from roll_factor_zero_recorded_hook import recorded_actors
        from roll_scene_fingerprint_hook import scene_inventory
        from run_local_getup_ppo import MemoryWatch

        bounds = read(zero_entry.inside(zero_entry.RANGES))['skills']['roll']
        ranges = {'foot_friction': bounds['training_foot_friction'], 'vin_range_v': bounds['training_voltage']}
        nominal = {'foot_friction': bounds['nominal_foot_friction'], 'vin_range_v': bounds['nominal_voltage']}
        kind = plan['factor_arm']
        factor = dict(name=kind, **plan['parameter_factors'], changed_factors=[] if kind == 'nominal' else ['foot_friction'])
        reference_root = zero_entry.inside(plan['reference_root'])
        motor = read(zero_entry.inside(zero_entry.MOTOR_PARAMS))
        reference_zero = ROOT / plan['reference_zero']
        watch = MemoryWatch(plan['max_gpu_fraction'])

        def continue_checked():
            watch.check()
            return bridge.check()

        with watch.active(), bindings.factor_guard_bindings(base, scene, env_module, arm=factor, nominal=nominal,
                ranges=ranges, startup_root=reference_root, reference_sha256=plan['reference_sha256']) as factor_state:
            install = base.install_guard
            def force_checked_install(spec, folder, requested_root, mode):
                guarded = install(spec, folder, requested_root, mode)
                class ForceCheckedEnv(guarded):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        try:
                            metadata, arrays = force_guard.check(self.sim.mj_model, plan['walking']['joint_names'], 7.7, motor)
                            target = folder / 'actuator-force-limits.npz'
                            with target.open('xb') as stream:
                                np.savez_compressed(stream, **arrays)
                            metadata.update(protocol_sha256=digest, arrays_sha256=sha(target), checked_before_reset=True)
                            baseline.put_new(folder / 'actuator-force-limits.json', metadata)
                        except (Exception, KeyboardInterrupt):
                            super().close()
                            raise
                return ForceCheckedEnv

            collector = make_guarded_collector(base, continue_checked)
            def collect(*args, **kwargs):
                bindings.require_complete(factor_state)
                bridge.begin_collect()
                return collector(*args, **kwargs)
            with ExitStack() as stack:
                stack.enter_context(scene_inventory())
                hook = stack.enter_context(recorded_actors(base, path, continue_checked))
                recorded = base.actors
                def compare_actors(env, obs, spec, folder):
                    output = recorded(env, obs, spec, folder)
                    with np.load(reference_zero / 'actuator-observed-start.npz', allow_pickle=False) as ref, \
                         np.load(folder / 'actuator-observed-start.npz', allow_pickle=False) as now:
                        if set(ref.files) != set(now.files):
                            raise ValueError('Actuator startup fields differ')
                        for name in ref.files:
                            np.testing.assert_array_equal(now[name], ref[name], err_msg=name)
                    old, current = read(reference_zero / 'start-fingerprints.json'), read(folder / 'start-fingerprints.json')
                    if old['rng_digest'] != current['rng_digest'] or set(old['fields']) != set(current['fields']):
                        raise ValueError('Startup RNG or fingerprint field inventory differs')
                    changed = sorted(name for name in old['fields'] if old['fields'][name] != current['fields'][name])
                    known = set(read(ROOT / 'records/reports/roll-matched-factor-preflight-20261116/'
                                     'raw-nominal-zero-verification.json')['same_factor_changed_fields'])
                    unexpected = sorted(set(changed) - known)
                    write(folder / 'matched-zero-runtime.json', dict(status='stop-unexpected-fingerprint' if unexpected else 'passed-input-and-rng-check',
                        actuator_arrays_exact=True, rng_exact=True, fields_checked=len(old['fields']),
                        differing_field_names=changed, unexpected_field_names=unexpected, physical_substeps=0,
                        scope='Matched reset inputs and RNG; changed internal fingerprints are recorded, not assumed equal.'))
                    if unexpected:
                        raise ValueError('Unexpected startup fingerprint differences: ' + ', '.join(unexpected))
                    return output
                stack.enter_context(patch.object(base, 'actors', compare_actors))
                stack.enter_context(patch.multiple(base, PLAN=path, OUT=out, PAGE=ROOT / plan['page'],
                    PROJECT=ROOT / 'project.json', STARTUP=reference_root, install_guard=force_checked_install, collect=collect))
                stack.enter_context(patch.object(base.live, 'publish', side_effect=bridge.publish))
                stack.enter_context(torch.no_grad())
                base.run(False)
                if not hook['completed']:
                    raise RuntimeError('Startup actuator and fingerprint hook did not complete')
            bindings.require_complete(factor_state)
        folder = ROOT / 'records/evaluations' / plan['run_id']
        manifest = read(folder / 'guarded-evaluation.json')
        result.update(status=manifest['status'], reason=manifest['stop_reason'], observer_refusal=bridge.refusal,
                      evaluation=rel(folder), actual_control_steps=manifest['actual_control_steps'],
                      actual_physics_substeps=manifest['actual_physics_substeps'])
        code = 0 if manifest['status'] == 'collected-review-required' else 3
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        result.update(status='stopped-review-required', reason=type(exc).__name__ + ': ' + str(exc),
                      observer_refusal=bridge.refusal)
    finally:
        try:
            bridge.close(result)
        finally:
            if 'watch' in locals():
                write(out / 'gpu-telemetry.json', dict(samples=watch.rows, failure=repr(watch.error) if watch.error else None))
            if running:
                project = read(ROOT / 'project.json')
                trial = project['goal_execution']['roll_factor_dynamic']
                if trial['arms'][role].get('owner_pid') == owner:
                    project['staged_training_request']['simulation_running'] = False
                    trial['arms'][role].update(status=result['status'], result=rel(out / 'result.json'))
                    write(ROOT / 'project.json', project)
                else:
                    result['project_cleanup_error'] = 'owner-mismatch'
                    code = 3
            result['finished_at'] = datetime.now(timezone.utc).isoformat()
            baseline.put_new(out / 'result.json', result)
    return code


def run(role):
    os.environ.update(MUJOCO_GL='egl', GALLIUM_DRIVER='d3d12', MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
    lock_path = Path('/home/atlas/XGO-Duck/training.lock')
    with lock_path.open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            master, path, plan, out = register(role)
            command = [sys.executable, '-B', str(ROOT / 'scripts/training/parent_exit_exec.py'), str(os.getpid()), '--',
                       sys.executable, '-u', str(Path(__file__).resolve()), 'child', '--arm', role]
            receipt = run_recorded(command, cwd=ROOT, log_path=out / 'execution.log',
                receipt_path=out / 'execution-receipt.json', max_wall_seconds=900)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    print('MATCHED_ROLL_RECEIPT', role, receipt['status'], receipt['exit_code'], flush=True)
    return 0 if receipt['status'] == 'completed' and receipt['exit_code'] == 0 else 2


def verify(role):
    master, path, plan, out = checked(role)
    folder = ROOT / 'records/evaluations' / plan['run_id']
    if not (out / 'result.json').is_file() or not (folder / 'guarded-evaluation.json').is_file():
        raise RuntimeError('No complete recorded evaluation is available for independent review')
    import verify_native_nominal_baseline as old
    with patch.object(old, 'RUN_ID', plan['run_id']), patch.object(old, 'REPORT', out), \
         patch.object(old, 'FOLDER', folder), patch.object(old, 'PLAN', path), patch.object(sys, 'argv', ['verify']):
        old.main()
    legacy = out / 'independent-analysis.json'
    original = read(legacy)
    legacy.rename(out / 'independent-analysis-legacy.json')
    match = read(folder / 'matched-zero-runtime.json')
    if match['status'] != 'passed-input-and-rng-check' or not match['actuator_arrays_exact'] or not match['rng_exact']:
        raise ValueError('Runtime zero-step match did not pass')
    result = dict(original, role=role, factor=plan['parameter_factors'],
        source_verifier_sha256=sha(ROOT / 'scripts/validation/verify_native_nominal_baseline.py'),
        adapter_sha256=sha(Path(__file__)), start_match=rel(folder / 'matched-zero-runtime.json'),
        scope='One matched-input official roll route arm; other arms and full-state equality are not implied.')
    write(legacy, result)
    project = read(ROOT / 'project.json')
    trial, task = project['goal_execution']['roll_factor_dynamic'], project['staged_training_request']
    if task['simulation_running'] or task['learning_running'] or trial['next_arm'] != role:
        raise RuntimeError('Project changed before independent result registration')
    passed = result['status'] == 'completed-baseline-independently-verified' and result['passed_development_gate'] is True
    trial['arms'][role].update(status='independently-verified', analysis=rel(legacy), passed_original_gate=passed)
    index = ROLES.index(role)
    trial['next_arm'] = ROLES[index + 1] if passed and index + 1 < len(ROLES) else None
    trial['status'] = 'ready-next-arm' if trial['next_arm'] else ('comparison-ready' if passed else 'stopped-after-review')
    task.update(current_stage=STAGE + '-ready', current_protocol=rel(MASTER), current_protocol_frozen=True,
                next_gate=('Observe and collect ' + trial['next_arm'] if trial['next_arm'] else 'Review the completed or stopped comparison.'))
    write(ROOT / 'project.json', project)
    print('MATCHED_ROLL_VERIFIED', role, result['counts'], 'passed', passed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'child', 'verify'))
    parser.add_argument('--arm', choices=ROLES)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    elif not args.arm:
        parser.error('--arm is required')
    elif args.action == 'run':
        raise SystemExit(run(args.arm))
    elif args.action == 'child':
        raise SystemExit(child(args.arm))
    else:
        verify(args.arm)
