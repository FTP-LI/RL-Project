"""Compare the completed monitored PPO pair with its common parent.

This entry reuses the frozen shared-release collector and original scoring.
The only observation change is copying an already rendered video frame to a
local view; it never advances physics or changes the recorded action. There is
no training, model promotion, automatic retry, or hardware access here.
"""
from __future__ import annotations

import json
import os
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
sys.path.insert(0, str(ROOT / 'scripts/validation'))

import numpy as np
import torch
import collect_getup_supine_evaluation as collector
import evaluate_getup_early_support_ppo as early
import getup_shared_release as shared
import training_live_page as live
import verify_getup_supine_evaluation as verifier
from getup_anchor_same_release import model_metadata
from getup_shared_release_terrain import cpu_scene_readback
from native_roll_observation import NativeRollObservation
from observed_getup_entry import viewing_confirmed
from record_io import write
from s1_reference_inputs import read, sha

PLAN = ROOT / 'configs/experiments/getup_monitored_pair_action_s3267.json'
BASE = ROOT / 'configs/experiments/local_getup_ppo_evaluation.json'
TRAINING = ROOT / 'records/reports/getup-monitored-authorization-20261116'
OUT = ROOT / 'records/reports/getup-monitored-pair-action-s3267'
STAGE = 'observed-getup-action-evaluation'
ROLES = ('parent', 'control', 'candidate')


def rel(path):
    return path.relative_to(ROOT).as_posix()


def freeze():
    if PLAN.exists() or OUT.exists():
        raise FileExistsError('Preserve the previous plan and attempt')
    comparison = read(TRAINING / 'comparison.json')
    if comparison['status'] != 'completed-monitored-runtime-review':
        raise ValueError('Training pair has not passed its record review')
    spec = deepcopy(read(BASE))
    p = spec['environment']
    n, seed = 1024, 3267
    codes = np.concatenate(([1], np.random.default_rng(seed).permutation(
        np.repeat(np.arange(4), [96, 95, 416, 416]))))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(shared.POSES)}
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_monitor-pair_action_e{n}_s{seed}_{stamp}'
    p.update(num_envs=n, seed=seed, stage=STAGE, run_id=run_id,
             split=dict(pose_codes=codes.tolist(), groups=groups),
             maximum_control_transitions=n * p['maximum_control_steps'],
             video_worlds={pose: groups[pose][0] for pose in shared.POSES})
    parent = read(ROOT / 'records/experiments' / comparison['arms']['control']['run_id'] / 'protocol.json')
    other = read(ROOT / 'records/experiments' / comparison['arms']['throughput']['run_id'] / 'protocol.json')
    for arm in (parent, other):
        if (arm['model_sha256'] != p['model_sha256'] or arm['model'] != p['model']
                or arm['parent_checkpoint_sha256'] != parent['parent_checkpoint_sha256']
                or sha(ROOT / arm['parent_checkpoint']) != arm['parent_checkpoint_sha256']):
            raise ValueError('The evaluated parent is not the common training checkpoint')
    policies = {'parent': dict(model=p['model'], model_sha256=p['model_sha256'], seed=seed)}
    references = [TRAINING / 'comparison.json', BASE, Path(__file__).resolve()]
    references += [ROOT / parent['parent_checkpoint']]
    for role, source in (('control', 'control'), ('candidate', 'throughput')):
        item = comparison['arms'][source]
        proof = TRAINING / 'training' / source / 'independent-verification.json'
        run = ROOT / 'records/experiments' / item['run_id']
        model = f"artifacts/{item['run_id']}/policy.onnx"
        actual = dict(model=model, model_sha256=item['onnx_sha256'])
        if (read(proof)['status'] != 'passed-record-and-export-checks'
                or sha(ROOT / model) != item['onnx_sha256']
                or sha(run / 'model-final.pt') != item['checkpoint_sha256']
                or model_metadata(p, actual) != spec['model_interface']):
            raise ValueError('Model, checkpoint, or interface review failed: ' + role)
        policies[role] = dict(actual, seed=seed)
        references += [proof, run / 'protocol.json', run / 'parent-readback.json', ROOT / model]
    for name in ('scripts/validation/evaluate_monitored_getup_pair.py',
                 'scripts/training/native_roll_observation.py',
                 'scripts/training/native_run_observer.py',
                 'scripts/training/native_observer_guard.py',
                 'scripts/training/observed_getup_entry.py'):
        spec['source_sha256'][name] = sha(ROOT / name)
    spec['references'].update({rel(path): sha(path) for path in references})
    spec.update(stage=STAGE, branches=list(ROLES), policies=policies,
                cameras={role: {pose: groups[pose][0] for pose in shared.POSES} for role in ROLES},
                maximum_control_transitions=n * spec['maximum_unique_controls'],
                maximum_warp_snapshot_bytes=512 * 1024 * 1024,
                maximum_sampling_wall_seconds=1200,
                choice='Diagnostic only. A model may be considered for independent confirmation only after all original per-pose and overall gates pass; no automatic promotion.',
                comparison_scope='Three policies, same physical fall and restored release, original 11-second scoring.',
                training_evidence=rel(TRAINING / 'comparison.json'),
                learning_updates=0, hardware_io=False, heldout_used=False,
                automatic_retry=False, automatic_extension=False)
    shared.assert_sources(spec)
    scene = cpu_scene_readback(p, spec)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preparation unexpectedly initialized CUDA')
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT / 'preflight.json', dict(status='passed-cpu-zero-step', scene=scene,
          protocol_sha256=sha(PLAN), seed=seed, num_envs=n, physical_substeps=0,
          learning_updates=0, hardware_io=False))
    project = read(ROOT / 'project.json')
    task = project['staged_training_request']
    if (task['pause_requested'] or task['simulation_running'] or task['learning_running']
            or project['goal_execution']['status'] != 'active'):
        raise RuntimeError('Project is paused or another run is active')
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
                next_gate='Observe the current run, collect one bounded shared-release evaluation, and independently verify saved trajectories.')
    project['goal_execution']['monitored_getup_action_evaluation'].update(
        status='frozen-cpu-zero-step-passed', protocol=rel(PLAN), protocol_sha256=sha(PLAN),
        run_id=run_id, environments=n, learning_updates=0)
    write(ROOT / 'project.json', project)
    print('ACTION_EVALUATION_FROZEN', run_id, sha(PLAN), flush=True)


def allowed(protocol_hash):
    try:
        project = read(ROOT / 'project.json')
        task, goal = project['staged_training_request'], project['goal_execution']
        return (goal['status'] == 'active' and not goal.get('pause_requested_by_user', False)
                and not task['pause_requested'] and not task['learning_running']
                and task['current_stage'] == STAGE and task['current_protocol_frozen']
                and task['current_protocol'] == rel(PLAN) and sha(PLAN) == protocol_hash)
    except (OSError, KeyError, ValueError, TypeError):
        return False


def running(value):
    project = read(ROOT / 'project.json')
    task = project['staged_training_request']
    item = project['goal_execution']['monitored_getup_action_evaluation']
    if value:
        if not allowed(sha(PLAN)) or task['simulation_running']:
            raise RuntimeError('Project does not allow a new evaluation')
        item['owner_pid'] = os.getpid()
    elif item.get('owner_pid') != os.getpid():
        raise RuntimeError('Evaluation owner changed')
    task['simulation_running'] = value
    item['status'] = 'running' if value else 'stopped-review-required'
    write(ROOT / 'project.json', project)


def run():
    spec = read(PLAN)
    digest = sha(PLAN)
    if read(OUT / 'preflight.json')['protocol_sha256'] != digest or not allowed(digest):
        raise RuntimeError('The frozen evaluation is not admitted')
    shared.assert_sources(spec)
    run_id = spec['environment']['run_id']
    record = ROOT / 'records/evaluations' / run_id
    if record.exists() or (OUT / 'entry-result.json').exists():
        raise FileExistsError('No automatic retry of this evaluation')
    observer = NativeRollObservation(run_id, OUT, lambda: allowed(digest))
    observer.state.update(purpose='起身三组共同跌落评估：等待查看，尚未采样',
                          maximum_controls=spec['maximum_unique_controls'])
    result = dict(status='not-started', run_id=run_id, protocol_sha256=digest,
                  learning_updates=0, hardware_io=False)
    owned = False
    try:
        observer.start()
        print('WAITING_FOR_CURRENT_VIEW', str(observer.view), flush=True)
        deadline = time.monotonic() + 240
        while not viewing_confirmed(OUT / 'viewing-confirmation.json', observer, digest):
            if not observer.check() or time.monotonic() >= deadline:
                raise RuntimeError('Current native view unavailable: ' + str(observer.refusal))
            time.sleep(.2)
        running(True)
        owned = True
        previous_frame = [0.0]
        base_camera = collector.FixedGetupCamera

        class ActualFrameCamera(base_camera):
            def render(self):
                pixels = super().render()
                if time.monotonic() - previous_frame[0] >= 30:
                    from PIL import Image
                    temporary = record / 'live-frame.jpg.tmp'
                    Image.fromarray(pixels).save(temporary, format='JPEG', quality=85)
                    os.replace(temporary, record / 'live-frame.jpg')
                    previous_frame[0] = time.monotonic()
                return pixels

        def publish(actual_record, state):
            state = dict(state)
            if state['status'] not in ('starting', 'running', 'gpu-zero-step'):
                state['purpose'] += '；轨迹正在保存或等待独立复核'
                state['status'] = 'running'
                observer.in_collect = False
            if (record / 'live-frame.jpg').is_file():
                state['frame'] = 'live-frame.jpg'
            if state['controls'] and not observer.in_collect and state['controls'] < spec['maximum_unique_controls']:
                observer.begin_collect()
            observer.publish(actual_record, state)
            description = record / 'README.md'
            description.write_text(
                '# 起身共同跌落评估\n\n父模型、同步组和监控组共享同一物理释放，'
                '保留四种姿态完整 11 秒原片；此运行不进行学习。\n\n'
                f'[方案]({os.path.relpath(PLAN, record).replace(os.sep, "/")}) · '
                f'[本机观察]({os.path.relpath(OUT / "native-view/native-status.txt", record).replace(os.sep, "/")}) · '
                '[状态](evaluation.json) · [日志](execution.log)\n', encoding='utf-8')

        base_watch = early.MemoryWatch

        class ObservedWatch(base_watch):
            def check(self):
                super().check()
                if not observer.check():
                    raise RuntimeError('Native observation stopped: ' + str(observer.refusal))

        with ExitStack() as stack:
            stack.enter_context(patch.object(early, 'STAGE', STAGE))
            stack.enter_context(patch.object(early, 'PLAN', PLAN))
            stack.enter_context(patch.object(early, 'MemoryWatch', ObservedWatch))
            stack.enter_context(patch.object(live, 'publish', publish))
            stack.enter_context(patch.object(collector, 'FixedGetupCamera', ActualFrameCamera))
            early.run(spec, record)
        result['status'] = 'collected-review-required'
    except BaseException as exc:
        result.update(status='stopped-review-required', reason=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        try:
            observer.close(result)
        finally:
            if owned:
                running(False)
            result['finished_at'] = datetime.now(timezone.utc).isoformat()
            write(OUT / 'entry-result.json', result)


def verify():
    spec = read(PLAN)
    if sha(PLAN) != read(OUT / 'preflight.json')['protocol_sha256']:
        raise RuntimeError('Protocol differs from CPU preflight')
    shared.assert_sources(spec)
    record = ROOT / 'records/evaluations' / spec['environment']['run_id']
    state = read(record / 'evaluation.json')
    if state['status'] != 'collected-review-required':
        raise RuntimeError('Sampling was not completed')

    def select(roles, reports, comparisons, minimum_gain):
        return dict(eligible_for_independent_confirmation={role: bool(reports[role]['all_motion_gates_passed'])
                                                            for role in roles},
                    selected_for_independent_confirmation=None,
                    model_promotion_allowed=False,
                    explanation='This is a development comparison; promotion requires a separate independent confirmation.')

    with patch.object(verifier, 'select', select):
        result = verifier.verify(spec, record, state, minimum_gain=0)
    write(OUT / 'analysis.json', dict(result, protocol_sha256=sha(PLAN)))
    project = read(ROOT / 'project.json')
    project['goal_execution']['monitored_getup_action_evaluation']['status'] = 'independently-verified'
    write(ROOT / 'project.json', project)
    print('ACTION_EVALUATION_VERIFIED', {r: a['combined_successes'] for r, a in result['arms'].items()}, flush=True)


if __name__ == '__main__':
    modes = {'freeze': freeze, 'run': run, 'verify': verify}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit('Usage: evaluate_monitored_getup_pair.py {freeze|run|verify}')
    modes[sys.argv[1]]()
