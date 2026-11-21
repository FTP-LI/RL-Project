"""Train the matched forward reward arms and verify both actual exports."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_forward_yaw_capacity as capacity

D, ROOT, PROJECT = capacity.D, capacity.ROOT, capacity.PROJECT
REPORT = capacity.admission.REPORT / 'pair'
PLAN = ROOT / 'configs/experiments/forward_low_friction_yaw_reward_pair.json'
ENTRY = Path(__file__).resolve()
STAGE = capacity.ZERO.check.STAGE + '-pair'
KEY = 'forward_yaw_reward_pair'
ROLES = ('control', 'candidate')
LABELS = {'control': '原行走奖励对照组', 'candidate': '低摩擦前进偏航奖励候选组'}


def freeze():
    project, selection_plan = D.read(PROJECT), D.read(capacity.PLAN)
    task, goal = project['staged_training_request'], project['goal_execution']
    selection = goal[capacity.KEY]
    pr = ROOT / selection['capacity_evidence']
    selected = D.read(pr)
    D.require(D.pair.trial.allowed(project, capacity.STAGE)
              and not (task['learning_running'] or task['simulation_running'])
              and selection['status'] == 'capacity-passed' and D.sha(pr) == selection['capacity_evidence_sha256']
              and D.sha(capacity.PLAN) == selection['capacity_protocol_sha256']
              and selected['status'] == 'passed' and not selected['test_3328_warranted']
              and not PLAN.exists() and not REPORT.exists(), 'Finish the measured capacity scope before formal pairing')
    draft = D.read(ROOT / selection_plan['draft'])
    budget = draft['pair_budget']
    n = selected['selected_worlds']
    when = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    arms = {r: dict(role=r, worlds=n, updates=budget['updates_per_arm'], critic_only_updates=budget['critic_only_updates'],
                   run_id=f'duck_forward_yaw_reward_{r}_e{n}_s{selection_plan["seed"]}_{when}',
                   transitions=n*budget['updates_per_arm']*budget['controls_per_update'], record_late=True)
            for r in ROLES}
    instructions = [ENTRY, capacity.PLAN, pr, ROOT / 'scripts/training/walk_backward_export.py']
    selection_plan['sources'].update({D.rel(f): D.sha(f) for f in instructions})
    selection_plan.update(stage=STAGE, scope='Matched forward reward training', arms=arms, order=list(arms),
                          worlds=n, capacity_selection=D.rel(pr), wall_seconds=540,
                          change='The candidate adds the predefined capped absolute physical yaw cost.')
    capacity.ZERO.verify_sources(selection_plan)
    REPORT.mkdir()
    D.put(PLAN, selection_plan)
    goal[KEY] = dict(status='capacity-ready', capacity_completed={}, capacity_protocol_sha256=D.sha(PLAN),
                     protocol=D.rel(PLAN), selection=D.rel(pr))
    task.update(current_stage=STAGE, current_protocol=D.rel(PLAN), current_protocol_frozen=True,
                next_gate='Train both equal-sample forward arms, export, and verify their actual outputs.')
    D.pair.trial.write(PROJECT, project)
    print('FORWARD_YAW_PAIR_FROZEN', n, D.sha(PLAN), flush=True)


@contextmanager
def paired_worker(role):
    bounded_observer = capacity.bounded.BoundedNativePPO

    def observer(*args, **kwargs):
        kwargs['purpose'] = LABELS[role]
        return bounded_observer(*args, **kwargs)

    with patch.multiple(capacity, REPORT=REPORT, PLAN=PLAN, ENTRY=ENTRY, STAGE=STAGE, KEY=KEY), \
            patch.object(capacity.bounded, 'BoundedNativePPO', observer):
        yield


def verify_export(plan, arm, result):
    import imageio_ffmpeg
    import numpy as np
    import torch
    from common_walk_startup import make_actor
    from verify_roll_exposure_parent import finite
    from walk_backward_export import export_walking_actor

    torch.set_num_threads(4)
    folder = ROOT / result['record']
    observation, local = (D.read(folder / file) for file in ('observation.json', 'starting-state.json'))
    metrics_path, checkpoint_path = folder / 'learning-metrics.jsonl', folder / 'model-final.pt'
    case = D.read(capacity.ZERO.check.OUTPUT)
    reward_weight = -case['coefficient_per_rad_s'] if arm['role'] == 'candidate' else 0.
    D.require(D.sha(metrics_path) == result['metrics_sha256'] and D.sha(checkpoint_path) == result['checkpoint_sha256']
              and result['transitions'] == arm['transitions'] and result['updates'] == arm['updates']
              and result['plan_sha256'] == D.sha(PLAN) and local['parent_sha256'] == plan['parent_sha256']
              and local['exact_shared_parent'] and local['yaw_cost_weight'] == reward_weight,
              'Actual training artifacts or reward role differ')
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding='utf-8').splitlines()]
    expected_controls = arm['updates']*24
    D.require(metrics == observation['rows'] and len(metrics) == arm['updates']
              and [r['control'] for r in metrics] == list(range(24, expected_controls+1, 24)), 'Training log differs')
    parent_path = ROOT / plan['parent']
    a = torch.load(parent_path, map_location='cpu', weights_only=True)
    b = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    D.require(finite(b) and b['iter'] == arm['updates']-1
              and b['infos']['env_state']['common_step_counter'] == expected_controls
              and b['infos']['parent_sha256'] == plan['parent_sha256'], 'Checkpoint could not restore the declared update')
    state = b['actor_state_dict']
    D.require(all(torch.equal(state[k], v) for k, v in a['actor_state_dict'].items()
                  if k.startswith('obs_normalizer.')) and
              any(not torch.equal(state[k], v) for k, v in a['actor_state_dict'].items() if k.startswith('mlp.'))
              and len(b['optimizer_state_dict']['state']) > 0, 'Trained actor, normalization or optimizer differs')
    media = []
    for entry in result['clips']:
        path = folder / entry['file']
        frames = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
        try:
            metadata = next(frames)
            decoded = sum(1 for _ in frames)
        finally:
            frames.close()
        D.require(decoded == entry['frames'] and metadata['fps'] == entry['fps'] == 25
                  and not entry['extra_physics_steps'], 'Training clip differs from actual simulation evidence')
        media.append(dict(path=D.rel(path), sha256=D.sha(path), frames=decoded, complete=entry['full_action']))
    D.require(any(v['complete'] for v in media), 'A complete training sample must be visible')
    walking = D.read(ROOT / plan['route'])['walking']
    actor = make_actor(walking, 'cpu')
    actor.load_state_dict({name: state[name] for name in actor.state_dict()}, strict=True)
    zero_inputs = parent_path.parent / 'control-initial-state.npz'
    with np.load(zero_inputs, allow_pickle=False) as data:
        inputs = data['actor'][::16]
    output = ROOT / 'artifacts' / arm['run_id']
    output.mkdir()
    manifest = export_walking_actor(actor, ROOT / walking['model'], walking['model_sha256'],
                                    output / 'policy.onnx', inputs)
    payload = dict(manifest, role=arm['role'], scope='Forward reward comparison awaiting independent action scoring',
                   onnx=D.rel(output / 'policy.onnx'), checkpoint=D.rel(checkpoint_path),
                   checkpoint_sha256=D.sha(checkpoint_path), record=D.rel(folder), training_clips=media,
                   protocol=D.rel(PLAN), protocol_sha256=D.sha(PLAN), samples=arm['transitions'],
                   updates=arm['updates'], warmup_updates=arm['critic_only_updates'], hardware_io=False)
    D.put(output / 'manifest.json', payload)
    return D.rel(output / 'manifest.json'), payload


def batch():
    with paired_worker('control'):
        project, plan, state = capacity.checked()
    state.update(status='capacity-batch', capacity_parent_pid=os.getpid())
    D.pair.trial.write(PROJECT, project)
    page = REPORT / 'current-training.md'
    ready = False
    try:
        for completed, role in enumerate(ROLES):
            folder = REPORT / role
            folder.mkdir()
            page.write_text(f'# 低摩擦前进偏航奖励配对\n\n已完成训练{completed}/2组。正在运行{LABELS[role]}。\n\n'
                            f'[运行状态]({role}/native-view/native-status.txt) · [命令与日志]({role}/execution.log)'
                            ' · [实验说明](../README.md)\n', encoding='utf-8')
            cmd = [sys.executable, '-B', '-u', str(ENTRY), 'worker', role]
            with (folder / 'execution.log').open('x', encoding='utf-8') as log:
                log.write('COMMAND ' + subprocess.list2cmdline(cmd) + '\n')
                log.flush()
                env = dict(os.environ, MUJOCO_GL='egl', GALLIUM_DRIVER='d3d12', MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        project = D.read(PROJECT)
        active = project['goal_execution'][KEY]
        exports = {}
        for role, arm in plan['arms'].items():
            marked = active['capacity_completed'][role]
            proof = ROOT / marked['path']
            D.require(D.sha(proof) == marked['sha256'], 'Trained run receipt changed')
            manifest_path, manifest = verify_export(plan, arm, D.read(proof))
            exports[role] = dict(manifest=manifest_path, manifest_sha256=D.sha(ROOT / manifest_path), **manifest)
        result = REPORT / 'training-check.json'
        D.put(result, dict(status='passed', paired_samples=plan['arms']['control']['transitions'], runs=exports,
                           original_gate_evaluation_pending=True, hardware_io=False))
        active.update(status='trained-exports-verified', evidence=D.rel(result), evidence_sha256=D.sha(result))
        project['staged_training_request'].update(current_evidence=D.rel(result),
            next_gate='Independently score both frozen forward exports against the original 20-second gates.')
        D.pair.trial.write(PROJECT, project)
        page.write_text('# 低摩擦前进偏航奖励配对\n\n两组训练及导出均已核验，进入独立动作评价。\n\n'
                        '[权重与录像清单](training-check.json) · [对照运行](control/execution.log) · '
                        '[候选运行](candidate/execution.log)\n', encoding='utf-8')
        ready = True
        print('FORWARD_YAW_PAIR_TRAINED', result, flush=True)
    finally:
        project = D.read(PROJECT)
        state = project['goal_execution'][KEY]
        state.pop('capacity_parent_pid', None)
        if not ready:
            state['status'] = 'training-review'
            if state.pop('capacity_worker_pid', None) is not None:
                project['staged_training_request'].update(learning_running=False, simulation_running=False)
        D.pair.trial.write(PROJECT, project)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('freeze', 'batch', 'worker'))
    parser.add_argument('role', nargs='?', choices=ROLES)
    args = parser.parse_args()
    if args.operation == 'worker':
        with paired_worker(args.role):
            capacity.worker(args.role)
    else:
        with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (freeze if args.operation == 'freeze' else batch)()
