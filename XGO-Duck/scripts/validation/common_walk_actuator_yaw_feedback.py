"""Evaluate a bounded forward-only correction on the fixed seven-command plant."""
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import common_walk_forward_phase_screen as rollout
import common_walk_yaw_pulse as pulse
from common_walk_targets import ProjectedWalkAction
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_actuator_yaw_feedback import ActuatorYawFeedback, projected_feedback_target
from walking_command_controller import create_controller


PLAN = ROOT/'configs/experiments/common_walk_actuator_yaw_feedback.json'
DRAFT = ROOT/'configs/experiments/common_walk_actuator_yaw_feedback_draft.json'
OUT = ROOT/'records/reports/common-walk-actuator-yaw-feedback'
PAGE = ROOT/'records/walk-actuator-feedback-live.html'
STAGE = 'S2-forward-yaw-actuator-feedback'
ROLES = ('reference-a', 'reference-b', 'actuator-feedback')


class Controller:
    def __init__(self, plan, settings, desired):
        self.base = create_controller(len(desired), .02)
        self.desired = desired
        self.forward = desired[:, 0] > 0.
        assert np.flatnonzero(self.forward).tolist() == sorted(plan['groups']['forward'])
        self.hip = [20+plan['joint_names'].index(f'{side}_hip_pitch') for side in ('left', 'right')]
        self.previous = np.zeros((len(desired), 2), np.float32)
        self.ready = np.zeros(len(desired), bool)

    def features(self, observed):
        current = np.column_stack((observed[:, 2], observed[:, self.hip].sum(axis=1)))
        change = np.where(self.ready[:, None], current-self.previous, 0.)
        self.previous[:] = current
        self.ready[:] = True
        return np.column_stack((current, change))

    def reset(self, ids):
        self.base.reset(ids)
        self.previous[ids] = 0.
        self.ready[ids] = False

    def snapshot(self):
        return self.base.ema.copy(), self.base.ready.copy(), self.previous.copy(), self.ready.copy()

    def restore(self, state):
        self.base.ema[:], self.base.ready[:], self.previous[:], self.ready[:] = state

    def map(self, observed, step, candidate, active):
        return self.base.map(self.desired, observed)


def component_probe(p, settings, desired, screen):
    controller = Controller(p, settings, desired)
    base = create_controller(len(desired), .02)
    active = np.ones(len(desired), bool)
    obs = np.zeros((len(desired), 61), np.float32)
    obs[:, [2, *controller.hip]] = [1., 2., 3.]
    np.testing.assert_array_equal(controller.features(obs), np.tile([1., 5., 0., 0.], (len(desired), 1)))
    np.testing.assert_array_equal(controller.map(obs, 50, False, active), base.map(desired, obs))
    fork = controller.snapshot()
    obs[:, [2, *controller.hip]] = [2., -2., 1.]
    changed = controller.features(obs)
    np.testing.assert_array_equal(changed, np.tile([2., -1., 1., -6.], (len(desired), 1)))
    expected = controller.map(obs, 51, True, active)
    np.testing.assert_array_equal(expected, base.map(desired, obs))
    controller.restore(fork)
    np.testing.assert_array_equal(controller.features(obs), changed)
    np.testing.assert_array_equal(controller.map(obs, 51, False, active), expected)
    controller.restore(fork)
    controller.reset([0])
    reset = controller.features(obs)
    np.testing.assert_array_equal(reset[0], [2., -1., 0., 0.])
    np.testing.assert_array_equal(reset[1:], changed[1:])
    law = ActuatorYawFeedback(screen, screen['nominated_cap_rad_s'])
    assert np.isfinite(law.commands(reset)).all() and (abs(law.commands(reset)) <= law.cap).all()
    return dict(per_world_selective_reset=True, exact_fork_restore=True, nominal_commands_exact=True,
                number_of_worlds=len(desired), actual_observation_velocity_indices=controller.hip)


def freeze():
    from mjlab.envs import ManagerBasedRlEnv

    assert not PLAN.exists() and not OUT.exists()
    draft = read(DRAFT)
    spec = read(rollout.PLAN)
    p = spec['environment']
    matched, screen_path, proof_path = (ROOT/draft[key] for key in
                                        ('matched_target_evidence', 'feedback_screen', 'target_preflight'))
    saved_inputs = ROOT/'records/reports/common-walk-confirmation/development-yaw-review.json'
    screen, proof = read(screen_path), read(proof_path)
    assert read(matched)['response']['passes_history_isolation_gate']
    assert screen['proof_sha256'] == sha(matched)
    assert proof['screen_sha256'] == sha(screen_path)
    assert proof['source_sha256'] == screen['source_sha256'] == sha(saved_inputs)
    assert proof['controller_sha256'] == sha(Path(__file__).with_name('walking_actuator_yaw_feedback.py'))
    assert proof['admitted_to_new_closed_loop'] and set(proof['cohorts']) == {'1042', '1043'}
    assert all(row['passed'] for row in proof['cohorts'].values())
    feedback = draft['feedback']
    assert feedback['maximum_yaw_input_rad_s'] == screen['nominated_cap_rad_s']
    assert feedback['maximum_joint_correction_rad'] == proof['maximum_joint_change_rad']
    seed, n, k, h = (draft[key] for key in ('development_seed', 'worlds', 'shared_prefix_controls', 'controls_per_branch'))
    assert not any(f'_s{seed}_' in path.name for path in (ROOT/'records/evaluations').iterdir())
    permutation = np.random.default_rng(70000+seed).permutation(n)
    size = p['episodes_per_command']
    slot = list(p['commands']).index('forward')*size
    origin = np.flatnonzero(permutation == 0).item()
    permutation[slot], permutation[origin] = permutation[origin], permutation[slot]
    groups = {name: permutation[i*size:(i+1)*size].tolist() for i, name in enumerate(p['commands'])}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    p.update(seed=seed, assignment_seed=70000+seed, groups=groups, stage=STAGE,
             run_id=f'duck_walk_actuator-feedback_paired_s{seed}_{stamp}')
    assert n == p['parallel_worlds'] and h == p['maximum_control_steps']
    controls = k+len(ROLES)*(h-k)
    assert n*controls == draft['maximum_transitions'] and 5*controls == draft['maximum_batched_substeps']
    assert spec['gate']['minimum_yaw_gain_rad_s'] == screen['minimum_gain_rad_s']
    spec.update(stage=STAGE, roles=list(ROLES), phase=dict(warmup_controls=k),
                status='frozen-before-physical-controls', maximum_control_transitions=n*controls,
                maximum_physics_substeps=5*controls, maximum_sampling_wall_seconds=720,
                feedback=feedback, feedback_screen=draft['feedback_screen'],
                scope='One new development seed, original seven command gates, paired controls after a shared one-second prefix.')
    spec['gate']['bootstrap_seed'] = seed*100
    for path in (rollout.PLAN, DRAFT, matched, screen_path, proof_path, saved_inputs):
        spec['reference_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (Path(__file__), Path(__file__).with_name('verify_walk_actuator_yaw_feedback.py'),
                 Path(__file__).with_name('walking_actuator_yaw_feedback.py'), Path(pulse.__file__),
                 ROOT/'scripts/training/training_live_page.py'):
        spec['source_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    rollout.check_sources(spec)
    _, desired = rollout.groups_and_commands(p)
    component = component_probe(p, spec['phase'], desired, screen)
    torch.set_num_threads(4)
    cfg, _ = rollout.make_config(p)
    env = ManagerBasedRlEnv(cfg=cfg, device='cpu', render_mode=None)
    try:
        env.reset(seed=seed)
        env.command_manager.get_command('twist').copy_(torch.from_numpy(desired))
        snapshot = rollout.snapshot_probe(env, spec['maximum_warp_snapshot_bytes'])
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT/'preflight.json', dict(status='passed-cpu-zero-step', protocol_sha256=sha(PLAN), snapshot=snapshot,
          component=component, physical_substeps=0, learning_updates=0, heldout_used=False))
    (OUT/'README.md').write_text('# 前进执行器反馈\n\n[冻结方案](../../../configs/experiments/'+PLAN.name+
        ') · [零步检查](preflight.json) · [运行观察](../../'+PAGE.name+')\n', encoding='utf-8')
    print('WALK_ACTUATOR_FEEDBACK_FROZEN', sha(PLAN), p['run_id'], flush=True)


original_collect = rollout.collect


def collect(env, actor, spec, record, state, desired):
    term = env.action_manager.get_term('joint_pos')
    assert isinstance(term, ProjectedWalkAction) and term.cfg.scale == 1. and not term.cfg.use_default_offset
    process = term.process_actions
    limits = torch.as_tensor(term.cfg.physical_limits, device=env.device, dtype=torch.float32)
    settings = spec['feedback']
    law = ActuatorYawFeedback(read(ROOT/spec['feedback_screen']), settings['maximum_yaw_input_rad_s'])
    pending = None
    receipt = []

    class BranchController(Controller):
        def map(self, observed, step, candidate, active):
            nonlocal pending
            assert pending is None
            command = super().map(observed, step, candidate, active)
            features = self.features(observed)
            if candidate:
                ids = np.flatnonzero(self.forward & active)
                if ids.size:
                    assert step > spec['phase']['warmup_controls'] and not command[ids, 2].any()
                    alternate = observed[ids].copy()
                    alternate[:, 48:51] = command[ids]
                    alternate[:, 50] = law.commands(features[ids])
                    raw = actor(rollout.td(torch.from_numpy(alternate).to(env.device)))
                    pending = step, ids, alternate[:, 50], raw
            return command

    def process_once(actions):
        nonlocal pending
        process(actions)
        if pending is None:
            return
        step, ids, yaw, raw = pending
        idx = torch.as_tensor(ids, device=env.device)
        home = term._offset if term._offset.ndim == 1 else term._offset[idx]
        target, scale = projected_feedback_target(term._processed_actions[idx], raw, home, limits,
                                                   settings['maximum_joint_correction_rad'])
        # Keep manager history and the actor's raw request on the original path.
        term._processed_actions[idx] = target
        term.projected_destination[idx] = target
        receipt.append((np.full(len(ids), step), ids, yaw, rollout.cpu(raw), rollout.cpu(scale)))
        pending = None

    try:
        with patch.object(term, 'process_actions', process_once), patch.object(rollout, 'ForwardPhase', BranchController):
            original_collect(env, actor, spec, record, state, desired)
        assert receipt and pending is None
    finally:
        if receipt:
            columns = zip(*receipt, strict=True)
            np.savez_compressed(record/'feedback-readback.npz', **dict(zip(
                ('step', 'world', 'yaw', 'alternate_raw', 'fraction'), map(np.concatenate, columns), strict=True)))


def sample():
    live_publish = pulse.live.publish

    def publish(record, state):
        state['purpose'] = state['purpose'].replace('偏航单步脉冲', '前进执行器反馈')
        frame = record/'evaluation-actuator-feedback/frame-step-0.jpg'
        if frame.is_file():
            state['frame'] = frame.relative_to(record).as_posix()
        live_publish(record, state)

    with patch.object(pulse, 'PLAN', PLAN), patch.object(pulse, 'OUT', OUT), patch.object(pulse, 'PAGE', PAGE), \
         patch.object(pulse, 'STAGE', STAGE), patch.object(pulse, 'ROLES', ROLES), \
         patch.object(rollout, 'collect', collect), patch.object(pulse.live, 'publish', publish):
        spec = pulse.checked()
        try:
            pulse.sample(spec)
        finally:
            readme = ROOT/'records/evaluations'/spec['environment']['run_id']/'README.md'
            if readme.is_file():
                value = readme.read_text(encoding='utf-8')
                readme.write_text(value.replace('前进偏航单步脉冲', '前进执行器反馈').replace(
                    pulse.__name__+'.json', PLAN.name).replace('common-walk-yaw-pulse', OUT.name), encoding='utf-8')


if __name__ == '__main__':
    pulse.check_project()
    assert sys.argv[1:] in (['--freeze'], ['--sample'])
    (freeze if sys.argv[1] == '--freeze' else sample)()
