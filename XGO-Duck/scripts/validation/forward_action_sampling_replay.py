"""Isolate training-style exploration inside the official-stance evaluation."""

import numpy as np
import torch

from check_forward_teacher_actor_readback import infer, session
from command_input_mapping import transform
from evaluate_forward_official_stance import OfficialStandControl
from forward_low_friction_yaw_reward import target_with_exploration
from roll_stand_walk_sequence import FAILED, FORWARD, STAND_START
from run_continuous_roll_stand_walk import cpu
from run_roll_stand_forward_diagnostic import ROOT, read, require
from walking_actuator_yaw_feedback import ActuatorYawFeedback


def noise_schedule(plan):
    sampling = plan['action_sampling']
    require(np.__version__ == sampling['numpy_version'], 'The frozen NumPy sampling environment changed')
    shape = (plan['forward_controls'], plan['worlds'], 14)
    if not sampling['enabled']:
        return np.zeros(shape, dtype=np.float32)
    generator = np.random.Generator(np.random.PCG64(sampling['seed']))
    return generator.standard_normal(shape, dtype=np.float32) * np.asarray(sampling['std_rad'], np.float32)


class ForwardSamplingControl(OfficialStandControl):
    def __init__(self, plan, networks, homes, sim_home, limits):
        super().__init__(plan, networks, homes, sim_home, limits)
        self.exploration = noise_schedule(plan)

    def next(self, step, observations, sequence):
        applied, commands = super().next(step, observations, sequence)
        sampled = commands['raw_policy_actions']
        mean = sampled.copy()
        commands['mean_policy_actions'] = mean
        forward = commands['phase_used'] == FORWARD
        if self.p['action_sampling']['enabled'] and forward.any():
            index = step - self.p['initial_stand_controls'] - 1
            require(0 <= index < len(self.exploration), 'Exploration outside the frozen forward window')
            sampled[forward] += self.exploration[index, forward]
            actual = torch.as_tensor(sampled[forward], device=observations.device)
            baseline = torch.as_tensor(mean[forward], device=observations.device)
            alternate = torch.as_tensor(commands['alternate_action'][forward], device=observations.device)
            target = target_with_exploration(actual, baseline, alternate, self.home['walk'], self.limits,
                                             self.p['feedback']['maximum_joint_correction_rad'])
            applied[forward] = target
            commands['applied_targets'][forward] = cpu(target)
            # PPO feeds the sampled action back to the next actor input.
            self.history[forward] = actual
            self.previous[forward] = target
        return applied, commands


def replay_policy(commands, scene, plan, phase):
    """Independently replay all physical targets and sampled ONNX calls on CPU."""
    require(np.isin(phase, (STAND_START, FORWARD, FAILED)).all(), 'Unexpected policy phase')
    require(not commands['bridge'].any() and not commands['history_reset'].any(), 'Unexpected policy boundary')
    stand, forward = phase == STAND_START, phase == FORWARD
    active = phase != FAILED
    start = plan['initial_stand_controls']
    require(not forward[:start].any() and not stand[start:].any(), 'Unexpected forward entry time')

    inputs, observed = commands['actor_input'], commands['observed']
    mean, raw = commands['mean_policy_actions'], commands['raw_policy_actions']
    expected_commands = np.zeros_like(commands['external_command'])
    expected_commands[forward] = plan['walking']['commands']['forward']
    np.testing.assert_array_equal(commands['external_command'], expected_commands)
    expected_inputs = observed.copy()
    home = scene['walk_home']
    expected_inputs[..., 6:20] += scene['simulation_home'] - home
    expected_inputs[0, :, 34:48] = 0.
    expected_inputs[1:, :, 34:48] = raw[:-1]
    expected_inputs[..., 48:51] = transform(expected_commands)
    np.testing.assert_array_equal(inputs[active], expected_inputs[active])
    np.testing.assert_array_equal(inputs[~active], 0.)

    expected_raw = mean.copy()
    if len(phase) > start:
        perturbation = noise_schedule(plan)[:len(phase) - start]
        expected_raw[start:] += np.where(forward[start:, :, None], perturbation, 0.)
    np.testing.assert_array_equal(raw, expected_raw)
    np.testing.assert_array_equal(mean[~active], 0.)
    np.testing.assert_array_equal(commands['requested_targets'][active], (mean + home)[active])

    hip = [20 + plan['walking']['joint_names'].index(f'{side}_hip_pitch') for side in ('left', 'right')]
    features = np.stack((observed[..., 2], observed[..., hip].sum(axis=-1)), axis=-1)
    changes = np.diff(features, axis=0, prepend=features[:1])
    feedback = ActuatorYawFeedback(read(ROOT / plan['feedback_screen']), plan['feedback']['maximum_yaw_input_rad_s'])
    manual_feedback = feedback.commands(np.concatenate((features[forward], changes[forward]), axis=1))
    np.testing.assert_allclose(commands['feedback_yaw'][forward], manual_feedback, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(commands['feedback_yaw'][~forward], 0.)
    np.testing.assert_array_equal(commands['alternate_action'][~forward], 0.)

    low, high = np.asarray(plan['walking']['joint_limits'], dtype=np.float32).T
    target = np.clip(commands['requested_targets'], low, high)
    arm = np.clip(commands['alternate_action'][forward] + home, low, high)
    shift = arm - target[forward]
    divisor = np.maximum(np.abs(shift).max(axis=-1, keepdims=True), np.float32(1e-8))
    mix = np.minimum(1., np.float32(plan['feedback']['maximum_joint_correction_rad']) / divisor)
    target[forward] = np.clip(target[forward] + shift * mix + (raw[forward] - mean[forward]), low, high)
    previous = np.concatenate((scene['simulation_home'][None], commands['applied_targets'][:-1]))
    target[~active] = previous[~active]
    np.testing.assert_allclose(commands['applied_targets'], target, atol=2e-7, rtol=0)
    np.testing.assert_allclose(commands['actual_motor_target'], target, atol=1e-6, rtol=0)

    counts = dict(official_stand=0, learned_forward=0)
    for label, mask, actor in (('official_stand', stand, plan['official_stand']),
                               ('learned_forward', forward, plan['walking'])):
        locations = np.argwhere(mask)
        count = min(1024, len(locations))
        if not count:
            continue
        controls, worlds = locations[np.linspace(0, len(locations) - 1, count, dtype=int)].T
        selected = inputs[controls, worlds]
        runtime = session(ROOT / actor['model'])
        np.testing.assert_allclose(mean[controls, worlds], infer(runtime, selected), atol=2e-5, rtol=2e-5)
        if label == 'learned_forward':
            selected[:, 50] = commands['feedback_yaw'][controls, worlds]
            np.testing.assert_allclose(commands['alternate_action'][controls, worlds], infer(runtime, selected),
                                       atol=2e-5, rtol=2e-5)
        counts[label] = count
    return dict(actual_inputs_replayed=sum(counts.values()), model_counts=counts, raw_history_exact=True,
                sampled_actions_exact=True, control_targets_verified=True,
                explored_actions=int(np.count_nonzero((raw != mean).any(axis=-1))))
