"""Capture and compare real first-action inputs for the head-command pilot.

The neck command changes the policy input, so identical actions are NOT a valid
matching requirement. Match the physical state, random state and all other
observation columns; verify the declared neck scaling and common action noise.
This inspects tensors only, without drawing randomness or advancing simulation.
"""
from pathlib import Path
import numpy as np


def capture(runner, obs):
    import torch
    env = runner.env.unwrapped
    packet = {f'obs_{k}': v.detach().cpu().numpy().copy() for k, v in obs.items()}
    packet['rng_cpu'] = torch.get_rng_state().numpy().copy()
    packet['rng_cuda'] = torch.cuda.get_rng_state(runner.device).cpu().numpy().copy()
    for name in ('qpos', 'qvel', 'ctrl'):
        packet['physical_'+name] = getattr(env.sim.data, name).detach().cpu().numpy().copy()
    packet['episode_length'] = env.episode_length_buf.detach().cpu().numpy().copy()
    robot = env.scene['robot']
    packet['default_joint_pos'] = robot.data.default_joint_pos.detach().cpu().numpy().copy()
    layout = {}
    manager = env.observation_manager
    for group, names in manager.active_terms.items():
        offset = 0
        for name, shape in zip(names, manager.group_obs_term_dim[group], strict=True):
            size = int(np.prod(shape))
            if name == 'head_command':
                assert size == 4
                layout[group] = [offset, offset+size]
            offset += size
        assert packet['obs_'+group].shape[1] == offset
    assert set(layout) == set(obs.keys())
    return packet, layout


def compare(control: Path, candidate: Path, control_layout, candidate_layout, ratio):
    assert control_layout == candidate_layout
    with np.load(control, allow_pickle=False) as a, np.load(candidate, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        deltas = {}
        for key in a.files:
            assert np.isfinite(a[key]).all() and np.isfinite(b[key]).all(), key
            if key.startswith('obs_'):
                start, end = control_layout[key[4:]]
                np.testing.assert_array_equal(a[key][:, :start], b[key][:, :start])
                np.testing.assert_array_equal(a[key][:, start+1:], b[key][:, start+1:])
                np.testing.assert_allclose(b[key][:, start], a[key][:, start]*ratio, atol=2e-7, rtol=2e-6)
                deltas[key] = float(np.max(np.abs(a[key]-b[key])))
            elif key in ('actions', 'distribution_mean'):
                continue  # The policy sees a legitimately changed neck command.
            else:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        noise_a = (a['actions']-a['distribution_mean'])/a['distribution_std']
        noise_b = (b['actions']-b['distribution_mean'])/b['distribution_std']
        np.testing.assert_allclose(noise_a, noise_b, atol=5e-5, rtol=5e-5)
        return dict(status='passed', physical_state_and_rng_equal=True,
            only_neck_observation_scaled=True, neck_command_ratio=ratio,
            same_first_action_standard_normal_noise=True,
            observation_max_absolute_difference=deltas,
            max_action_noise_difference=float(np.max(np.abs(noise_a-noise_b))),
            identical_actions_required=False)
