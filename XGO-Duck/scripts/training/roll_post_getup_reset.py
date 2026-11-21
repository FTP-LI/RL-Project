"""Reset selected training worlds from actual standing roll-entry states."""

from pathlib import Path

import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab_microduck.tasks.mdp import _roulade_pitch_angle


ROOT = Path(__file__).resolve().parents[2]


class RollEntryStates:
    def __init__(self, env, bank, stride, joint_names):
        data = env.sim.data
        with np.load(ROOT / bank, allow_pickle=False) as saved:
            self.qpos = torch.as_tensor(saved['local_qpos'], device=env.device, dtype=data.qpos.dtype)
            self.qvel = torch.as_tensor(saved['qvel'], device=env.device, dtype=data.qvel.dtype)
            self.prior_target = torch.as_tensor(saved['prior_applied_target'], device=env.device, dtype=data.qpos.dtype)
        self.joints, names = env.scene['robot'].find_joints(joint_names, preserve_order=True)
        assert list(names) == list(joint_names)
        assert self.qpos.shape[1:] == data.qpos.shape[1:] and self.qvel.shape[1:] == data.qvel.shape[1:]
        assert self.prior_target.shape == (len(self.qpos), len(self.joints))
        self.stride = stride
        self.worlds_with_archive = (env.num_envs + stride - 1) // stride
        self.uses = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
        self.last_case = torch.full_like(self.uses, -1)

    def reset(self, env, env_ids):
        ids = env_ids[env_ids % self.stride == 0]
        # A local counter cycles through the development archive without consuming
        # the RNG draws used by nominal resets, disturbances or observation noise.
        cases = ((ids // self.stride) + self.uses[ids] * self.worlds_with_archive) % len(self.qpos)
        pose = self.qpos[cases]
        pose[:, :3] += env.scene.env_origins[ids]
        env.sim.data.qpos[ids] = pose
        env.sim.data.qvel[ids] = self.qvel[cases]
        # Use the preceding joint target only for the physical actuator reset.
        # The actor still starts with the roll-entry convention of zero history.
        env.scene['robot'].set_joint_position_target(self.prior_target[cases], joint_ids=self.joints, env_ids=ids[:, None])
        # The archived states are upright and start from zero roll progress.
        # Set the angle reference to the state that will actually be integrated.
        env._roulade_prev_pitch[ids] = _roulade_pitch_angle(pose[:, 3:7])
        self.last_case[ids] = cases
        self.uses[ids] += 1


def reset_from_entry_archive(env, env_ids, bank, stride, joint_names):
    if not hasattr(env, 'roll_entry_states'):
        assert stride > 1
        env.roll_entry_states = RollEntryStates(env, bank, stride, joint_names)
    env.roll_entry_states.reset(env, env_ids)


def add_curriculum(cfg, bank, stride, joint_names):
    name = 'roll_post_getup_start'
    assert name not in cfg.events and cfg.events['ground_roulade_reset'].mode == 'reset'
    spawn = cfg.events['set_roulade_state'].params
    assert spawn['standing_prob'] == 1. and spawn['midroll_prob'] == 0.
    cfg.events[name] = EventTermCfg(func=reset_from_entry_archive, mode='reset',
        params=dict(bank=bank, stride=stride, joint_names=joint_names))
