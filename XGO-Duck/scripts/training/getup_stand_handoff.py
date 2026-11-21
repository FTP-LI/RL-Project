"""Prospective recovery-to-stand interface; no learning or physical resets.

Why: late neck relief removes contact without completing upright posture.
Use one sensed signal/noise stream, model-specific HOME and separate raw
histories. A one-way observed-tilt/time latch may select the standing model.
From 5s both arms share a projected destination and bounded target slew.
An old out-of-range request is allowed only to approach the legal interval;
this is not an instantaneous hard-bound guarantee or measured servo limit.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
import math
import torch
from getup_model_coordinates import ModelHomeFreezeAction, ModelHomeFreezeActionCfg


NOISE = {'base_ang_vel': .06, 'projected_gravity': .03,
         'joint_pos': .01, 'joint_vel': .25}
SETTINGS = dict(start_seconds=5., tilt_max_deg=45., rate_rad_s=2.)


def apply_common_actor_noise(cfg):
    """Only future configs; never inject noise into an old saved trajectory."""
    cfg = deepcopy(cfg)
    for name, limit in NOISE.items():
        noise = cfg.observations['actor'].terms[name].noise
        assert noise is not None and noise.operation == 'add'
        noise.n_min, noise.n_max = -limit, limit
    return cfg


class HandoffState:
    """State for a single continuous episode batch; reset only at real resets."""
    def __init__(self, source_home, destination_home, num_envs, *, enabled,
                 device='cpu', settings=None):
        self.source_home = torch.as_tensor(source_home, dtype=torch.float32, device=device)
        self.destination_home = torch.as_tensor(destination_home, dtype=torch.float32, device=device)
        assert self.source_home.shape == self.destination_home.shape == (14,)
        self.settings = dict(SETTINGS if settings is None else settings)
        self.enabled = bool(enabled)
        self.shadow_latched = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.standing = torch.zeros_like(self.shadow_latched)
        self.source_history = torch.zeros(num_envs, 14, device=device)
        self.destination_history = torch.zeros_like(self.source_history)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.shadow_latched[ids] = False
        self.standing[ids] = False
        self.source_history[ids] = 0
        self.destination_history[ids] = 0

    def inputs(self, sensed, ages, dt):
        assert sensed.shape == (len(self.standing), 61) and torch.isfinite(sensed).all()
        gravity = sensed[:, 3:6]
        norm = torch.linalg.vector_norm(gravity, dim=-1)
        assert (norm > 1e-8).all()
        aligned = -gravity[:, 2] / norm >= math.cos(math.radians(self.settings['tilt_max_deg']))
        eligible = (ages >= round(self.settings['start_seconds'] / dt)) & aligned
        self.shadow_latched |= eligible
        self.standing = self.shadow_latched & self.enabled
        source, destination = sensed.clone(), sensed.clone()
        source[:, 34:48] = self.source_history
        destination[:, 6:20] += self.source_home - self.destination_home
        destination[:, 34:48] = self.destination_history
        # This component is explicitly zero-command getup -> stand, not walking.
        assert torch.all(sensed[:, 48:].abs() <= 1e-6)
        source[:, 48:] = 0
        destination[:, 48:] = 0
        return source, destination

    def select(self, source_raw, destination_raw, ages, dt):
        assert source_raw.shape == destination_raw.shape == self.source_history.shape
        assert torch.isfinite(source_raw).all() and torch.isfinite(destination_raw).all()
        frozen = ages < round(2. / dt)
        assert not (frozen & self.standing).any()
        selected = torch.where(self.standing[:, None], destination_raw, source_raw).clone()
        selected[frozen] = 0
        source_active = ~self.standing & ~frozen
        self.source_history[source_active] = source_raw[source_active]
        self.destination_history[self.standing] = destination_raw[self.standing]
        self.source_history[frozen] = 0
        self.destination_history[frozen] = 0
        return selected


def projected_slew(previous, desired, limits, dt, rate):
    """Approach a legal destination without increasing any old range excess."""
    assert previous.shape == desired.shape and limits.shape == (desired.shape[1], 2)
    assert torch.isfinite(previous).all() and torch.isfinite(desired).all()
    assert dt > 0 and rate > 0 and (limits[:, 0] < limits[:, 1]).all()
    destination = torch.maximum(torch.minimum(desired, limits[:, 1]), limits[:, 0])
    actual = previous + (destination - previous).clamp(-rate * dt, rate * dt)
    return actual, destination


@dataclass(kw_only=True)
class HandoffActionCfg(ModelHomeFreezeActionCfg):
    destination_home: tuple[float, ...]
    physical_limits: tuple[tuple[float, float], ...]
    transition_start_seconds: float = 5.
    transition_rate_rad_s: float = 2.

    def build(self, env):
        return HandoffAction(self, env)


class HandoffAction(ModelHomeFreezeAction):
    cfg: HandoffActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous_target = torch.zeros_like(self._raw_actions)
        self.previous_valid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.standing_mask = torch.zeros_like(self.previous_valid)
        self.requested_target = torch.zeros_like(self._raw_actions)
        self.projected_destination = torch.zeros_like(self._raw_actions)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, 'previous_valid'):
            ids = slice(None) if env_ids is None else env_ids
            self.previous_valid[ids] = False
            self.standing_mask[ids] = False
            self.previous_target[ids] = 0
            self.requested_target[ids] = 0
            self.projected_destination[ids] = 0

    def process_actions(self, actions):
        super().process_actions(actions)
        age, dt = self._env.episode_length_buf, float(self._env.step_dt)
        standing = self.standing_mask
        assert standing.shape == age.shape
        assert not (standing & (age < round(self.cfg.transition_start_seconds / dt))).any()
        home = torch.as_tensor(self.cfg.destination_home, device=actions.device, dtype=actions.dtype)
        self._processed_actions[standing] = (actions + home)[standing]
        self.requested_target = self._processed_actions.clone()
        enabled = age >= round(self.cfg.transition_start_seconds / dt)
        assert self.previous_valid[enabled].all(), 'Missing previous physical request; do not fabricate a switch state'
        limits = torch.as_tensor(self.cfg.physical_limits, device=actions.device, dtype=actions.dtype)
        actual, projected = projected_slew(self.previous_target, self.requested_target, limits,
                                           dt, self.cfg.transition_rate_rad_s)
        self.projected_destination = projected
        self._processed_actions[enabled] = actual[enabled]
        self.previous_target = self._processed_actions.clone()
        self.previous_valid[:] = True


def adapt_action(cfg, destination_home, limits, settings=None):
    cfg = deepcopy(cfg)
    old = cfg.actions['joint_pos']
    assert type(old) is ModelHomeFreezeActionCfg
    values = {f.name: deepcopy(getattr(old, f.name)) for f in fields(old) if f.init}
    settings = SETTINGS if settings is None else settings
    cfg.actions['joint_pos'] = HandoffActionCfg(
        **values, destination_home=tuple(destination_home),
        physical_limits=tuple(tuple(r) for r in limits),
        transition_start_seconds=settings['start_seconds'],
        transition_rate_rad_s=settings['rate_rad_s'])
    return cfg
