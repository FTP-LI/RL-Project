"""Use the normal post-forward contact reading for one PPO reward term."""

import torch
from mjlab_microduck.tasks import mdp

from roll_contact_endpoint import found_after_forward
from roll_endpoint_progress import advance_progress


class EndpointRollProgress:
    """Preserve installed roll geometry, and replace only the contact reading."""

    name = 'roulade_progress'

    def __init__(self, env, settings, *, verify_cached=False):
        self.env = env
        self.manager = env.reward_manager
        self.index = self.manager._term_names.index(self.name)
        term = self.manager._term_cfgs[self.index]
        self.weight = settings['weight']
        self.knobs = dict(step_dt=settings['control_dt_s'], target_angle=settings['target_angle_rad'],
                          max_paid_rate=settings['max_paid_rate_rad_s'])
        if (term.func is not mdp.roulade_progress or term.weight != self.weight or not self.manager._scale_by_dt
            or term.params['target_angle'] != self.knobs['target_angle']
            or term.params['max_paid_rate'] != self.knobs['max_paid_rate']
            or env.step_dt != self.knobs['step_dt'] or not env.cfg.auto_reset
            or set(term.params) != {'target_angle', 'max_paid_rate'}):
            raise RuntimeError('Installed roll reward differs from the endpoint training protocol')
        self.asset = env.scene[mdp._DEFAULT_ASSET_CFG.name]
        self.sensor = env.scene.sensors[mdp._ROULADE_SUPPORT_SENSOR]
        self.install_compute = self.manager.compute
        self.verify_cached = verify_cached
        self.state = self._start_counters()
        self.cached_state = self._start_counters() if verify_cached else None
        self.last_check = None
        self.manager.compute = self._capture_original_geometry

    def _start_counters(self):
        # The installed reset sets geometry, support and frontier to the same
        # starting pitch, including when the reverse curriculum spawns mid-roll.
        spawn = self.env._roulade_accum
        return (spawn.clone(), spawn.clone(), spawn.clone())

    def _capture_original_geometry(self, dt):
        rewards = self.install_compute(dt=dt)
        current_pitch = self.env._roulade_prev_pitch
        self.increment = torch.nan_to_num(mdp.wrap_to_pi(current_pitch - self.previous_pitch), nan=0.0)
        quaternion = self.asset.data.root_link_quat_w
        self.lateral = torch.nan_to_num(mdp._lateral_axis_z(quaternion), nan=1.0)
        if self.verify_cached:
            self.cached_contact = mdp._sensor_any_contact(self.env, mdp._ROULADE_SUPPORT_SENSOR)
        return rewards

    def _advance(self, state, contact, done):
        next_geometry, next_support, next_frontier, amount = advance_progress(
            *state, self.increment, self.lateral, contact, **self.knobs)
        spawn = self.env._roulade_accum
        state = tuple(torch.where(done, spawn, value)
                      for value in (next_geometry, next_support, next_frontier))
        return state, amount * self.weight

    def step(self, original_step, actions):
        self.previous_pitch = self.env._roulade_prev_pitch.clone()
        self.increment = None
        transition = original_step(actions)
        if self.increment is None:
            raise RuntimeError('Original progress geometry was not captured during the control step')

        _, reward, reset, _ = transition
        done = reset.bool()
        term_rate = self.manager._step_reward[:, self.index]
        step_dt = self.knobs['step_dt']
        original = term_rate * step_dt
        contact = found_after_forward(self.sensor, 1).squeeze(1)
        self.state, proposed = self._advance(self.state, contact, done)
        # Auto-reset overwrites the ended scene's endpoint before the normal
        # forward call. Its current reading belongs to the new episode.
        difference = torch.where(done, 0., proposed - original)
        if self.verify_cached:
            self.cached_state, cached = self._advance(self.cached_state, self.cached_contact, done)
            self.last_check = dict(original_reward=reward.clone(), original_payment=original, cached_payment=cached,
                                   replacement=proposed, adjustment=difference, contact_changed=contact != self.cached_contact,
                                   reset=done, original_terms=self.manager._step_reward.clone(),
                                   original_episode_sum=self.manager._episode_sums[self.name].clone())

        # The manager returns its own reward buffer. Keep the episode sum and
        # unscaled per-step display for this term aligned with what PPO receives.
        reward.add_(difference)
        term_rate.add_(difference / step_dt)
        self.manager._episode_sums[self.name].add_(difference)
        return transition

    def close(self):
        self.manager.compute = self.install_compute
