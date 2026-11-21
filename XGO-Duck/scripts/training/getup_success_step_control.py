"""Bound a complete PPO update with the original success replay limits."""
from copy import deepcopy
import math

import torch

from getup_recovery_retention_ppo import RecoveryRetentionPPO
from getup_value_bootstrap import td


class ReplayStepRejected(RuntimeError):
    """No allowed actor learning rate satisfied the saved-action limits."""


class SuccessStepControlledPPO(RecoveryRetentionPPO):
    def __init__(self, *args, replay_limits, step_scales=(1., .5, .25), **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_limits = tuple(float(limit) for limit in replay_limits)
        self.step_scales = tuple(float(scale) for scale in step_scales)
        assert len(self.replay_limits) == 2 and all(math.isfinite(limit) and limit > 0 for limit in self.replay_limits)
        assert self.step_scales and self.step_scales[0] == 1.
        assert all(0 < scale <= 1 for scale in self.step_scales)
        assert all(a > b for a, b in zip(self.step_scales, self.step_scales[1:]))

    @torch.no_grad()
    def replay_drift(self):
        error = self.actor(td('actor', self.anchor_inputs))-self.anchor_targets
        return tuple(torch.stack((error.square().mean().sqrt(), error.abs().max())).tolist())

    def _within_limits(self, drift):
        return all(math.isfinite(value) and value <= cap for value, cap in zip(drift, self.replay_limits, strict=True))

    def _snapshot_update(self):
        # These objects carry every persistent mutation made by an update.
        return (
            deepcopy(self.actor.state_dict()), deepcopy(self.critic.state_dict()),
            deepcopy(self.actor_optimizer.state_dict()), deepcopy(self.critic_optimizer.state_dict()),
            self.generator.get_state(), self.anchor_generator.get_state(), self.actor_steps, self.critic_steps,
        )

    def _restore_update(self, snapshot):
        actor, critic, actor_adam, critic_adam, minibatches, anchors, actor_steps, critic_steps = snapshot
        self.actor.load_state_dict(actor)
        self.critic.load_state_dict(critic)
        self.actor_optimizer.load_state_dict(deepcopy(actor_adam))
        self.critic_optimizer.load_state_dict(deepcopy(critic_adam))
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.generator.set_state(minibatches)
        self.anchor_generator.set_state(anchors)
        self.actor_steps, self.critic_steps = actor_steps, critic_steps

    def update(self, batch, last_values):
        start = self.replay_drift()
        if not self._within_limits(start):
            raise ReplayStepRejected(f'Replay drift before the update {start}, limits {self.replay_limits}')
        snapshot = self._snapshot_update()
        nominal_rates = [group['lr'] for group in self.actor_optimizer.param_groups]
        attempts = []
        try:
            for scale in self.step_scales:
                if attempts:
                    # Retry the same saved batch and minibatches, without another rollout.
                    self._restore_update(snapshot)
                for group, rate in zip(self.actor_optimizer.param_groups, nominal_rates, strict=True):
                    group['lr'] = rate*scale
                metrics = super().update(batch, last_values)
                drift = self.replay_drift()
                attempts.append(dict(scale=scale, rms_rad=drift[0], max_rad=drift[1]))
                if self._within_limits(drift):
                    for group, rate in zip(self.actor_optimizer.param_groups, nominal_rates, strict=True):
                        group['lr'] = rate
                    return dict(metrics, actor_step_scale=scale, replay_drift_before_rad=start, replay_step_attempts=attempts)
            raise ReplayStepRejected(f'Replay step exceeded {self.replay_limits} after {attempts}')
        except BaseException:
            # Leave a valid complete checkpoint even if every proposal failed.
            self._restore_update(snapshot)
            raise
