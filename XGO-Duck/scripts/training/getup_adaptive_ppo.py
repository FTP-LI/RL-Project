"""Connect adaptive success replay to the existing transactional PPO update."""
from time import perf_counter
from unittest.mock import patch

import torch

import getup_recovery_retention_ppo as retention
from getup_adaptive_replay import adaptive_replay
from getup_success_step_control import SuccessStepControlledPPO


class AdaptiveGetupPPO(SuccessStepControlledPPO):
    def __init__(self, *args, adaptive_settings, reward_scale, check_active, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_loss = adaptive_replay(self, adaptive_settings)
        self.reward_scale = reward_scale
        self.check_active = check_active
        self.last_update_end = None
        self.total_controls = 0

    def update(self, batch, last_values):
        device = batch['reward'].device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = perf_counter()
        objective = retention.anchored_objective
        calls = 0

        def add_adaptive(policy, auxiliary, weight):
            nonlocal calls
            # The frozen update invokes this first for official replay and
            # second for parent replay. Apply the term only to the former.
            if calls % 2 == 0:
                self.check_active()
                assert weight == self.anchor_weight
                auxiliary = auxiliary+self.adaptive_loss(self.plan['initial_std'])
            calls += 1
            return objective(policy, auxiliary, weight)

        with patch.object(retention, 'anchored_objective', add_adaptive):
            metrics = super().update(batch, last_values)
        assert calls == 2*self.plan['epochs']*self.plan['minibatches']*len(metrics['replay_step_attempts'])
        controls, worlds = batch['reward'].shape
        end = self.total_controls+controls
        hold = sum(step % 400 >= 350 for step in range(self.total_controls, end))*worlds
        self.total_controls = end
        metrics.update(environment_reward_mean=float(batch['reward'].mean())*self.reward_scale,
                       active_policy_samples=int(batch['active'].sum()), final_hold_samples=hold,
                       completed_episodes=int(batch['done'].sum()))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        finished = perf_counter()
        metrics.update(ppo_seconds=finished-started,
                       between_updates_seconds=started-self.last_update_end if self.last_update_end is not None else None)
        self.last_update_end = finished
        return metrics
