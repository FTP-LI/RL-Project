"""Add parent-action retention to the supine-weighted actor objective.

The earlier optimizer and its evidence remain frozen. This continuation uses
the same finite-horizon advantages, weighting, clipping and Adam path. The new
replay touches only the actor loss and samples independently of physical RNG.
"""
import hashlib

import numpy as np
import torch
from torch.distributions import Normal

from getup_ppo_math import finite_gae, normalize_active_advantages
from getup_supine_ppo import SupineWeightedPPO, weighted_surrogate
from getup_teacher_anchor import anchored_objective, balanced_indices, mean_anchor_loss
from getup_value_bootstrap import td


def retention_indices(cells, seed, minibatch, count):
    # Index the independent stream by the critic minibatch count. Skipped
    # actor updates cannot shift the draw sequence or require another RNG file.
    sampler = torch.Generator(device='cpu').manual_seed(seed + minibatch)
    return balanced_indices(cells, sampler, count)


class RecoveryRetentionPPO(SupineWeightedPPO):
    def __init__(self, *args, retention_replay, retention_weight, retention_seed,
                 retention_samples_per_cell, **kwargs):
        super().__init__(*args, **kwargs)
        observations, targets, self.retention_cells = retention_replay
        device = next(self.actor.parameters()).device
        self.retention_inputs = torch.as_tensor(observations, dtype=torch.float32, device=device).detach().clone()
        self.retention_targets = torch.as_tensor(targets, dtype=torch.float32, device=device).detach().clone()
        assert self.retention_inputs.shape == (900, 61) and self.retention_targets.shape == (900, 14)
        assert len(self.retention_cells) == 24
        assert np.isfinite(retention_weight) and 0. <= retention_weight <= .02
        assert retention_seed > 0 and retention_samples_per_cell > 0
        self.retention_weight = float(retention_weight)
        self.retention_seed = int(retention_seed)
        self.retention_samples_per_cell = int(retention_samples_per_cell)

    def update(self, batch, last_values):
        p = self.plan
        shape = batch['reward'].shape
        device = batch['reward'].device
        focus, active_batch = batch['early_supine'], batch['active']
        assert focus.shape == shape == active_batch.shape and focus.dtype == active_batch.dtype == torch.bool
        assert not (focus & ~active_batch).any()
        with torch.no_grad():
            advantages, returns = finite_gae(batch['reward'], batch['value'], batch['done'], last_values, p['gamma'], p['lambda'])
            advantages = normalize_active_advantages(advantages, active_batch)
        flat = {name: value.reshape((-1,) + value.shape[2:]) for name, value in batch.items()}
        flat['advantage'], flat['returns'] = advantages.flatten(), returns.flatten()
        size = int(np.prod(shape))
        assert size % p['minibatches'] == 0
        before_steps = self.actor_steps
        rows, sampled, retained = [], [], []
        actor_limited = False
        for _ in range(p['epochs']):
            order = torch.randperm(size, generator=self.generator)
            for ids in order.chunk(p['minibatches']):
                ids = ids.to(device)
                active = flat['active'][ids]
                mean = self.actor(td('actor', flat['actor'][ids]))
                log_prob = Normal(mean, p['initial_std']).log_prob(flat['action'][ids]).sum(-1)
                with torch.no_grad():
                    kl = (mean-flat['mean'][ids]).square().sum(-1) / (2*p['initial_std']**2)
                    active_kl = float(kl[active].mean()) if active.any() else 0.
                assert active_kl < p['severe_kl_stop'], 'Per-rollout KL exceeded severe divergence limit'
                if active_kl > p['target_kl']:
                    actor_limited = True
                actor_live = bool(active.any()) and not actor_limited
                policy_loss = weighted_surrogate(log_prob, flat['log_prob'][ids], flat['advantage'][ids],
                                                 active, flat['early_supine'][ids], self.focus_weight, p['clip'])
                replay = balanced_indices(self.anchor_cells, self.anchor_generator, 8)
                sampled.append(replay)
                replay = replay.to(device)
                anchored = self.actor(td('actor', self.anchor_inputs[replay]))
                auxiliary = mean_anchor_loss(anchored, self.anchor_targets[replay], p['initial_std'])
                actor_loss = anchored_objective(policy_loss, auxiliary, self.anchor_weight)

                extra = retention_indices(self.retention_cells, self.retention_seed, self.critic_steps,
                                          self.retention_samples_per_cell)
                retained.append(extra)
                extra = extra.to(device)
                # The zero-coefficient arm sees the same draw. It keeps the
                # original loss graph and the original Adam updates.
                with torch.set_grad_enabled(actor_live and self.retention_weight > 0.):
                    teacher_mean = self.actor(td('actor', self.retention_inputs[extra]))
                    retention = mean_anchor_loss(teacher_mean, self.retention_targets[extra], p['initial_std'])
                actor_loss = anchored_objective(actor_loss, retention, self.retention_weight)

                value = self.critic(td('critic', flat['critic'][ids])).squeeze(-1)
                clipped = flat['value'][ids] + (value-flat['value'][ids]).clamp(-p['clip'], p['clip'])
                value_loss = torch.maximum((value-flat['returns'][ids]).square(),
                                           (clipped-flat['returns'][ids]).square()).mean()
                assert torch.isfinite(actor_loss) and torch.isfinite(value_loss)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), p['gradient_norm'])
                assert torch.isfinite(critic_norm)
                self.critic_optimizer.step()
                self.critic_steps += 1
                actor_norm = 0.
                if actor_live:
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.mlp.parameters(), p['gradient_norm'])
                    assert torch.isfinite(actor_norm)
                    self.actor_optimizer.step()
                    self.actor_steps += 1
                rows.append(dict(policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                                 auxiliary_loss=float(auxiliary.detach()), retention_loss=float(retention.detach()),
                                 kl=active_kl, actor_grad_norm=float(actor_norm), critic_grad_norm=float(critic_norm)))
        self.check_static()
        assert all(torch.isfinite(v).all() for net in (self.actor, self.critic) for v in net.state_dict().values())
        return dict(
            **{name: float(np.mean([row[name] for row in rows])) for name in rows[0]},
            active_policy_fraction=float(active_batch.float().mean()), actor_gradient_steps=self.actor_steps-before_steps,
            actor_steps_this_run=self.actor_steps, critic_steps_this_run=self.critic_steps, actor_kl_limited=actor_limited,
            anchor_weight=self.anchor_weight, anchor_batches=len(sampled),
            anchor_batch_sha256=hashlib.sha256(torch.stack(sampled).numpy().tobytes()).hexdigest(),
            early_supine_samples=int(focus.sum()), focus_weight=self.focus_weight,
            retention_weight=self.retention_weight, retention_batches=len(retained),
            retention_batch_sha256=hashlib.sha256(torch.stack(retained).numpy().tobytes()).hexdigest(),
        )
