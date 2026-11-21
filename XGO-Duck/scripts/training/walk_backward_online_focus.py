"""Weight backward PPO samples from stalls observed in the current native episode."""

import torch


class BackwardStallFocus:
    def __init__(self, num_envs, block_controls, maximum_speed, stop_control, multiplier, device):
        assert num_envs > 0 and block_controls > 0 and maximum_speed >= 0
        assert stop_control > 2 * block_controls and multiplier > 1
        self.block_controls = block_controls
        self.maximum_speed = maximum_speed
        self.stop_control = stop_control
        self.multiplier = multiplier
        self.sums = torch.zeros((2, num_envs), device=device, dtype=torch.float64)
        self.counts = torch.zeros_like(self.sums, dtype=torch.int64)
        self.tagged = torch.zeros(num_envs, device=device, dtype=torch.bool)

    @torch.no_grad()
    def observe(self, forward_velocity, completed_age, ended):
        """Observe the result of one control, excluding any autoreset row."""
        assert forward_velocity.shape == completed_age.shape == ended.shape == self.tagged.shape
        assert ended.dtype == torch.bool
        clear = ended | (completed_age == 1)
        self.sums.masked_fill_(clear.unsqueeze(0), 0)
        self.counts.masked_fill_(clear.unsqueeze(0), 0)
        self.tagged.masked_fill_(clear, False)

        block = self.block_controls
        valid = ~ended & torch.isfinite(forward_velocity)
        first = valid & (completed_age > 0) & (completed_age <= block)
        second = valid & (completed_age > block) & (completed_age <= 2 * block)
        included = torch.stack((first, second))
        self.sums.add_(torch.where(included, forward_velocity.to(self.sums.dtype), 0))
        self.counts.add_(included)
        complete = (self.counts == block).all(dim=0)
        near_static = (self.sums.abs() <= block * self.maximum_speed).all(dim=0)
        self.tagged.logical_or_((completed_age == 2 * block) & ~ended & complete & near_static)

    def actor_weights(self, age_before_action, focused):
        """With a 50-control block, the first eligible action is control 101."""
        assert age_before_action.shape == self.tagged.shape
        eligible = self.tagged & (age_before_action >= 2 * self.block_controls)
        eligible &= (age_before_action < self.stop_control) & focused
        return torch.where(eligible, self.multiplier, 1.0)


def weighted_surrogate(log_prob, old_log_prob, advantages, active, weights, clip=0.2):
    """Preserve PPO clipping and normalize by the sum of active actor weights."""
    assert log_prob.shape == old_log_prob.shape == advantages.shape == active.shape == weights.shape
    assert active.dtype == torch.bool
    delta = torch.where(active, log_prob, 0) - torch.where(active, old_log_prob, 0)
    advantage = torch.where(active, advantages, 0)
    ratio = torch.exp(delta)
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    effective_weight = torch.where(active, weights, 0)
    return -(surrogate * effective_weight).sum() / effective_weight.sum().clamp_min(1)
