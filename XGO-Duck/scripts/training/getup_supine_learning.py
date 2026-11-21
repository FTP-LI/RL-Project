"""Restore the trained parent and classify physical release for a new PPO pair."""
from pathlib import Path

import torch

from getup_anchor_ppo import AnchorPPO
from getup_value_bootstrap import td


def load_parent(path, plan, replay, device, optimizer_type=AnchorPPO):
    """Keep the trained value/actor and both Adam moments, with fresh samplers."""
    from rsl_rl.models import MLPModel

    saved = torch.load(Path(path), map_location='cpu', weights_only=True)
    # Model constructors draw on CPU before loading. Keep the live rollout RNG.
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        actor = MLPModel(
            td('actor', torch.zeros(1, 61)), {'actor': ['actor']}, 'actor', 14,
            hidden_dims=(512, 256, 128), activation='elu', obs_normalization=True,
            distribution_cfg=dict(class_name='GaussianDistribution', init_std=plan['initial_std'], std_type='scalar'),
        )
        critic = MLPModel(
            td('critic', torch.zeros(1, 75)), {'critic': ['critic']}, 'critic', 1,
            hidden_dims=plan['critic_hidden_dims'], activation='elu', obs_normalization=True,
        )
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    critic.load_state_dict(saved['critic_state_dict'], strict=True)
    algorithm = optimizer_type(
        actor.to(device), critic.to(device), saved['critic_optimizer_state_dict'], plan['learner'],
        anchor_inputs=replay[0], anchor_targets=replay[1], anchor_cells=replay[2],
        anchor_weight=plan['anchor_weight'], anchor_seed=plan['anchor_seed'],
    )
    algorithm.actor_optimizer.load_state_dict(saved['actor_optimizer_state_dict'])
    return algorithm, saved


def actual_supine_at_release(quaternion, alignment):
    """Classify the true pose before the first active control, without IMU noise."""
    w, x, y, z = quaternion.to(torch.float64).unbind(dim=-1)
    gx = -2 * (x*z - w*y)
    gy = -2 * (y*z + w*x)
    gz = -(1 - 2 * (x*x + y*y))
    # Match the first-axis tie convention of the saved evaluation scorer.
    return (gx <= -alignment) & (-gx >= torch.maximum(gy.abs(), gz.abs()))


def in_early_supine_window(release_supine, age, start, stop):
    """Latch release once per episode in the runner; the age mask gates resets."""
    return release_supine & (age >= start) & (age < stop)
