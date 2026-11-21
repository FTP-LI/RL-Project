"""First-episode targets and a normalized critic with raw-value output."""

import numpy as np
import torch
from torch import nn

from walk_backward_ppo import make_value_model


def episode_returns(rewards, done, gamma):
    assert rewards.shape == done.shape and done.dtype == np.bool_
    assert (done.sum(axis=0) == 1).all() and np.isfinite(rewards).all()
    terminal = done.argmax(axis=0)
    valid = np.arange(len(rewards))[:, None] <= terminal
    assert not rewards[~valid].any()
    carry = np.zeros(rewards.shape[1], dtype=np.float64)
    result = np.empty(rewards.shape, dtype=np.float32)
    for i in reversed(range(len(rewards))):
        carry = rewards[i] + gamma * carry * ~done[i]
        result[i] = carry
    np.testing.assert_array_equal(result[done], rewards[done])
    return result, valid


def world_split(label, validation_fraction, seed):
    rng = np.random.default_rng(seed)
    train, valid = [], []
    for group in (False, True):
        ids = rng.permutation(np.flatnonzero(label == group))
        count = round(len(ids) * validation_fraction)
        train.append(ids[count:])
        valid.append(ids[:count])
    return tuple(np.sort(np.concatenate(split)) for split in (train, valid))


class SharedBackwardValue(nn.Module):
    def __init__(self, seed, device):
        super().__init__()
        self.network = make_value_model(seed, device)
        self.register_buffer("input_center", torch.zeros(77, device=device))
        self.register_buffer("input_scale", torch.ones(77, device=device))
        self.register_buffer("reward_scale", torch.ones((), device=device))

    def normalized(self, observations):
        return self.network((observations - self.input_center) / self.input_scale)

    def forward(self, observations):
        return self.normalized(observations) * self.reward_scale
