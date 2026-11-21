"""Latch the recovery policy once, from the first released observation."""
import torch


def right_side_mask(gravity, alignment):
    norm = torch.linalg.vector_norm(gravity, dim=-1, keepdim=True)
    valid = torch.isfinite(gravity).all(dim=-1) & (norm[:, 0] > 0)
    direction = gravity / torch.where(valid[:, None], norm, torch.ones_like(norm))
    axis = direction.abs().argmax(dim=-1)
    return valid & (axis == 1) & (direction[:, 1] <= -alignment)


class GravityRoutedGetup(torch.nn.Module):
    """Choose one frozen actor per environment for one recovery episode."""

    def __init__(self, parent, right_side, alignment):
        super().__init__()
        self.parent = parent
        self.right_side = right_side
        self.alignment = alignment
        self.reset_episode()

    def reset_episode(self):
        self.first_gravity = None
        self.right = None
        self.calls = 0

    def forward(self, observation):
        gravity = observation['actor'][:, 3:6]
        if self.right is None:
            self.first_gravity = gravity.detach().clone()
            self.right = right_side_mask(self.first_gravity, self.alignment)
        assert self.right.shape == (len(gravity),)
        self.calls += 1
        return torch.where(self.right[:, None], self.right_side(observation), self.parent(observation))
