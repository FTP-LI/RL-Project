"""Shared forward control and the single prospective reward change.

The objective penalizes the same physical body-z velocity measured by the
20-second score. Feedback stays on the device so large batches do not transfer
policy inputs through the host on every step.
"""

import torch

from walking_actuator_yaw_feedback import ActuatorYawFeedback, projected_feedback_target


def physical_yaw_cost(yaw, cap):
    return yaw.abs().clamp(max=cap)


def forward_yaw_cost(env, cap):
    return physical_yaw_cost(env.scene["robot"].data.root_link_ang_vel_b[:, 2], cap)


class ForwardFeedback:
    def __init__(self, screen, cap, device):
        reference = ActuatorYawFeedback(screen, cap)
        self.center = self.tensor(reference.center, device)
        self.scale = self.tensor(reference.scale, device)
        self.weights = self.tensor(reference.weights, device)
        self.bias = self.tensor(reference.bias, device)
        self.kernel = self.tensor(reference.kernel, device)
        self.gradient_scale = reference.gradient_scale
        self.cap = cap

    @staticmethod
    def tensor(values, device):
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    def commands(self, features, changes):
        inputs = torch.cat((features, changes), dim=-1)
        signs = (((inputs - self.center) / self.scale) @ self.weights + self.bias).clamp(-1., 1.)
        return -self.cap * ((signs @ self.kernel) / self.gradient_scale).clamp(-1., 1.)


def target_with_exploration(raw, mean, alternate, home, limits, correction_cap):
    """Apply the deterministic feedback correction and then bound exploration."""
    lower, upper = limits.T
    base = (mean + home).clamp(min=lower, max=upper)
    corrected, _ = projected_feedback_target(base, alternate, home, limits, correction_cap)
    # For a deterministic action this preserves the existing projected target,
    # including when the unprojected mean already exceeds a joint range.
    return (corrected + (raw - mean)).clamp(min=lower, max=upper)
