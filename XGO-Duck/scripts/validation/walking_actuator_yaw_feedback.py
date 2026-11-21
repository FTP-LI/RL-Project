"""Bound a forward yaw correction without modifying the policy action history."""
import numpy as np
import torch


class ActuatorYawFeedback:
    def __init__(self, screen, cap):
        models = screen['forecast_models']
        self.center = np.asarray(models[0]['means'])
        self.scale = np.asarray(models[0]['scales'])
        assert all(np.array_equal(m['means'], self.center) and np.array_equal(m['scales'], self.scale)
                   and m['columns'] == [0, 1, 2, 3] for m in models)
        self.weights = np.asarray([m['weights'] for m in models]).T
        self.bias = np.asarray([m['bias'] for m in models])
        self.kernel = np.asarray(screen['kernel'])
        self.gradient_scale = screen['gradient_scale']
        self.cap = cap

    def commands(self, features):
        signs = np.clip(((features-self.center)/self.scale)@self.weights+self.bias, -1., 1.)
        gradient = signs@self.kernel
        return -self.cap*np.clip(gradient/self.gradient_scale, -1., 1.)


def projected_feedback_target(base, alternate_raw, home, limits, maximum_change):
    proposal = torch.clamp(alternate_raw+home, min=limits[:, 0], max=limits[:, 1])
    change = proposal-base
    largest = change.abs().amax(dim=-1, keepdim=True)
    # Scale the whole correction vector. Both endpoints are inside the same
    # joint envelope, so the interpolation preserves that envelope.
    fraction = torch.minimum(torch.ones_like(largest), maximum_change/largest.clamp_min(1e-8))
    return base+change*fraction, fraction.squeeze(-1)
