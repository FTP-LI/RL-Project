"""Compute one bounded neck target correction from the live policy observation."""

import torch


def limit_target(target, previous, observation, eligible, neck, lower, upper, spec):
    gravity = observation[:, 3:6].to(torch.float64)
    x, y, z = gravity.unbind(dim=1)
    lateral = (1. - y.square()).clamp_min(0.).sqrt().clamp_min(1e-6)
    phase = torch.rad2deg(torch.atan2(x, -z / lateral)).remainder(360.)
    a, b, c, d = spec['phase_ramp_deg']
    weight = torch.minimum((phase - a) / (b - a), (d - phase) / (d - c)).clamp(0., 1.)
    enabled = eligible & (observation[:, 1] > 0.) & (y.abs() <= spec['maximum_lateral_gravity_magnitude'])
    original = target[:, neck]
    nominal_jump = original - previous[:, neck]
    addition = (spec['minimum_nominal_target_delta_rad_per_control'] - nominal_jump).clamp(
        0., spec['maximum_candidate_addition_rad'])
    proposed = (original + (addition * weight * enabled).to(original.dtype)).clamp(lower, upper)
    applied = target.clone()
    applied[:, neck] = torch.where(enabled, proposed, original)
    return applied
