"""Compute the original bounded roll progress payment from endpoint geometry."""

import torch


def advance_progress(geometric, supported, frontier, increment, lateral_axis_z, ground_contact, *, step_dt, target_angle, max_paid_rate):
    # Use the original sagittal weighting and keep every reversal in both
    # accumulators. A positive displacement only qualifies when supported.
    flatness = ((.866-lateral_axis_z.abs())/(.866-.5)).clamp(0.,1.)
    positive_weight = ground_contact*flatness.square()*(3.-2.*flatness)
    next_geometric = geometric+increment
    next_supported = supported+torch.where(increment<0.,increment,increment*positive_weight)
    next_frontier = torch.maximum(frontier,torch.minimum(next_geometric,next_supported))
    new_paid = (next_frontier.clamp(max=target_angle)-frontier.clamp(max=target_angle)).clamp(0.,max_paid_rate*step_dt)
    # RewardManager applies the configured weight and the control period.
    # This amount includes the period, ready for a post-step replacement.
    return next_geometric,next_supported,next_frontier,new_paid/target_angle
