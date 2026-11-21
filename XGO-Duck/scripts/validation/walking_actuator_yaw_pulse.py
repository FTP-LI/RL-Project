"""Apply a bounded diagnostic target while keeping raw walking action history."""
import torch

from common_walk_targets import ProjectedWalkAction


def apply_projected_pulse(term, alternate_raw, enabled, maximum_change):
    assert isinstance(term, ProjectedWalkAction)
    assert term.cfg.scale == 1. and not term.cfg.use_default_offset
    assert alternate_raw.shape == term._processed_actions.shape
    assert enabled.dtype == torch.bool and enabled.shape == alternate_raw.shape[:1]
    assert 0. < maximum_change and torch.isfinite(alternate_raw).all()
    limits = torch.as_tensor(term.cfg.physical_limits, device=alternate_raw.device, dtype=alternate_raw.dtype)
    alternate = torch.clamp(alternate_raw+term._offset, min=limits[:, 0], max=limits[:, 1])
    target = torch.where(enabled[:, None], alternate, term._processed_actions)
    assert (target-term._processed_actions).abs().max() <= maximum_change
    # The policy and manager keep the unmodified raw and requested actions.
    term._processed_actions = target
    term.projected_destination = target.clone()
