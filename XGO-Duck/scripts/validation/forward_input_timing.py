"""Ramp the tested forward input relative to each world's first forward control."""

import numpy as np

from command_input_mapping import transform


def post_entry_forward_input(command, control, first_forward, delay, ramp, external, internal):
    """Retain the original command for standing, rolling, and other requests."""
    command = np.asarray(command, dtype=np.float32)
    first_forward = np.asarray(first_forward, dtype=np.float32)
    elapsed = np.asarray(control, dtype=np.float32) - first_forward - delay
    portion = np.clip(elapsed/ramp, 0., 1.)
    changed = command.copy()
    original_x = np.float32(external)
    active = ((first_forward >= 0) & (command[..., 0] == original_x)
              & (command[..., 1] == 0) & (command[..., 2] == 0))
    changed[..., 0] += np.where(active, portion*(np.float32(internal)-original_x), 0.)
    return transform(changed)
