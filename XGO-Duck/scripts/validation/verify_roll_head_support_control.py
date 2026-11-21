"""Replay the pulse from saved observations and audit the original actor independently."""

import numpy as np

from roll_stand_walk_sequence import FAILED, ROLL, STAND_LANDING
from verify_continuous_roll_stand_walk import check_policy


def expected_selection(observed, phase, candidate):
    # Reconstruct roll intervals and integrate their measured angle increments.
    x, y, z = np.asarray(observed[..., 3:6], dtype=np.float64).transpose(2, 0, 1)
    angle = np.arctan2(x, -z / np.maximum(np.sqrt(np.maximum(1 - y * y, 0)), 1e-6))
    selected = np.zeros(phase.shape, dtype=bool)
    for world in range(phase.shape[1]):
        rolling = phase[:, world] == ROLL
        edges = np.flatnonzero(np.diff(rolling.astype(np.int8), prepend=0, append=0))
        for start, end in edges.reshape(-1, 2):
            advance = np.diff(angle[start:end, world], prepend=angle[start, world])
            progress = np.cumsum((advance + np.pi) % (2 * np.pi) - np.pi)
            crossing = np.flatnonzero(progress >= np.deg2rad(candidate['entry_degrees']))
            if crossing.size:
                first = start + crossing[0]
                selected[first:min(end, first + candidate['controls']), world] = True
    return selected


def audit(commands, scene, spec, shared, enabled):
    candidate = spec['head_support_pulse']
    phase, actual = commands['phase_used'], commands['applied_targets']
    joint = spec['walking']['joint_names'].index(candidate['joint'])
    lower, upper = np.asarray(spec['walking']['joint_limits'][joint], dtype=np.float32)
    expected_enabled = np.zeros(phase.shape, dtype=bool)
    expected_enabled[shared:] = enabled
    np.testing.assert_array_equal(commands['pulse_enabled'], expected_enabled)
    selected = expected_selection(commands['observed'], phase, candidate)
    np.testing.assert_array_equal(commands['pulse_selected'], selected)
    changed = expected_enabled & selected
    original = commands['pulse_original_head_roll']
    expected = original.copy()
    expected[changed] = np.maximum(original[changed] + np.float32(candidate['target_increment_rad']), lower)
    np.testing.assert_array_equal(actual[..., joint], expected)
    np.testing.assert_array_equal(commands['pulse_applied_delta_rad'], expected - original)
    np.testing.assert_allclose(commands['actual_motor_target'], actual, atol=1e-6, rtol=0)

    # Replay the unchanged actor on the candidate's real inputs. Remove the
    # pulse only for this target check. Recompute any subsequent slew or hold
    # that inherited the original altered target, and audit the actual value.
    neutral = actual.copy()
    neutral[..., joint] = original
    bridge = commands['bridge']
    assert not (bridge & (phase != STAND_LANDING)).any()
    max_step = np.float32(.02 * spec['bridge_slew_rad_s'])
    for k in range(1, len(phase)):
        actual_prev, neutral_prev = actual[k-1, :, joint], neutral[k-1, :, joint]
        inherited = actual_prev != neutral_prev
        hold = inherited & (phase[k] == FAILED)
        np.testing.assert_array_equal(original[k, hold], actual_prev[hold])
        neutral[k, hold, joint] = neutral_prev[hold]
        slew = inherited & bridge[k]
        requested = np.clip(commands['requested_targets'][k, slew, joint], lower, upper)
        new_target = actual_prev[slew] + np.clip(requested - actual_prev[slew], -max_step, max_step)
        np.testing.assert_allclose(original[k, slew], new_target, atol=2e-7, rtol=0)
        neutral[k, slew, joint] = neutral_prev[slew] + np.clip(requested - neutral_prev[slew], -max_step, max_step)
    replay = check_policy(dict(commands, applied_targets=neutral, actual_motor_target=neutral), scene, spec, phase)
    return dict(replay, pulse_selected_controls=int(selected.sum()), pulse_applied_controls=int(changed.sum()),
                maximum_pulse_rad=float(np.max(np.abs(expected - original), initial=0.)))
