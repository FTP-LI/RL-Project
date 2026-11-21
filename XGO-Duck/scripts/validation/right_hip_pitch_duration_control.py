"""Continue an already-applied right-support pitch pulse within the existing envelope."""

import numpy as np

from roll_stand_walk_sequence import FORWARD
from run_roll_stand_forward_diagnostic import require


SCHEDULE_FIELDS = ('previous_label', 'last_known_support', 'age', 'fired')
PHASE_FIELDS = ('phase', 'dwell', 'stand_at', 'forward_at', 'phase_changed')


def schedule_state(schedule):
    return {name: getattr(schedule, name).copy() for name in SCHEDULE_FIELDS}


def targets(schedule, design, k, command, observed, extending=None):
    requested, parent = command['requested_targets'], command['applied_targets']
    forward = command['phase_used'] == FORWARD
    target, onset = schedule.apply(k, requested, parent, observed, forward)
    # Advance the previously verified onset rule before testing the current
    # observation's support age. The late correction uses its own time window.
    start_age, end_age = design['recognized_right_support_age_controls']
    joints = schedule.headroom_joints[1]
    nominal = np.clip(requested[:, joints], schedule.lower[joints], schedule.upper[joints])
    minimum = np.maximum(schedule.lower[joints], nominal-schedule.cap)
    maximum = np.minimum(schedule.upper[joints], nominal+schedule.cap)
    delta = design['single_control_delta_rad']
    margin = ((parent[:, joints]-delta >= minimum) & (parent[:, joints]+delta <= maximum)).all(axis=1)
    eligible = ((k >= design['first_zero_based_control']) & (schedule.previous_label == 2) & schedule.fired
                & (schedule.age >= start_age) & (schedule.age <= end_age) & (observed[:, schedule.gyro] > 0) & margin)
    extra = np.zeros_like(eligible) if extending is None else eligible & extending
    if extra.any():
        require(not onset[extra].any(), 'The onset and continuation would stack on one control')
        target[extra, schedule.pitch[1]] += np.float32(delta)
    detail = dict(hip_pitch_pulse_side=onset, right_extension_eligible=eligible, right_extension_applied=extra,
                  observer_label=schedule.previous_label.copy(), observer_age=schedule.age.copy(), onset_latch=schedule.fired.copy())
    return target, detail
