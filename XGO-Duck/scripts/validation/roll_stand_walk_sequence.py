"""Track a stand, roll, stand and forward sequence in one physics episode."""
import numpy as np

from planar_block_metrics import block_statistic
from roll_landing_latch import RollLandingLatch
from roll_metrics import rotation


FAILED, STAND_START, ROLL, STAND_LANDING, FORWARD = range(-1, 4)
STATE_FIELDS = ('active', 'quaternion', 'head_quaternion', 'height', 'angular_velocity',
                'ground_contact', 'head_contact', 'foot_contact', 'head_penetration_m', 'joint_envelope_ok')


class Sequence:
    def __init__(self, plan):
        self.plan = plan
        self.criteria = plan['roll']['criteria']
        count = plan['worlds']
        self.phase = np.zeros(count, dtype=np.int8)
        self.roll_begin = plan['initial_stand_controls']
        self.roll_end = self.roll_begin + plan['roll_controls']
        self.roll_latch = RollLandingLatch(self.criteria, count)
        self.dwell = np.zeros(count, dtype=np.int32)
        self.stand_at = np.full(count, -1, dtype=np.int32)
        self.forward_at = self.stand_at.copy()
        self.phase_changed = self.stand_at.copy()

    def stable(self, state):
        r = rotation(state['quaternion'])
        c = self.criteria
        return (state['active'] & state['foot_contact'].all(axis=-1) & state['joint_envelope_ok']
                & (r[:, 2, 2] >= np.cos(np.deg2rad(c['tilt_max_deg'])))
                & (abs(state['height'] - c['nominal_height_m']) <= c['height_tolerance_m'])
                & (np.linalg.norm(state['angular_velocity'], axis=-1) <= c['angular_speed_max_rad_s']))

    def update(self, step, state):
        old = self.phase.copy()
        self.phase[~state['active']] = FAILED
        stand = np.isin(self.phase, (STAND_START, STAND_LANDING))
        self.dwell = np.where(stand & self.stable(state), self.dwell + 1, 0)
        ready = self.dwell >= self.plan['standing_dwell_controls']
        if step == self.roll_begin:
            waiting = self.phase == STAND_START
            self.phase[waiting] = np.where(ready[waiting], ROLL, FAILED)
        landed = ((self.phase == STAND_LANDING) &
                  (step == self.stand_at + self.plan['landing_stand_controls']))
        self.phase[landed] = np.where(ready[landed], FORWARD, FAILED)
        self.forward_at[landed & ready] = step
        if self.roll_begin <= step <= self.roll_end:
            rolling = self.phase == ROLL
            fire = self.roll_latch.advance(dict(state, active=state['active'] & rolling))
            self.stand_at[fire] = step
            self.phase[fire] = STAND_LANDING
            if step == self.roll_end:
                self.phase[self.phase == ROLL] = FAILED
        changed = self.phase != old
        self.phase_changed[changed] = step
        self.dwell[changed] = 0
        return changed


def score_forward(trace, first, plan):
    """Keep the original forward gates, including the full 20-second window."""
    walk = plan['walking']
    criteria = walk['criteria']
    command = np.asarray(walk['commands']['forward'])
    duration = plan['forward_controls']
    result = []
    for world, begin in enumerate(first):
        if begin < 0:
            result.append(dict(world=world, entered_forward=False, success=False, failed_checks=['no-forward-entry']))
            continue
        stop = begin + duration + 1
        assert stop <= len(trace['active'])
        part = slice(begin+1, stop)
        active = trace['active'][part, world]
        velocity = trace['velocity'][part, world]
        block = block_statistic(velocity[:, None], active[:, None], command, criteria['block_steps'],
                                criteria['warmup_steps'], criteria['missing_block_penalty'])
        joints = trace['joint_positions'][part, world]
        contacts = trace['foot_contact'][part, world] & active[:, None]
        rotation_matrix = rotation(trace['quaternion'][part, world])
        separation = ((trace['foot_positions'][part, world, 0] - trace['foot_positions'][part, world, 1])
                      * rotation_matrix[:, :, 1]).sum(axis=1)
        hip = np.rad2deg(joints[:, [walk['joint_names'].index(f'{side}_hip_roll') for side in ('left', 'right')]])
        later = active.copy()
        later[:criteria['warmup_steps']] = False
        c = criteria['posture']
        posture_bad = ((rotation_matrix[:, 2, 2] < np.cos(np.deg2rad(c['tilt_max_deg'])))
                       | (abs(trace['height'][part, world] - c['nominal_height_m']) > c['height_tolerance_m']))
        denominator = max(1, int(later.sum()))
        yaw = float(abs(trace['angular_velocity'][part, world, 2][active]).mean()) if active.any() else None
        speed = float(velocity[active, 0].mean()) if active.any() else None
        slip = float(trace['foot_speed'][part, world][contacts].mean()) if contacts.any() else None
        tests = dict(survived=active.all(), block_error=block['mean_error_per_episode'][0] <= criteria['linear_error']['forward'],
            yaw_error=yaw is not None and yaw <= criteria['yaw_error_max'],
            signed_speed=speed is not None and speed >= criteria['signed_speed']['forward'],
            inward=(((hip[:, 0] < -20.) & (hip[:, 1] > 20.) & later).sum()/denominator <= criteria['inward_fraction_max']),
            contact_slip=slip is not None and slip <= criteria['contact_slip_max'],
            physical_joints=(trace['joint_envelope_ok'][part, world] | ~active).all(),
            posture=(posture_bad & later).sum()/denominator <= c['maximum_bad_sample_fraction'],
            feet_separation=(((separation < c['minimum_signed_site_separation_m']) & later).sum()
                             / denominator <= c['maximum_bad_sample_fraction']))
        result.append(dict(world=world, entered_forward=True, success=all(tests.values()),
            forward_start=int(begin), signed_speed_m_s=speed, yaw_error_rad_s=yaw, contact_slip_m_s=slip,
            block_error_m_s=float(block['mean_error_per_episode'][0]),
            failed_checks=[name for name, passed in tests.items() if not passed]))
    return result
