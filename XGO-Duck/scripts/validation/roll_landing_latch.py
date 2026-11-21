"""Stateful simulation-only readiness for a forward roll to stand."""
import numpy as np

from roll_metrics import rotation


class RollLandingLatch:
    def __init__(self, plan, worlds):
        self.dt = plan['step_dt']
        self.hold = round(plan['hold_seconds'] / self.dt)
        self.deadline = plan['deadline_seconds']
        self.low, self.high = np.deg2rad([360 - plan['completion_tolerance_deg'], 360 + plan['completion_tolerance_deg']])
        self.reverse = np.deg2rad(plan['backward_tolerance_deg'])
        self.alignment = np.cos(np.deg2rad(plan['tilt_max_deg']))
        self.gyro_limit = plan['angular_speed_max_rad_s']
        self.nominal_height = plan['nominal_height_m']
        self.height_tolerance = plan['height_tolerance_m']
        self.depth_limit = plan['head_penetration_max_m']
        self.pitch = np.zeros(worlds)
        self.angle = np.zeros(worlds)
        self.supported = np.zeros(worlds)
        self.seen_feet = np.zeros(worlds, dtype=bool)
        self.seen_head = self.seen_feet.copy()
        self.finished = self.seen_feet.copy()
        self.intact = ~self.seen_feet
        self.dwell = np.zeros(worlds, dtype=np.int32)
        self.first_ready = np.full(worlds, -1, dtype=np.int32)
        self.step = -1

    def advance(self, state):
        self.step += 1
        r = rotation(state['quaternion'])
        lateral = np.maximum(np.linalg.norm(r[:, :2, 1], axis=-1), 1e-6)
        forward = (r[:, 0, 0]*r[:, 1, 1] - r[:, 1, 0]*r[:, 0, 1]) / lateral
        pitch = np.arctan2(-r[:, 2, 0], forward)
        delta = (pitch - self.pitch + np.pi) % (2*np.pi) - np.pi
        self.pitch = pitch
        if self.step == 0:
            delta.fill(0.)
        self.angle += delta
        flat = np.clip((.866 - abs(r[:, 2, 1])) / (.866 - .5), 0., 1.)
        self.supported += np.where(delta < 0, delta, delta * flat**2 * (3. - 2.*flat) * state['ground_contact'])
        tilt_ok = r[:, 2, 2] >= self.alignment
        height_ok = abs(state['height'] - self.nominal_height) <= self.height_tolerance
        if self.step == 0:
            self.intact &= tilt_ok & height_ok
        path_ok = ((self.angle >= -self.reverse) & (self.angle <= self.high)
                   & state['joint_envelope_ok'] & (state['head_penetration_m'] <= self.depth_limit))
        self.intact &= path_ok & state['active']
        feet = state['foot_contact'].all(axis=-1)
        quiet = np.linalg.norm(state['angular_velocity'], axis=-1) <= self.gyro_limit
        stable = self.intact & self.finished & (self.angle >= self.low) & tilt_ok & height_ok & feet & quiet
        self.dwell = np.where(stable, self.dwell + 1, 0)
        fire = (self.dwell >= self.hold) & (self.first_ready < 0)
        if (self.step - self.hold)*self.dt > self.deadline + 1e-8:
            fire.fill(False)
        self.first_ready[fire] = self.step
        head_vertical = rotation(state['head_quaternion'])[:, 2, 1]
        self.seen_head |= (self.seen_feet & state['active'] & state['head_contact'] & (head_vertical < -.3)
                           & (self.angle > np.deg2rad(20)) & (self.angle < np.deg2rad(170)))
        self.seen_feet |= feet & state['active']
        self.finished |= self.seen_head & state['active'] & (np.minimum(self.angle, self.supported) >= self.low)
        return fire
