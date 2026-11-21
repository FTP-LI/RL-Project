"""Causal, simulation-only neck request during the first head support."""
import numpy as np

from roll_metrics import rotation


class NeckSupport:
    def __init__(self, size, settings, column, limits):
        assert settings['joint'] == 'neck_pitch'
        for key in ('requires_earlier_bilateral_foot_support',
                    'requires_current_head_contact_without_foot_contact',
                    'requires_unmodified_neck_request_within_hard_limits'):
            assert settings[key]
        self.settings = settings
        self.column = column
        self.lower, self.upper = limits
        self.phase = np.zeros(size)
        self.prior_feet = np.zeros(size, bool)
        self.previous_pitch = None

    def choose(self, quaternion, feet, head, omega_y, neck_velocity, active, request, enabled):
        r = rotation(quaternion)
        lateral = r[:, 2, 1]
        horizontal = (r[:, 0, 0] * r[:, 1, 1] - r[:, 1, 0] * r[:, 0, 1])
        horizontal /= np.maximum(np.linalg.norm(r[:, :2, 1], axis=-1), 1e-6)
        pitch = np.arctan2(-r[:, 2, 0], horizontal)
        if self.previous_pitch is not None:
            self.phase += (pitch - self.previous_pitch + np.pi) % (2 * np.pi) - np.pi
        self.previous_pitch = pitch

        config = self.settings
        angle = np.rad2deg(self.phase)
        start, end = config['forward_phase_degrees']
        original = request[:, self.column]
        cue = (active & self.prior_feet & head & ~feet.any(axis=1)
               & (angle >= start) & (angle <= end)
               & (np.abs(lateral) <= config['maximum_body_lateral_projection'])
               & (omega_y > config['forward_body_pitch_rate_above_rad_s'])
               & (neck_velocity < config['neck_rate_below_rad_s'])
               & (original >= self.lower) & (original <= self.upper))
        self.prior_feet |= active & feet.all(axis=1)
        # Keep the actor's output and history unchanged. Only the sent target
        # is changed, and only if the unmodified request is already legal.
        result = request.copy()
        if enabled:
            result[cue, self.column] = np.minimum(
                original[cue] + config['maximum_positive_target_offset_rad'], self.upper)
        return result, cue, angle

    def snapshot(self):
        assert self.previous_pitch is not None
        return tuple(value.copy() for value in (self.phase, self.prior_feet, self.previous_pitch))

    def restore(self, state):
        self.phase, self.prior_feet, self.previous_pitch = (value.copy() for value in state)
