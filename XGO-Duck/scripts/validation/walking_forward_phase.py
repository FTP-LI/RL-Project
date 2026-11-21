"""Add the frozen encoder-phase yaw offset to the existing walking controller."""
import numpy as np

from walking_command_controller import create_controller


class ForwardPhase:
    def __init__(self, plan, settings, desired):
        self.base = create_controller(len(desired), .02)
        self.columns = [20 + plan['joint_names'].index(f'{side}_hip_pitch') for side in ('left', 'right')]
        self.scale = np.float32(settings['scale'])
        self.peak = np.float32(settings['peak_rad_s'])
        self.polarity = settings['polarity']
        self.warmup = settings['warmup_controls']
        self.desired = desired
        self.forward = (desired == np.asarray(settings['external_command'], np.float32)).all(axis=1)
        assert np.flatnonzero(self.forward).tolist() == sorted(plan['groups']['forward'])
        assert np.isfinite(self.scale) and self.scale > 0. and self.peak > 0. and self.polarity in (-1, 1)

    def reset(self, ids):
        self.base.reset(ids)

    def map(self, observed, step, candidate, active):
        command = self.base.map(self.desired, observed)
        if candidate and step > self.warmup:
            enabled = self.forward & active
            left, right = self.columns
            signal = observed[enabled, left] + observed[enabled, right]
            assert not command[enabled, 2].any()
            command[enabled, 2] = -self.polarity * self.peak * np.clip(signal / self.scale, -1., 1.)
        return command

    def snapshot(self):
        return self.base.ema.copy(), self.base.ready.copy()

    def restore(self, state):
        self.base.ema[:], self.base.ready[:] = state
