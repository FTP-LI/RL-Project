"""A single signed yaw perturbation on the existing walking interface."""
import numpy as np

from walking_command_controller import create_controller


class ForwardYawPulse:
    def __init__(self, plan, settings, desired):
        self.base = create_controller(len(desired), .02)
        self.desired = desired
        self.pulse = np.asarray(settings['pulse_signs'], dtype=np.float32) * np.float32(settings['peak_rad_s'])
        self.step = settings['warmup_controls'] + 1
        self.forward = self.pulse != 0
        assert np.flatnonzero(self.forward).tolist() == sorted(plan['groups']['forward'])

    def map(self, observed, step, candidate, active):
        command = self.base.map(self.desired, observed)
        if candidate and step == self.step:
            enabled = self.forward & active
            assert not command[enabled, 2].any()
            command[enabled, 2] = self.pulse[enabled]
        return command

    def reset(self, ids):
        self.base.reset(ids)

    def snapshot(self):
        return self.base.ema.copy(), self.base.ready.copy()

    def restore(self, state):
        self.base.ema[:], self.base.ready[:] = state
