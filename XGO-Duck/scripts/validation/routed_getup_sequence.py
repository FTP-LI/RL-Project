"""Use the accepted getup, standing and forward actors without resetting physics."""

from math import cos, radians

import numpy as np
import torch

from command_input_mapping import transform
from getup_stand_handoff import projected_slew
from getup_value_bootstrap import td
from run_walk_forward_yaw_direction_pilot import components
from s1_reference_inputs import ROOT, read
from walking_actuator_yaw_feedback import ActuatorYawFeedback, projected_feedback_target


GETUP, STAND, FORWARD = range(3)


class ForwardFeedback:
    """The accepted forward law, with its clock starting at forward entry."""

    def __init__(self, spec, worlds):
        self.observer, self.predict, self.thresholds, window, width = components(spec)
        settings = spec['feedback']
        self.law = ActuatorYawFeedback(read(ROOT / spec['feedback_screen']), settings['maximum_yaw_input_rad_s'])
        self.history = np.zeros((worlds, window, width), np.float32)
        self.previous = np.zeros((worlds, 2), np.float32)
        self.hips = [20 + spec['joint_names'].index(f'{side}_hip_pitch') for side in ('left', 'right')]
        self.warmup = spec['phase']['warmup_controls']
        self.direction_start = spec['shared_prefix_controls']
        self.extra = spec['direction_pilot']['maximum_extra_input_rad_s']
        self.maximum_target_change = settings['maximum_joint_correction_rad']

    def advance(self, x, ids, step):
        window, width = self.history.shape[1:]
        self.history[ids, :-1] = self.history[ids, 1:]
        self.history[ids, -1] = x[:, :width]
        current = np.column_stack((x[:, 2], x[:, self.hips].sum(axis=1)))
        change = current - self.previous[ids] if step > 1 else np.zeros_like(current)
        self.previous[ids] = current
        if step <= self.warmup:
            return None
        baseline = self.law.commands(np.column_stack((current, change))).astype(np.float32)
        if step <= self.direction_start:
            return baseline
        assert step >= window
        phase = self.observer.predict(x)
        logit = self.predict(self.history[ids].reshape(len(ids), window * width))
        adjusted = np.clip(baseline - np.sign(logit) * self.extra, -self.law.cap, self.law.cap)
        return np.where(abs(logit) >= self.thresholds[phase], adjusted, baseline).astype(np.float32)


class SequenceController:
    def __init__(self, n, device, settings, homes, limits, forward, *, release=100, stand_at=550, forward_at=600):
        self.mode = torch.full((n,), GETUP, dtype=torch.int8, device=device)
        self.dwell = torch.zeros(n, dtype=torch.int32, device=device)
        self.history = torch.zeros((3, n, 14), device=device)
        self.homes = torch.as_tensor(homes, device=device, dtype=torch.float32)
        self.limits = torch.as_tensor(limits, device=device, dtype=torch.float32)
        self.alignment = cos(radians(settings['observed_tilt_max_deg']))
        self.speed = settings['observed_angular_speed_max_rad_s']
        self.dwell_steps = settings['standing_dwell_controls']
        self.bridge_steps = settings['cross_model_bridge_controls']
        self.rate = settings['bridge_slew_rad_s']
        self.forward_command = torch.from_numpy(transform([settings['forward_command_m_per_s'], 0., 0.])).to(device)
        self.release, self.stand_at, self.forward_at = release, stand_at, forward_at
        self.feedback = forward

    def advance(self, control, sensed):
        gravity = sensed[:, 3:6]
        norm = gravity.norm(dim=-1)
        ready = ((norm > 0) & (-gravity[:, 2] >= norm * self.alignment) &
                 (sensed[:, :3].norm(dim=-1) <= self.speed))
        self.dwell = torch.where(ready, self.dwell + 1, 0)
        if control == self.stand_at:
            self.mode[self.dwell >= self.dwell_steps] = STAND
            self.dwell.zero_()
        elif control == self.forward_at:
            self.mode[(self.mode == STAND) & (self.dwell >= self.dwell_steps)] = FORWARD
        bridge = (((self.mode == STAND) & (control - self.stand_at < self.bridge_steps)) |
                  ((self.mode == FORWARD) & (control - self.forward_at < self.bridge_steps)))
        x = sensed.clone()
        x[:, 48:] = 0.
        for role in (GETUP, STAND, FORWARD):
            selected = self.mode == role
            x[selected, 34:48] = self.history[role, selected]
            if role != GETUP:
                x[selected, 6:20] += self.homes[GETUP] - self.homes[role]
        x[self.mode == FORWARD, 48:51] = self.forward_command
        return x, bridge

    def infer(self, control, x, actors):
        raw = torch.zeros_like(self.history[GETUP])
        feedback_yaw = torch.zeros_like(self.dwell, dtype=raw.dtype)
        alternate = raw.clone()
        if control >= self.release and (self.mode == GETUP).any():
            # The getup route was latched on the original full batch at release.
            getup = actors[GETUP](td('actor', x))
            raw[self.mode == GETUP] = getup[self.mode == GETUP]
        for role, boundary in ((STAND, self.stand_at), (FORWARD, self.forward_at)):
            if control >= boundary:
                selected = self.mode == role
                if selected.any():
                    raw[selected] = actors[role](td('actor', x[selected]))
        forward = self.mode == FORWARD
        if forward.any():
            ids = forward.nonzero().flatten().cpu().numpy()
            alt_input = x[forward]
            yaw = self.feedback.advance(alt_input.cpu().numpy(), ids, control - self.forward_at + 1)
            if yaw is not None:
                feedback_yaw[forward] = torch.as_tensor(yaw, device=x.device)
                alt_input[:, 50] = feedback_yaw[forward]
                alternate[forward] = actors[FORWARD](td('actor', alt_input))
        for role in (GETUP, STAND, FORWARD):
            selected = self.mode == role
            self.history[role, selected] = raw[selected]
        return raw, feedback_yaw, alternate

    def apply(self, term, previous, raw, alternate, bridge, control, dt):
        for role in (STAND, FORWARD):
            mask = self.mode == role
            if not mask.any():
                continue
            request = raw[mask] + self.homes[role]
            target = request.clamp(min=self.limits[:, 0], max=self.limits[:, 1])
            if role == FORWARD and control - self.forward_at + 1 > self.feedback.warmup:
                target, _ = projected_feedback_target(target, alternate[mask], self.homes[role], self.limits,
                    self.feedback.maximum_target_change)
            proposed, _ = projected_slew(previous[mask], target, self.limits, dt, self.rate)
            applied = torch.where(bridge[mask, None], proposed, target)
            term.requested_target[mask] = request
            term.projected_destination[mask] = target
            term._processed_actions[mask] = applied
            term.previous_target[mask] = applied
