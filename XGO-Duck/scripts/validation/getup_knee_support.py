"""Probe one short knee-extension pulse after stalled two-foot support.

Saved failures asked for knee flexion while successful transitions asked for
extension. This component changes the two physical targets after the normal
action processor. It leaves raw policy actions, history and weights alone.
The contact gate is simulation-only. No environment is advanced here.
"""
import math

import torch


class KneeSupportPulse:
    def __init__(self, worlds, device, settings):
        self.settings = settings
        self.support_ticks = torch.zeros(worlds, device=device, dtype=torch.long)
        self.trigger = torch.full_like(self.support_ticks, -1)
        self.supine = torch.zeros_like(self.support_ticks, dtype=torch.bool)
        self.step_offset = settings['offset_rate_rad_s'] * settings['step_dt']
        self.ramp = round(settings['knee_extension_rad'] / self.step_offset)
        assert math.isclose(self.ramp * self.step_offset, settings['knee_extension_rad'], abs_tol=1e-9)
        self.duration = 2 * self.ramp + settings['plateau_controls']

    def observe(self, step, gravity, feet, height, pitch_rate):
        """Observe one completed control interval before computing the next target."""
        s = self.settings
        if step == s['release_control']:
            self.supine = (gravity.abs().argmax(dim=-1) == 0) & (gravity[:, 0] <= -s['release_alignment_min'])
        if step <= s['release_control'] or step > s['latest_trigger_control']:
            return
        self.support_ticks = torch.where(feet.all(dim=-1), self.support_ticks + 1, 0)
        enough_support = self.support_ticks >= s['sustained_foot_controls']
        vertical = -gravity[:, 2]
        safe_lean = vertical >= math.cos(math.radians(s['total_lean_max_deg']))
        forward_lean = gravity[:, 0] > vertical * math.tan(math.radians(s['signed_forward_lean_min_deg']))
        near_stand = (height - s['nominal_height_m']).abs() <= s['height_tolerance_m']
        stationary = pitch_rate.abs() <= s['body_pitch_speed_max_rad_s']
        start = self.supine & enough_support & safe_lean & forward_lean & near_stand & stationary & (self.trigger < 0)
        self.trigger[start] = step

    def offset(self, next_step):
        elapsed = next_step - self.trigger
        ramp_ticks = torch.minimum(elapsed, self.duration - elapsed).clamp(0, self.ramp)
        return torch.where(self.trigger >= 0, ramp_ticks * self.step_offset, 0.)

    def targets(self, next_step, target, knee_ids, knee_limits, enabled):
        """Keep original excess unchanged and bound any new knee offset."""
        if not enabled:
            return target
        knees = target[:, knee_ids]
        lower, upper = knee_limits[:, 0], knee_limits[:, 1]
        allowed = ((knees >= lower) & (knees <= upper)).all(dim=-1)
        signs = target.new_tensor(self.settings['signed_offset'])
        amount = self.offset(next_step).to(target.dtype)
        correction = amount[:, None] * signs
        changed = (knees + correction).clamp(lower, upper)
        result = target.clone()
        use_pulse = allowed & (amount > 0.)
        result[:, knee_ids] = torch.where(use_pulse[:, None], changed, knees)
        return result


def apply_after_processing(term, pulse, next_step, knee_ids, knee_limits, enabled):
    """Preserve the requested action and retain the target actually sent."""
    if enabled:
        sent = pulse.targets(next_step, term._processed_actions, knee_ids, knee_limits, True)
        term._processed_actions.copy_(sent)
        term.previous_target.copy_(sent)
