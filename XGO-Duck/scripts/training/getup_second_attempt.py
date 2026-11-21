"""A one-shot retreat before a second attempt, on the original episode clock.

The detector uses the policy's gravity and gyro readings. The retreat acts
after ordinary action processing, so the policy continues to see its own raw
action history. The motor target approaches a legal destination from the last
applied target. Triggered worlds also use this slew when rejoining the policy.
"""

import torch

from getup_stand_handoff import projected_slew


class SecondAttempt:
    def __init__(self, worlds, device, screen, settings, limits, home, neck_column, alignment):
        self.release = round(screen["release_seconds"]/screen["control_period_seconds"])
        self.first = screen["first_decision_control"]
        self.last = screen["last_decision_control"]
        self.length = screen["tilt_history_intervals"]+1
        self.minimum_tilt = screen["minimum_tilt_deg"]
        self.maximum_range = screen["maximum_tilt_range_deg"]
        self.dt = screen["control_period_seconds"]
        self.rate = settings["max_target_rate_rad_s"]
        self.duration = settings["controls"]
        self.settle = settings["settle_allowance_controls"]
        self.quiet_needed = settings["quiet_supine_consecutive_controls"]
        self.maximum_gyro = settings["maximum_angular_speed_rad_s"]
        self.neck_withdrawal = settings["physical_neck_withdrawal_min_rad"]
        self.alignment = alignment
        self.limits = torch.as_tensor(limits,device=device,dtype=torch.float32)
        self.retreat = torch.as_tensor(home,device=device,dtype=torch.float32).clone()
        self.retreat[neck_column] = self.limits[neck_column,0]+settings["neck_pitch_inside_original_lower_limit_rad"]
        self.history = torch.full((self.length,worlds),float("nan"),device=device,dtype=torch.float64)
        self.supine = torch.zeros(worlds,device=device,dtype=torch.bool)
        self.trigger = torch.full((worlds,),-1,device=device,dtype=torch.int32)
        self.ready = torch.full_like(self.trigger,-1)
        self.quiet = torch.zeros_like(self.trigger)
        self.neck_start = torch.zeros(worlds,device=device,dtype=torch.float32)
        self.boundary = self.release-1

    def observe(self, boundary, sensed, neck):
        assert boundary == self.boundary+1 and sensed.shape == (len(self.trigger),61)
        self.boundary = boundary
        gravity = sensed[:,3:6].to(torch.float64)
        gravity = gravity/torch.linalg.vector_norm(gravity,dim=1,keepdim=True)
        supine = gravity[:,0] <= -self.alignment
        if boundary == self.release:
            self.supine.copy_(supine)
        if self.first-self.length+1 <= boundary <= self.last:
            tilt = torch.rad2deg(torch.acos((-gravity[:,2]).clamp(-1.,1.)))
            self.history[boundary % self.length].copy_(tilt)
        if self.first <= boundary <= self.last:
            low,high = torch.aminmax(self.history,dim=0)
            new = self.supine&(self.trigger < 0)&(low >= self.minimum_tilt)&(high-low <= self.maximum_range)
            self.trigger.masked_fill_(new,boundary)
            self.neck_start.copy_(torch.where(new,neck,self.neck_start))
        if boundary >= self.first+self.duration-self.settle+1:
            elapsed = boundary-self.trigger
            gyro = torch.linalg.vector_norm(sensed[:,:3],dim=1)
            in_settle = (self.trigger >= 0)&(elapsed > self.duration-self.settle)&(elapsed <= self.duration)
            quiet = in_settle & supine & (gyro <= self.maximum_gyro)
            self.quiet.copy_(torch.where(quiet,self.quiet+1,0))
            done = in_settle&(elapsed == self.duration)&(self.quiet >= self.quiet_needed)
            self.ready.masked_fill_(done&(neck-self.neck_start >= self.neck_withdrawal),boundary)

    def targets(self, boundary, previous, requested, ordinary):
        assert boundary == self.boundary
        triggered = self.trigger >= 0
        retreating = triggered & (boundary-self.trigger < self.duration)
        desired = torch.where(retreating[:,None],self.retreat,requested)
        target,_ = projected_slew(previous,desired,self.limits,self.dt,self.rate)
        return torch.where(triggered[:,None],target,ordinary)

    def snapshot(self):
        return self.boundary,tuple(value.clone() for value in self._mutable())

    def restore(self, saved):
        self.boundary,values = saved
        for dest,source in zip(self._mutable(),values,strict=True):
            dest.copy_(source)

    def _mutable(self):
        return (self.history,self.supine,self.trigger,self.ready,self.quiet,self.neck_start)


def apply_processed_targets(term, controller, boundary, previous, enabled):
    """Keep requested targets intact and remember the target actually applied."""
    if enabled and boundary >= controller.first:
        actual = controller.targets(boundary,previous,term.requested_target,term._processed_actions)
        term._processed_actions.copy_(actual)
        term.previous_target.copy_(actual)
