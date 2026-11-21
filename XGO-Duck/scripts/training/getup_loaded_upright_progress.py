"""Elapsed-time reward for upright progress after sustained right-foot load."""

import torch


def upright_tilt_deg(quaternion):
    horizontal = torch.linalg.vector_norm(quaternion[..., 1:3], dim=-1)
    vertical = torch.linalg.vector_norm(quaternion[..., ::3], dim=-1)
    return torch.rad2deg(2. * torch.atan2(horizontal, vertical))


class LoadedUprightProgress:
    def __init__(self, worlds, device, *, body_weight, foot_fraction, both_feet_fraction,
                 settled_reads, window_reads, maximum_tick_progress_deg):
        assert body_weight>0 and 0<foot_fraction<=both_feet_fraction and 0<settled_reads<=window_reads
        assert maximum_tick_progress_deg>0
        self.foot_threshold = body_weight*foot_fraction
        self.both_feet_threshold = body_weight*both_feet_fraction
        self.body_weight = body_weight
        self.settled_reads = settled_reads
        self.window_reads = window_reads
        self.maximum_tick_progress_deg = maximum_tick_progress_deg
        self.streak = torch.zeros(worlds, dtype=torch.int32, device=device)
        self.elapsed = torch.full_like(self.streak, -1)

    def reset(self, done):
        self.elapsed.masked_fill_(done, -1)
        self.streak.masked_fill_(done, 0)

    def advance(self, loads, before, after, enabled):
        left,right = loads[..., 2],loads[..., 3]
        total = left+right
        active = enabled & (self.elapsed>=0) & (self.elapsed<self.window_reads)
        both = (left>self.foot_threshold)&(right>self.foot_threshold)&(total>=self.both_feet_threshold)
        weights = (total/self.body_weight).clamp(0.,1.)
        angle = upright_tilt_deg(before)-upright_tilt_deg(after)
        raw = torch.where(active&both, angle*weights, 0.)
        bounded = (raw/self.maximum_tick_progress_deg).clamp(-1.,1.)/self.window_reads

        waiting = enabled & (self.elapsed<0)
        self.streak.copy_(torch.where(waiting & (right>self.foot_threshold),self.streak+1,0))
        confirmed = waiting & (self.streak==self.settled_reads)
        # The solve that confirms support belongs to the preceding phase.
        self.elapsed.add_(active.to(torch.int32))
        self.elapsed.masked_fill_(confirmed,0)
        return raw,bounded,confirmed,active
