"""Latch early right-side stalls and measure bounded signed upright progress."""
import math

import torch


class RightUprightProgress:
    def __init__(self,worlds,device,release_age,alignment,entry,horizon):
        self.release_age = release_age
        self.alignment = alignment
        self.last_trigger = round(entry['last_trigger_seconds']/.02)
        self.min_tilt = entry['minimum_tilt_deg_exclusive']
        self.max_tilt = entry['maximum_tilt_deg_inclusive']
        self.max_reduction = entry['maximum_recent_tilt_reduction_deg_exclusive']
        self.horizon = horizon
        self.release_known = torch.zeros(worlds,dtype=torch.bool,device=device)
        self.right = torch.zeros_like(self.release_known)
        self.trigger_age = torch.full((worlds,),-1,dtype=torch.long,device=device)
        self.previous_potential = torch.zeros(worlds,dtype=torch.float32,device=device)
        shape = (entry['consecutive_samples'],worlds)
        self.tilts = torch.zeros(shape,dtype=torch.float32,device=device)
        self.supported = torch.zeros(shape,dtype=torch.bool,device=device)
        self.slot = 0

    def observe_release(self,actor_input,age):
        gravity = actor_input[:,3:6]
        norm = torch.linalg.vector_norm(gravity,dim=-1)
        valid = torch.isfinite(gravity).all(dim=-1)&(norm>0.)
        right = valid&(gravity.abs().argmax(dim=-1)==1)&(gravity[:,1]<=-self.alignment*norm)
        new = (age==self.release_age)&~self.release_known
        self.right.copy_(torch.where(new,right,self.right))
        self.release_known.logical_or_(new)

    def advance(self,gravity_z,feet,head_leg,age,active):
        tilt = torch.acos((-gravity_z).clamp(-1.,1.))*(180./math.pi)
        potential = ((45.-tilt)/30.).clamp(0.,1.)*feet.float().mean(dim=1)
        ready = (self.right&active&(age>self.release_age)&(age<=self.last_trigger)&feet.all(dim=1)&head_leg
                 &(tilt>self.min_tilt)&(tilt<=self.max_tilt))
        self.tilts[self.slot].copy_(tilt)
        self.supported[self.slot].copy_(ready)
        self.slot = (self.slot+1)%len(self.tilts)
        slow = self.tilts[self.slot]-tilt<self.max_reduction
        start = (self.trigger_age<0)&self.supported.all(dim=0)&slow
        self.trigger_age.copy_(torch.where(start,age,self.trigger_age))
        elapsed = age-self.trigger_age
        live = active&(self.trigger_age>=0)&(elapsed>0)&(elapsed<=self.horizon)
        reward = torch.where(live,potential-self.previous_potential,0.)
        self.previous_potential.copy_(potential)
        return reward

    def reset(self,ids):
        self.release_known[ids] = False
        self.right[ids] = False
        self.trigger_age[ids] = -1
        self.previous_potential[ids] = 0.
        self.tilts[:,ids] = 0.
        self.supported[:,ids] = False
