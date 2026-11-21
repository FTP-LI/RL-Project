"""Observe a persistent folded-neck stall before the old5s intervention.

Use only the actor's measured gravity, neck position/velocity, raw request
and controller clock. A brief pivot is allowed. The candidate must keep the
condition for0.30s of past observations before latching. No simulated contact
or privileged height is an input. The-75deg bound comes from the earlier saved
geometry audit; dynamics and independent reliability still require a new test.
The bounded upward correction is separate from raw policy history and does
not claim a bound on the original policy's total target slew.
"""
import math
import torch

SETTINGS=dict(start_seconds=2.,tilt_min_deg=15.,tilt_max_deg=45.,
    neck_below_deg=-110.,neck_speed_max_rad_s=1.,dwell_seconds=.30,
    lower_target_deg=-75.,upward_correction_rate_rad_s=2.)


def detect(count,latched,observed,raw_neck_target,ages,dt,home,neck,lower_limit,settings=SETTINGS):
    assert observed.shape==(len(ages),61) and raw_neck_target.shape==ages.shape
    assert count.dtype==torch.long and latched.dtype==torch.bool and ages.dtype==torch.long
    assert dt>0 and torch.isfinite(observed).all() and torch.isfinite(raw_neck_target).all()
    gravity=observed[:,3:6];norm=torch.linalg.vector_norm(gravity,dim=-1)
    assert (norm>1e-8).all()
    vertical=-gravity[:,2]/norm
    position=observed[:,6+neck]+home
    velocity=observed[:,20+neck]
    eligible=(ages>=round(settings['start_seconds']/dt))
    eligible &= (vertical>=math.cos(math.radians(settings['tilt_max_deg'])))
    eligible &= (vertical<=math.cos(math.radians(settings['tilt_min_deg'])))
    eligible &= (position<math.radians(settings['neck_below_deg']))
    eligible &= (velocity.abs()<=settings['neck_speed_max_rad_s']) & (raw_neck_target<lower_limit)
    #16samples span15elapsed intervals: no future observation is implied.
    required=math.ceil(settings['dwell_seconds']/dt-1e-9)+1
    count=torch.where(eligible,(count+1).clamp_max(required),torch.zeros_like(count))
    latched=latched|(count>=required)
    assert not (latched&(ages<round(settings['start_seconds']/dt))).any(),'Reset guard state at real episode reset'
    return count,latched,eligible


def correction_step(previous,latched,raw_neck_target,dt,settings=SETTINGS):
    bound=math.radians(settings['lower_target_deg'])
    desired=torch.where(latched,(bound-raw_neck_target).clamp_min(0),torch.zeros_like(previous))
    return torch.minimum(previous+settings['upward_correction_rate_rad_s']*dt,desired)
