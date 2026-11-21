"""Credit reaching and holding the upright band during support transfer."""

import math

import torch


def upright_credit(up, full_credit_tilt_deg, credit_starts_tilt_deg):
    """Use cosine of the body's vertical direction, positive when upright."""
    degrees = torch.acos(up.clamp(-1., 1.)) * (180./math.pi)
    return ((credit_starts_tilt_deg-degrees)/(credit_starts_tilt_deg-full_credit_tilt_deg)).clamp(0., 1.)


def upright_band_increment(previous_up, up, previous_age, actual_supine, *,
                           start_age, stop_age, full_credit_tilt_deg, credit_starts_tilt_deg):
    prior = upright_credit(previous_up,full_credit_tilt_deg,credit_starts_tilt_deg)
    current = upright_credit(up,full_credit_tilt_deg,credit_starts_tilt_deg)
    # Initial credit retains earlier progress. The following signed increments
    # remove that credit when the robot leaves the original upright band.
    before = torch.where(previous_age==start_age,0.,prior)
    active = actual_supine & (previous_age>=start_age) & (previous_age<stop_age)
    return torch.where(active,current-before,0.)
