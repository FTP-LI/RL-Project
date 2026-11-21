"""Candidate reward for the diagnosed airborne-support loss.

The official roll already pays only contact-qualified progress. The new
candidate adds a small penalty when forward rotation continues during the
30--240 degree mid-roll window without any foot/head/body ground contact.
It leaves the official actor, observations, action history and all landing
terms unchanged; the control arm does not add this term.
"""
import math

import torch


TERM = "roulade_airborne_support_penalty"


def airborne_forward_rotation_penalty(
    env,
    angle_lo=math.radians(30.0),
    angle_hi=math.radians(240.0),
    rate_norm=2.0,
):
    from mjlab_microduck.tasks import mdp

    asset = env.scene["robot"]
    mdp._update_roulade_accum(env, asset)
    accum, _, _ = mdp._roulade_state(env)
    omega = mdp._ROULADE_FWD_SIGN * asset.data.root_link_ang_vel_b[:, 1]
    forward_rate = torch.clamp(torch.nan_to_num(omega, nan=0.0) / rate_norm, min=0.0)
    window = ((accum >= angle_lo) & (accum <= angle_hi)).to(forward_rate.dtype)
    contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for name in ("feet_ground_contact", "head_ground_contact", "robot_ground_contact"):
        try:
            found = env.scene[name].data.found
        except (KeyError, AttributeError):
            continue
        contact |= found.reshape(env.num_envs, -1).any(dim=1)
    return forward_rate * window * (~contact).to(forward_rate.dtype)


def adapt(cfg, candidate: bool, weight: float = -0.35):
    """Return a copied task config with one declared candidate reward term."""
    from copy import deepcopy

    cfg = deepcopy(cfg)
    assert TERM not in cfg.rewards
    if candidate:
        template = cfg.rewards["roulade_progress"]
        cfg.rewards[TERM] = type(template)(
            func=airborne_forward_rotation_penalty,
            weight=weight,
            params={
                "angle_lo": math.radians(30.0),
                "angle_hi": math.radians(240.0),
                "rate_norm": 2.0,
            },
        )
    return cfg
