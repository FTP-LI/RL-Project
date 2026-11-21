"""One early support term with a pre-action release latch and selective reset."""
import math

import torch


class SupineReleaseLatch:
    def __init__(self, worlds, device, release_age, alignment):
        self.release_age = release_age
        self.alignment = alignment
        self.latched = torch.zeros(worlds, dtype=torch.bool, device=device)
        self.supine = torch.zeros_like(self.latched)

    def observe(self, actor_input, age):
        # Called on the same actor observation used to generate this action.
        gravity = actor_input[:, 3:6]
        norm = torch.linalg.vector_norm(gravity, dim=-1, keepdim=True)
        valid = torch.isfinite(gravity).all(dim=-1) & (norm[:, 0] > 0)
        normalized = gravity / torch.where(valid[:, None], norm, torch.ones_like(norm))
        selected = valid & (normalized.abs().argmax(dim=-1) == 0) & (normalized[:, 0] <= -self.alignment)
        update = (age == self.release_age) & ~self.latched
        self.supine.copy_(torch.where(update, selected, self.supine))
        self.latched.logical_or_(update)

    def reset(self, ids):
        self.latched[ids] = False
        self.supine[ids] = False


def reset_early_support(env, env_ids):
    latch = getattr(env, '_early_support_release', None)
    if latch is not None:
        latch.reset(slice(None) if env_ids is None else env_ids)


def inputs(env):
    gravity_z = env.scene['robot'].data.projected_gravity_b[:, 2]
    feet = env.scene['feet_ground_contact'].data.found.reshape(env.num_envs, 2, -1).any(dim=2)
    head_feet, head_hips = (env.scene[name].data.found.reshape(env.num_envs, -1).any(dim=1)
                            for name in ('head_feet_contact', 'head_hips_contact'))
    return gravity_z, feet, head_feet | head_hips


def component(gravity_z, feet, head_leg, age, supine, *, start_age, stop_age, tilt_limit_degrees):
    tilt = torch.acos((-gravity_z).clamp(-1., 1.)) * (180./math.pi)
    progress = ((90.-tilt)/(90.-tilt_limit_degrees)).clamp(0., 1.)
    eligible = supine & (age >= start_age) & (age <= stop_age) & ~head_leg
    return torch.where(eligible, progress * feet.float().mean(dim=-1), 0.)


def early_support(env, *, start_age, stop_age, tilt_limit_degrees):
    return component(*inputs(env), env.episode_length_buf, env._early_support_release.supine,
                     start_age=start_age, stop_age=stop_age, tilt_limit_degrees=tilt_limit_degrees)
