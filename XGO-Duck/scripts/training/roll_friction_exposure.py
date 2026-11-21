"""Keep the official friction range; expose the roll learner to its low end more often."""

import torch
from mjlab.envs.mdp.dr._types import Distribution, uniform

RANGE = (0.7, 1.3)
LOW_EDGE = 0.85
FEET = ('left_foot_collision', 'right_foot_collision')


def remap(draw, lo, hi):
    """Map the two equally probable halves to [0.7,0.85] and [0.85,1.3]."""
    middle = (lo + hi) * .5
    return torch.where(draw < middle,
        lo + (draw-lo) * ((LOW_EDGE-lo)/(middle-lo)),
        LOW_EDGE + (draw-middle) * ((hi-LOW_EDGE)/(hi-middle)))


def sample_enriched(lo, hi, shape, device):
    # Use the official draw first. The remap needs no additional RNG draws, so
    # other startup randomization is exposed to the same random stream.
    return remap(uniform.sample(lo, hi, shape, device), lo, hi)


ENRICHED = Distribution(name='roll-low-friction-exposure', sample=sample_enriched)


def apply(cfg, role):
    """Modify only the distribution of the official shared-foot startup draw."""
    if role not in ('official-distribution', 'low-friction-enriched'):
        raise ValueError('Unknown friction arm')
    event = cfg.events['foot_friction']
    p = event.params
    if (event.mode != 'startup' or event.func.__name__ != 'geom_friction'
            or tuple(p['ranges']) != RANGE or p['operation'] != 'abs'
            or tuple(p['asset_cfg'].geom_names) != FEET or p['asset_cfg'].name != 'robot'
            or p.get('shared_random') is not True or p.get('axes') not in (None, [0])
            or p.get('distribution', 'uniform') != 'uniform'):
        raise ValueError('The official foot-friction event does not match the experiment')
    if role == 'low-friction-enriched':
        event.params = dict(p, distribution=ENRICHED)
    return cfg
