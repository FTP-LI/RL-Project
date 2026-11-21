"""Development progress statistic: one-second mean velocity, first episode.

Preserve instantaneous error and legacy flags as separate diagnostics. This
statistic cannot certify full gait quality. Exclude the first second; every
subsequent second contributes equally. Incomplete blocks incur the frozen
missing-block penalty, so resetting/terminating cannot erase difficult data.
"""
import numpy as np


def block_statistic(velocity,active,command,block_steps=50,warmup_steps=50,missing_penalty=.5):
    velocity=np.asarray(velocity);active=np.asarray(active)
    assert velocity.ndim==3 and velocity.shape[:2]==active.shape and velocity.shape[2]==2
    assert active.dtype==bool and not np.any(np.diff(active.astype(int),axis=0)>0)
    total,n=active.shape;assert (total-warmup_steps)%block_steps==0
    blocks=(total-warmup_steps)//block_steps;assert blocks>0
    data=velocity[warmup_steps:].reshape(blocks,block_steps,n,2)
    valid=active[warmup_steps:].reshape(blocks,block_steps,n).all(axis=1)
    means=data.mean(axis=1)
    error=np.linalg.norm(means-np.asarray(command)[:2],axis=2)
    error=np.where(valid,error,missing_penalty)
    return dict(mean_error_per_episode=error.mean(axis=0),valid_blocks_per_episode=valid.sum(axis=0),
        expected_blocks=blocks,block_means=means,block_valid=valid)
