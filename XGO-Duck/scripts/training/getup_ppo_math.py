"""Finite-episode GAE and policy loss for the declared getup action freeze.

The source overrides actions for the first2seconds. Those samples still train
the critic but cannot train the actor: the sampled action did not control the
robot. Normalize advantages using only active-policy samples. At a genuine
finite episode boundary use terminal value0, including timeout; do not use the
already-reset observation to bootstrap the preceding episode.
"""
import torch


def finite_gae(rewards,values,dones,last_values,gamma=.99,lam=.95):
    assert rewards.shape==values.shape==dones.shape and rewards.ndim==2
    assert dones.dtype==torch.bool and last_values.shape==values.shape[1:]
    advantages=torch.empty_like(rewards);carry=torch.zeros_like(last_values)
    for t in range(len(rewards)-1,-1,-1):
        following=last_values if t==len(rewards)-1 else values[t+1]
        alive=(~dones[t]).to(rewards.dtype)
        delta=rewards[t]+gamma*following*alive-values[t]
        carry=delta+gamma*lam*carry*alive;advantages[t]=carry
    return advantages,advantages+values


def normalize_active_advantages(advantages,active):
    assert active.shape==advantages.shape and active.dtype==torch.bool
    result=torch.zeros_like(advantages)
    if active.any():
        chosen=advantages[active]
        result[active]=(chosen-chosen.mean())/(chosen.std(unbiased=False)+1e-8)
    return result


def masked_surrogate(log_prob,old_log_prob,advantages,active,clip=.2):
    assert log_prob.shape==old_log_prob.shape==advantages.shape==active.shape
    if not active.any():return log_prob.sum()*0.
    ratio=torch.exp(log_prob[active]-old_log_prob[active])
    return -torch.minimum(ratio*advantages[active],ratio.clamp(1-clip,1+clip)*advantages[active]).mean()
