"""Training-only protection of successful official getup action means.

Small local PPO KL did not prevent accumulated drift on successful inputs.
Use a fixed, traceable teacher replay and a detached equal-sigma mean cost.
Nothing here changes environment rewards, observes contacts, runs a teacher
on hardware, or starts learning. A separate matched PPO admission is required.
"""
import torch


def mean_anchor_loss(mean,target,sigma):
    assert mean.ndim==2 and mean.shape==target.shape and mean.shape[-1]==14
    assert sigma>0 and torch.isfinite(mean).all() and torch.isfinite(target).all()
    return (mean-target.detach()).square().sum(-1).mean()/(2*sigma*sigma)


def anchored_objective(policy_loss,anchor_loss,weight):
    assert weight>=0
    # Preserve the original tensor/graph when the control coefficient is0.
    return policy_loss if weight==0 else policy_loss+weight*anchor_loss


def balanced_indices(cells,generator,samples_per_cell=8):
    """Equal requested-pose/time cells with an isolated CPU RNG stream."""
    assert generator.device.type=='cpu' and samples_per_cell>0
    chunks=[]
    for name in sorted(cells):
        candidates=torch.as_tensor(cells[name],dtype=torch.long,device='cpu')
        assert len(candidates)>0
        local=torch.randint(len(candidates),(samples_per_cell,),generator=generator,device='cpu')
        chunks.append(candidates[local])
    return torch.cat(chunks)
