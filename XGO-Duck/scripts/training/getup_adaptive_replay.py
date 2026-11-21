"""Regularize the current worst successful-state errors across all joints."""
import torch

from getup_value_bootstrap import td


def adaptive_replay(learner, settings):
    rows = columns = None
    target = learner.anchor_targets
    count = settings['top_count']
    stride = learner.plan['minibatches']

    def loss(sigma):
        nonlocal rows, columns
        # Refresh at each epoch and on each retried proposal. Selection is
        # detached, and the next minibatches differentiate only selected rows.
        if rows is None or learner.critic_steps % stride == 0:
            with torch.no_grad():
                error = (learner.actor(td('actor', learner.anchor_inputs))-target).abs()
                indices = torch.argsort(error.flatten(), descending=True, stable=True)[:count]
                rows, columns = indices // target.shape[1], indices % target.shape[1]
        prediction = learner.actor(td('actor', learner.anchor_inputs[rows]))
        excess = ((prediction[torch.arange(len(rows), device=rows.device), columns]-target[rows, columns]).abs()
                  - settings['margin_rad']).clamp_min(0)
        return settings['multiplier']*excess.square().mean()/(2*sigma*sigma)

    return loss
