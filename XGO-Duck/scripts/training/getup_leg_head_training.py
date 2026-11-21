"""Restrict actor gradients to the ten leg outputs of the final linear layer."""

import torch


def restrict_actor_to_leg_output(actor, joint_names):
    layer = actor.mlp[-1]
    assert isinstance(layer, torch.nn.Linear) and layer.out_features == len(joint_names) == 14
    legs = torch.tensor([name.startswith(('left_', 'right_')) for name in joint_names],
                        device=layer.weight.device)
    assert int(legs.sum()) == 10 and layer.bias is not None
    actor.requires_grad_(False)
    layer.requires_grad_(True)
    layer.weight.register_hook(lambda gradient: gradient.masked_fill(~legs[:, None], 0.))
    layer.bias.register_hook(lambda gradient: gradient.masked_fill(~legs, 0.))
    return layer, legs
