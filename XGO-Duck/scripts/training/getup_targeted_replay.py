"""Side-lying success replay terms for the fixed getup development screen."""
import torch

from getup_value_bootstrap import td


def side_replay(learner, joint_order, spec, variant):
    """Use the complete declared cells, including the hardest early neck rows."""
    def load_rows(settings):
        rows = sorted({i for cell in settings['cells'] for i in learner.anchor_cells[cell]})
        columns = [joint_order.index(name) for name in settings['joints']]
        return learner.anchor_inputs[rows], learner.anchor_targets[rows][:, columns], columns

    neck = load_rows(spec['neck']) if variant in ('neck', 'combined') else None
    hips = load_rows(spec['hips']) if variant in ('hips', 'combined') else None

    def errors(group):
        inputs, targets, columns = group
        return learner.actor(td('actor', inputs))[:, columns]-targets

    def loss(sigma):
        value = 0.
        if neck is not None:
            settings = spec['neck']
            excess = (errors(neck).abs()-settings['margin_rad']).clamp_min(0).flatten()
            value += settings['multiplier']*torch.topk(excess, settings['top_count']).values.square().mean()
        if hips is not None:
            value += spec['hips']['multiplier']*errors(hips).square().sum(-1).mean()
        return value/(2*sigma*sigma)

    return loss
