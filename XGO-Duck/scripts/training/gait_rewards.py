"""Project-owned reward for the bounded gait-repair pilot.

Change/why: charge excessive requested joint targets instead of clipping the
executed action. The original position-limit cost sees actual position and
does not directly bound overdriving commands. The fixed-policy diagnosis
supports testing a narrower hip-roll envelope while learning balance again.
Expected: smaller target excess; normal posture/tracking still need evaluation.
The envelope is a simulation experiment, not a calibrated hardware range.
"""
import math
import torch


def action_target_envelope_cost(env, action_name='joint_pos',
                                hard_limit_margin_rad=0.3, hip_roll_bound_deg=10.0):
    """Sum target excess in radians; never mutate actions or physical limits.

    Extends the archived action_over_limit_penalty formula with a named hip
    bound. All other controlled joints retain their hard limit plus margin.
    A negative reward weight penalizes the returned nonnegative cost.
    """
    assert math.isfinite(hard_limit_margin_rad) and hard_limit_margin_rad >= 0
    assert math.isfinite(hip_roll_bound_deg) and 0 < hip_roll_bound_deg < 22
    term=env.action_manager.get_term(action_name)
    names=list(term.target_names)
    roll=[names.index(side+'_hip_roll') for side in ('left','right')]
    target=term.raw_action*term.scale+term.offset
    hard=env.scene['robot'].data.joint_pos_limits[:,term.target_ids]
    lower=hard[...,0]-hard_limit_margin_rad
    upper=hard[...,1]+hard_limit_margin_rad
    bound=math.radians(hip_roll_bound_deg)
    # Arithmetic above allocates independent bounds; these assignments do not
    # alter robot limits, ctrl, raw_action or processed_action.
    lower[...,roll]=-bound
    upper[...,roll]=bound
    return torch.sum(torch.relu(target-upper)+torch.relu(lower-target),dim=-1)
