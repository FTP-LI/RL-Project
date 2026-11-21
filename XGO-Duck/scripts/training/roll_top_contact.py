"""Roll reward/coordinate adapters; preserve physical reset and raw history.

Common to both arms: actor relative joint positions and absolute target offset
use model HOME, while robot default/reset stays at source22degrees. Both arms
compute the same pivot and top-contact gate; only candidate removes the30%
reward for non-top head contact. No new observation, filter or phase clock.
"""
import torch


def model_relative_joint_pos(env,model_home,biased=False,asset_cfg=None):
    from mjlab.envs.mdp.observations import joint_pos_rel
    assert asset_cfg is not None
    base=joint_pos_rel(env,biased=biased,asset_cfg=asset_cfg)
    asset=env.scene[asset_cfg.name]
    default=asset.data.default_joint_pos[:,asset_cfg.joint_ids]
    home=torch.as_tensor(model_home,dtype=base.dtype,device=base.device)
    assert base.shape[1]==home.numel()==14
    return base+default-home


def top_contact_pivot(env,only_top=False,**params):
    from mjlab_microduck.tasks import mdp
    value=mdp.roulade_head_pivot(env,**params)
    top=mdp._head_top_down(env,env.scene['robot']).to(value.dtype)
    return value*top if only_top else value


def adapt(cfg,metadata,only_top):
    from copy import deepcopy
    cfg=deepcopy(cfg);names=metadata['joint_names'].split(',');home=list(map(float,metadata['default_joint_pos'].split(',')))
    assert len(names)==len(home)==14 and cfg.actions['joint_pos'].scale==1.
    action=cfg.actions['joint_pos'];action.use_default_offset=False;action.offset=dict(zip(names,home))
    term=cfg.observations['actor'].terms['joint_pos']
    assert term.func.__name__=='joint_pos_rel' and 'model_home' not in term.params
    term.func=model_relative_joint_pos;term.params=dict(term.params,model_home=home)
    reward=cfg.rewards['roulade_head_pivot'];assert reward.func.__name__=='roulade_head_pivot'
    reward.func=top_contact_pivot;reward.params=dict(reward.params,only_top=only_top)
    return cfg
