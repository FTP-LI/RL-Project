"""Honor mjlab's ended-world reset while never reopening its first episode.

The original no-reset collector stopped when one world ended early. Capture
its real terminal state first, reset only ended IDs as required by the API,
clear only those histories/controller rows, and keep scoring masks outside
this helper permanently inactive. These resets are explicit and unscored;
they are not a physical reset at a successful skill switch.
"""
from types import SimpleNamespace
import numpy as np
import torch


def reset_ended(env,ended,previous,controller,valid,active,records):
    ids=torch.nonzero(ended,as_tuple=False).squeeze(-1)
    if not len(ids):return None
    cpu=lambda x:x.detach().cpu().numpy().copy()
    indices=cpu(ids);step=int(env.common_step_counter)
    assert not active[indices].any(),'A scored first episode cannot be reset'
    kept_ids=torch.as_tensor(np.flatnonzero(active),device=env.device,dtype=torch.long)
    kept_qpos=env.sim.data.qpos[kept_ids].clone();kept_qvel=env.sim.data.qvel[kept_ids].clone()
    kept_history={k:getattr(env.action_manager,k)[kept_ids].clone()
        for k in ('_action','_prev_action','_prev_prev_action') if hasattr(env.action_manager,k)}
    before_qpos=cpu(env.sim.data.qpos[ids]);before_qvel=cpu(env.sim.data.qvel[ids])
    obs,info=env.reset(env_ids=ids)
    assert int(env.common_step_counter)==step,'Reset must not advance physics'
    torch.testing.assert_close(env.sim.data.qpos[kept_ids],kept_qpos,atol=0,rtol=0)
    torch.testing.assert_close(env.sim.data.qvel[kept_ids],kept_qvel,atol=0,rtol=0)
    for k,v in kept_history.items():torch.testing.assert_close(getattr(env.action_manager,k)[kept_ids],v,atol=0,rtol=0)
    previous[indices]=0.;controller.reset(indices)
    torch.testing.assert_close(env.action_manager._action[ids],torch.zeros_like(env.action_manager._action[ids]),atol=0,rtol=0)
    after_qpos=cpu(env.sim.data.qpos[ids]);after_qvel=cpu(env.sim.data.qvel[ids])
    for j,world in enumerate(indices):
        records.append(dict(control_step=step,world=int(world),first_episode_just_ended=bool(valid[world]),
            other_world_states_exact=True,
            before_qpos=before_qpos[j],after_qpos=after_qpos[j],
            before_qvel=before_qvel[j],after_qvel=after_qvel[j]))
    return obs


def save_records(path,records):
    shapes=dict(control_step=(),world=(),first_episode_just_ended=(),other_world_states_exact=(),
        before_qpos=(21,),after_qpos=(21,),before_qvel=(20,),after_qvel=(20,))
    arrays={}
    for key,shape in shapes.items():
        dtype=np.int64 if key in ('control_step','world') else bool if key in ('first_episode_just_ended','other_world_states_exact') else np.float32
        arrays[key]=np.asarray([x[key] for x in records],dtype=dtype).reshape((len(records),)+shape)
    np.savez_compressed(path,**arrays)


def cpu_check():
    from walking_command_controller import create_controller
    n=4;controller=create_controller(n,.02)
    observed=np.zeros((n,61),np.float32);observed[:,2]=np.arange(n)
    controller.map(np.zeros((n,3),np.float32),observed)
    previous=np.ones((n,14),np.float32);valid=np.ones(n,bool);active=np.array([True,False,True,False])
    data=SimpleNamespace(qpos=torch.arange(n*21,dtype=torch.float32).reshape(n,21),
        qvel=torch.ones(n,20))
    manager=SimpleNamespace(_action=torch.ones(n,14));calls=[]
    env=SimpleNamespace(common_step_counter=79,device='cpu',sim=SimpleNamespace(data=data),action_manager=manager)
    def reset(*,env_ids):
        calls.append(env_ids.clone());data.qpos[env_ids]=-1;data.qvel[env_ids]=0;manager._action[env_ids]=0
        return {'actor':torch.zeros(n,61)},{}
    env.reset=reset;kept=data.qpos[[0,2]].clone();ema=controller.ema[[0,2]].copy();records=[]
    result=reset_ended(env,torch.tensor([False,True,False,True]),previous,controller,valid,active,records)
    assert result['actor'].shape==(n,61) and len(calls)==1 and len(records)==2
    np.testing.assert_array_equal(active,[True,False,True,False])
    np.testing.assert_array_equal(previous[[1,3]],0.);np.testing.assert_array_equal(previous[[0,2]],1.)
    np.testing.assert_array_equal(controller.ema[[0,2]],ema);assert not controller.ready[[1,3]].any()
    torch.testing.assert_close(data.qpos[[0,2]],kept,atol=0,rtol=0)
    # A repeated unscored failure stays unscored; a live first episode is denied.
    valid[:]=False;reset_ended(env,torch.tensor([False,True,False,False]),previous,controller,valid,active,records)
    assert records[-1]['first_episode_just_ended'] is False
    rejected=False
    try:reset_ended(env,torch.tensor([True,False,False,False]),previous,controller,valid,active,records)
    except AssertionError:rejected=True
    assert rejected and env.common_step_counter==79 and not torch.cuda.is_initialized()
    return dict(status='passed-cpu-real-helper',only_ended_inactive_ids_reset=True,
        first_episode_masks_never_reopened=True,active_world_physics_and_history_preserved=True,
        selective_controller_reset=True,real_terminal_before_reset_saved=True,
        repeated_unscored_termination_checked=True,live_first_episode_reset_rejected=True,
        physics_steps=0,learning_updates=0)
