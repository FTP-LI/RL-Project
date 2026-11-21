"""Fixed actor and actual zero-step guard for the common walking interface."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from metadata_actor_import import import_actor
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from common_scene_contract import model_arrays,compare_arrays,nominal_primary


def td(x):return TensorDict({'actor':x},batch_size=x.shape[:-1])


def make_actor(p,device):
    device=torch.device(device);devices=[] if device.type=='cpu' else [device.index or 0]
    cpu=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state(device).clone() if devices else None
    with torch.random.fork_rng(devices=devices),torch.device('cpu'):
        torch.manual_seed(42)
        actor=MLPModel(td(torch.zeros(1,61)),{'actor':['actor']},'actor',14,
            hidden_dims=(512,256,128),activation='elu',obs_normalization=True)
        import_actor(actor,ROOT/p['model'],p['model_sha256'])
        actor=actor.to(device).eval().requires_grad_(False)
    assert torch.equal(cpu,torch.get_rng_state())
    if devices:assert torch.equal(gpu,torch.cuda.get_rng_state(device))
    return actor


def startup(env,obs,actor,p,folder,out,desired):
    import onnxruntime as ort
    from walking_command_controller import create_controller
    cpu=lambda x:x.detach().cpu().numpy().copy()
    rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state(env.device).clone()
    qpos=cpu(env.sim.data.qpos);qvel=cpu(env.sim.data.qvel);observed=cpu(obs['actor'])
    assert observed.shape==(224,61) and env.common_step_counter==0
    with np.load(out/'cpu-scene-fields.npz') as f:compare_arrays(model_arrays(env.sim.mj_model),{k:f[k] for k in f.files})
    expanded={k:cpu(getattr(env.sim.model,k)) for k in sorted(env.sim.expanded_fields)}
    nominal_primary(expanded,env.num_envs)
    np.savez_compressed(folder/'expanded-fields-before-physics.npz',**expanded)
    np.savez_compressed(folder/'scene-static-fields.npz',**model_arrays(env.sim.mj_model))
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
    home=np.asarray(p['model_home'],np.float32)
    robot=env.scene['robot'];ids,names=robot.find_joints(p['joint_names'],preserve_order=True)
    assert names==p['joint_names']
    default=cpu(robot.data.default_joint_pos[:,ids])
    adapted=observed.copy();adapted[:,6:20]+=default-home
    np.testing.assert_array_equal(adapted[:,34:48],0.)
    c=create_controller(224,.02);adapted[:,48:51]=c.map(desired,adapted)
    selected=adapted[:32].copy()
    expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in selected])
    with torch.no_grad():actual=cpu(actor(td(torch.from_numpy(selected).to(env.device))))
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    with np.load(out/'cpu-import-inputs.npz') as f:
        old=f['observations'];old_expected=f['onnx_actions']
    with torch.no_grad():old_actual=cpu(actor(td(torch.from_numpy(old).to(env.device))))
    np.testing.assert_allclose(old_actual,old_expected,atol=2e-5,rtol=2e-5)
    term=env.action_manager.get_term('joint_pos')
    attrs=('_raw_actions','_processed_actions','requested_target','projected_destination')
    saved={k:getattr(term,k).clone() for k in attrs}
    histories={k:getattr(env.action_manager,k).clone() for k in ('_action','_prev_action','_prev_prev_action')}
    sample=torch.full_like(term._raw_actions,4.);original=sample.clone()
    env.action_manager.process_action(sample)
    torch.testing.assert_close(sample,original,atol=0,rtol=0)
    torch.testing.assert_close(env.action_manager._action,original,atol=0,rtol=0)
    requested=original+torch.tensor(home,device=env.device)
    limits=torch.tensor(p['joint_limits'],device=env.device,dtype=sample.dtype)
    legal=torch.maximum(torch.minimum(requested,limits[:,1]),limits[:,0])
    torch.testing.assert_close(term.requested_target,requested,atol=0,rtol=0)
    torch.testing.assert_close(term._processed_actions,legal,atol=0,rtol=0)
    assert (requested!=legal).any()
    for k,v in saved.items():setattr(term,k,v.clone())
    for k,v in histories.items():getattr(env.action_manager,k).copy_(v)
    np.testing.assert_array_equal(cpu(env.sim.data.qpos),qpos);np.testing.assert_array_equal(cpu(env.sim.data.qvel),qvel)
    np.testing.assert_array_equal(cpu(obs['actor']),observed)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state(env.device))
    assert env.common_step_counter==0
    np.savez_compressed(folder/'live-startup.npz',observations=selected,torch_actions=actual,onnx_actions=expected,
        qpos=qpos,qvel=qvel,**{k:cpu(v) for k,v in saved.items()})
    write(folder/'startup.json',dict(status='passed',physical_steps=0,model_sha256=p['model_sha256'],
        live_inputs=32,saved_inputs=len(old),maximum_live_error=float(abs(actual-expected).max()),
        maximum_saved_error=float(abs(old_actual-old_expected).max()),
        full_scene_exact=True,nominal_expanded_fields=True,cpu_cuda_rng_unchanged=True,
        caller_and_raw_history_unchanged=True,actual_projected_processor_checked=True,
        complete_processor_memory_restored=True,observations_and_physical_state_unchanged=True))
