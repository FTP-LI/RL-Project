"""Common physics adoption and zero-step live contracts for fixed skills.

Preserve each frozen collector's observations, controller and acceptance.
Replace only the explicitly declared physical recipe. Compare complete CPU
scene fields and nominal expanded primary fields before the first step, then
record actual contacts and model-HOME interfaces without advancing physics.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import torch
from common_skill_plant import make_config,RESET_NAMES
from inspect_common_skill_plant import FIELDS,stable,digest
from s1_reference_inputs import ROOT,read,sha
from record_io import write

DERIVED={'body_invweight0','dof_invweight0'}


def adopt_physics(cfg,skill):
    common=make_config('roll');n=cfg.scene.num_envs
    cfg.scene=deepcopy(common.scene)
    cfg.scene.num_envs=cfg.scene.terrain.num_envs=n
    cfg.sim=deepcopy(common.sim);cfg.decimation=common.decimation
    cfg.events={k:v for k,v in cfg.events.items() if k in RESET_NAMES}
    for key in ('foot_friction','expand_bam_friction_fields'):
        cfg.events[key]=deepcopy(common.events[key])
    cfg.curriculum={};cfg.auto_reset=False
    return cfg


def physical_description(cfg):
    return dict(robot=asdict(cfg.scene.entities['robot']),sensors=[asdict(s) for s in cfg.scene.sensors],
        terrain={k:v for k,v in asdict(cfg.scene.terrain).items() if k!='num_envs'},
        sim=asdict(cfg.sim),decimation=cfg.decimation,
        startup={k:asdict(cfg.events[k]) for k in ('foot_friction','expand_bam_friction_fields')})


def model_arrays(model):return {k:np.asarray(getattr(model,k)).copy() for k in FIELDS}


def compare_arrays(actual,expected):
    assert set(actual)==set(expected)
    for key in actual:np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)


def nominal_primary(fields,n):
    result={}
    for key,value in fields.items():
        if key in DERIVED:continue
        assert value.shape[0]==n,(key,value.shape,n)
        np.testing.assert_array_equal(value,np.broadcast_to(value[:1],value.shape),err_msg=key)
        result[key]=value[0].copy()
    return result


def interface_check(env,obs,plan,folder,tag):
    """Actual observation parity; preserve both CPU and CUDA RNG streams."""
    import onnxruntime as ort
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict
    from metadata_actor_import import import_actor
    cpu=lambda x:x.detach().cpu().numpy().copy()
    rng_cpu=torch.get_rng_state().clone();rng_cuda=torch.cuda.get_rng_state(env.device).clone()
    model=ROOT/plan['model'];assert sha(model)==plan['model_sha256']
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    meta=session.get_modelmeta().custom_metadata_map;names=meta['joint_names'].split(',')
    assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
    assert float(meta['action_scale'])==1.
    home=np.asarray(meta['default_joint_pos'].split(','),np.float32)
    robot=env.scene['robot'];ids,order=robot.find_joints(names,preserve_order=True);assert order==names
    term=env.action_manager.get_term('joint_pos');assert list(term._target_names)==names
    n=env.num_envs;sim_home=np.broadcast_to(cpu(term._offset),(n,14)).copy()
    scale=np.broadcast_to(np.asarray(cpu(term._scale) if torch.is_tensor(term._scale) else term._scale),(n,14)).copy()
    assert np.isfinite(scale).all() and (scale!=0).all()
    np.testing.assert_allclose(sim_home,cpu(robot.data.default_joint_pos[:,ids]),atol=1e-7,rtol=0)
    raw_obs=cpu(obs['actor']);assert raw_obs.shape==(n,61) and np.isfinite(raw_obs).all()
    np.testing.assert_allclose(raw_obs[:,34:48],0,atol=1e-7,rtol=0)
    np.testing.assert_allclose(raw_obs[:,51:],0,atol=1e-6,rtol=0)
    adapted=raw_obs.copy();adapted[:,6:20]+=sim_home-home;adapted[:,34:48]=0
    if plan['skill']=='walk':
        from walking_command_controller import create_controller
        desired=np.zeros((n,3),np.float32)
        for name,group in plan['groups'].items():desired[group]=plan['commands'][name]
        controller=create_controller(n,.02);controller.reset(np.arange(n))
        adapted[:,48:51]=controller.map(desired,adapted)
    else:np.testing.assert_allclose(adapted[:,48:],0,atol=1e-6,rtol=0)
    selected=adapted[:32].copy()
    expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in selected])
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,
            hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
            distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'scalar'})
        import_actor(actor,model,plan['model_sha256']);actor.eval()
        with torch.no_grad():actual=actor(TensorDict({'actor':torch.from_numpy(selected)},batch_size=[len(selected)])).numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    assert torch.equal(rng_cpu,torch.get_rng_state()) and torch.equal(rng_cuda,torch.cuda.get_rng_state(env.device))
    targets=home+expected;sent=(targets-sim_home[:len(selected)])/scale[:len(selected)]
    np.testing.assert_allclose(sim_home[:len(selected)]+scale[:len(selected)]*sent,targets,atol=1e-6,rtol=0)
    path=folder/f'{tag}-interface.npz'
    np.savez_compressed(path,raw_observations=raw_obs,adapted_observations=adapted,
        torch_actions=actual,onnx_actions=expected,simulation_home=sim_home,model_home=home,
        action_scale=scale,initial_qpos=cpu(env.sim.data.qpos),initial_qvel=cpu(env.sim.data.qvel))
    result=dict(status='passed',skill=plan['skill'],common_step_counter=int(env.common_step_counter),
        model_sha256=sha(model),maximum_inference_error=float(abs(actual-expected).max()),
        home_and_order_checked=True,initial_raw_history_zero=True,rng_unchanged=True,
        archive_sha256=sha(path),action_units='model HOME + raw output; source action offset/scale inverted',
        getup_policy_output_executed=False if plan['skill']=='getup' else None,
        scope='Zero-step interface only; getup remains frozen for2s before policy execution')
    write(folder/f'{tag}-interface.json',result);return result


def install_guard(plan,folder,startup_root,mode):
    import mjlab.envs
    base=mjlab.envs.ManagerBasedRlEnv
    skill=plan['skill'];reference=startup_root/skill

    class AuditedEnv(base):
        def __init__(self,*args,**kwargs):
            self._common_ready=False;self.guard_steps=0;self.guard_peak_contacts=0;self.guard_resets=[];self.guard_slots=0
            super().__init__(*args,**kwargs)
            self._common_ready=True
            assert self.common_step_counter==0
            arrays=model_arrays(self.sim.mj_model)
            np.savez_compressed(folder/'scene-static-fields.npz',**arrays)
            if mode=='startup':
                if skill!='walk':
                    with np.load(startup_root/'walk/scene-static-fields.npz') as f:compare_arrays(arrays,{k:f[k] for k in f.files})
            else:
                with np.load(reference/'scene-static-fields.npz') as f:compare_arrays(arrays,{k:f[k] for k in f.files})
            assert self.physics_dt==.004 and self.step_dt==.02
            opt=self.sim.mj_model.opt
            assert opt.timestep==.004 and opt.iterations==30 and opt.ls_iterations==50
            self.guard_slots=int(self.sim.wp_data.contact.dist.shape[0]);assert self.guard_slots>0
            write(folder/'scene-static-contract.json',dict(status='passed',skill=skill,mode=mode,common_step_counter=0,
                compared_fields=list(FIELDS),archive_sha256=sha(folder/'scene-static-fields.npz'),
                contact_slots=self.guard_slots,num_envs=self.num_envs,physics_dt=self.physics_dt,control_dt=self.step_dt,
                comparison='Same complete CPU model physics arrays; rollout compared to its frozen startup check'))

        def reset(self,*args,**kwargs):
            result=super().reset(*args,**kwargs)
            if not self._common_ready or kwargs.get('env_ids') is not None:return result
            tag=f'reset-{len(self.guard_resets)}'
            checked=interface_check(self,result[0],plan,folder,tag);self.guard_resets.append(checked)
            if self.common_step_counter==0:
                fields={k:getattr(self.sim.model,k).detach().cpu().numpy().copy() for k in sorted(self.sim.expanded_fields)}
                primary=nominal_primary(fields,self.num_envs)
                np.savez_compressed(folder/'nominal-primary-fields.npz',**primary)
                target=(startup_root/'walk' if mode=='startup' else reference)/'nominal-primary-fields.npz'
                if mode!='startup' or skill!='walk':
                    with np.load(target) as f:compare_arrays(primary,{k:f[k] for k in f.files})
                np.savez_compressed(folder/'expanded-fields-before-physics.npz',**fields)
                write(folder/'nominal-contract.json',dict(status='passed',common_step_counter=0,
                    primary_fields=list(primary),derived_fields_recorded_not_cross_skill_compared=sorted(set(fields)&DERIVED),
                    primary_sha256=sha(folder/'nominal-primary-fields.npz'),
                    expanded_sha256=sha(folder/'expanded-fields-before-physics.npz'),
                    scope='All worlds nominal and identical in expanded primary parameters at the same zero-physics instant; dynamic BAM friction is not compared across later times'))
            write(folder/'interface-checks.json',self.guard_resets)
            return result

        def step(self,actions):
            assert mode!='startup','Startup checks must never advance physics'
            assert torch.isfinite(actions).all(),'Nonfinite action'
            result=super().step(actions);self.guard_steps+=1
            count=int(self.sim.wp_data.nacon.numpy()[0]);self.guard_peak_contacts=max(self.guard_peak_contacts,count)
            if not 0<=count<self.guard_slots:
                np.savez_compressed(folder/'capacity-stop-state.npz',qpos=self.sim.data.qpos.cpu().numpy(),qvel=self.sim.data.qvel.cpu().numpy())
                raise RuntimeError('Contact capacity reached; preserve data and review')
            if self.guard_steps%25==0:
                assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False),'Stop requested'
            return result

        def close(self):
            if self._common_ready:
                write(folder/'runtime-contract.json',dict(status='closed',skill=skill,mode=mode,
                    control_steps=self.guard_steps,control_transitions=self.guard_steps*self.num_envs,
                    contact_peak=self.guard_peak_contacts,contact_slots=self.guard_slots,
                    capacity_reached=self.guard_peak_contacts>=self.guard_slots,interface_checks=len(self.guard_resets),
                    physics_substeps=self.guard_steps*5,extra_observation_or_render_steps=0))
            super().close()
    mjlab.envs.ManagerBasedRlEnv=AuditedEnv
    return AuditedEnv
