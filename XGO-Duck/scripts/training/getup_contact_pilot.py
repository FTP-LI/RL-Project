"""Bounded full-contact getup adaptation with episode-complete observations.

Both arms start from the same official actor and admitted local critic. The
candidate enables only the declared late head-leg cost. No random lengths,
unlogged extra rollout or automatic extension. A severe late kinematic decline
or numeric/interface/capacity error stops and preserves the current attempt.
"""
from copy import deepcopy
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import sys
import time
import traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/validation'))
from s1_reference_inputs import read,sha
from record_io import write
from getup_contact_training import prepare as base_prepare
from getup_value_bootstrap import networks,td
from recover_getup_value_startup import startup
from getup_masked_ppo import MaskedPPO
from action_video import ActionVideo
from training_watch import Snapshot


def prepare(p):
    cfg,display,metadata=base_prepare(p)
    cfg.env.auto_reset=True
    cfg.env.rewards['late_head_leg_contact'].params['enabled']=p['arm']=='candidate'
    display.update(recorder='ActionVideo',duration='complete8s episode; terminal before auto reset')
    return cfg,display,metadata


def kinematic_stable(qpos,qvel,addresses,ranges):
    q=qpos[...,3:7];q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)
    tilt=torch.acos((1-2*(q[...,1].square()+q[...,2].square())).clamp(-1,1))
    bounded=((qpos[...,addresses]>=ranges[:,0]-.034906585)&(qpos[...,addresses]<=ranges[:,1]+.034906585)).all(-1)
    return ((qpos[...,2]-.139158).abs()<=.02)&(tilt<=np.deg2rad(15.))&(torch.linalg.vector_norm(qvel[...,3:6],dim=-1)<=1.)&bounded


def save_checkpoint(path,algorithm,env,completed):
    state=dict(actor_state_dict=algorithm.actor.state_dict(),critic_state_dict=algorithm.critic.state_dict(),
        actor_optimizer_state_dict=algorithm.actor_optimizer.state_dict(),critic_optimizer_state_dict=algorithm.critic_optimizer.state_dict(),
        completed_updates=completed,actor_steps_this_run=algorithm.actor_steps,critic_steps_this_run=algorithm.critic_steps,
        control_steps=int(env.common_step_counter),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(env.device))
    def finite(v):
        if torch.is_tensor(v):return bool(torch.isfinite(v).all())
        if isinstance(v,dict):return all(finite(x) for x in v.values())
        if isinstance(v,(tuple,list)):return all(finite(x) for x in v)
        return True
    assert finite(state)
    temp=path.with_suffix('.partial');torch.save(state,temp);temp.replace(path)


def export(actor,metadata,probe,p,record):
    import onnx,onnxruntime as ort
    artifact=ROOT/'artifacts'/p['run_id'];artifact.mkdir()
    model=deepcopy(actor).cpu().eval();wrapper=model.as_onnx(False)
    torch.onnx.export(wrapper,torch.zeros(1,61),str(artifact/'policy.onnx'),opset_version=18,
        input_names=['obs'],output_names=['actions'],dynamo=False)
    policy=onnx.load(artifact/'policy.onnx')
    onnx.helper.set_model_props(policy,{**metadata,'run_name':p['run_id'],
        'training_source':'common full-contact masked PPO; locally fitted critic','hardware_validated':'false'})
    onnx.checker.check_model(policy);onnx.save(policy,artifact/'policy.onnx')
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(artifact/'policy.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    with torch.no_grad():expected=model(td('actor',probe.cpu())).numpy()
    actual=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in probe.cpu().numpy()])
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    np.savez_compressed(record/'export-parity.npz',observations=probe.cpu().numpy(),torch_actions=expected,onnx_actions=actual)
    write(artifact/'manifest.json',dict(status='exported-parity-checked',run_id=p['run_id'],
        model_sha256=sha(artifact/'policy.onnx'),checkpoint_sha256=sha(record/'model-final.pt'),
        observations=len(probe),maximum_error=float(abs(actual-expected).max()),hardware_deployment_ready=False))


def run(p,master_path,out):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from common_scene_contract import compare_arrays
    from evaluate_standup_reference import Tee
    assert os.isatty(1),'Use the single visible VS Code task terminal'
    progress=out/p['arm']/'progress.json';assert not progress.exists(),'No automatic retry'
    record=ROOT/'records/experiments'/p['run_id'];record.mkdir();write(record/'protocol.json',p)
    state=dict(status='training',run_id=p['run_id'],path=str(record),pid=os.getpid(),completed_updates=0,
        actor_updates=0,critic_updates=0,control_steps=0,planned_updates=50,num_envs=1024,
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(master_path),hardware_io=False)
    env=snapshot=video=algorithm=None;completed=0;trace=[];peak=0
    def publish(**fields):
        state.update(fields)
        if env is not None:state['control_steps']=int(env.common_step_counter)
        if video is not None:state.update(clips=video.clips,images=[video.image] if video.image else [])
        write(record/'state.json',state);write(progress,state)
    publish();write(ROOT/'records/training-current.json',dict(record=str(record),path=str(record),run_id=p['run_id'],pid=os.getpid(),purpose='masked-PPO-full-contact-getup'))
    with (record/'execution.log').open('x',buffering=1) as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            print('VISIBLE_MASKED_GETUP',p['arm'],'50updates x24steps x1024; full8s clips; no automatic extension',flush=True)
            configure_torch_backends(allow_tf32=False);torch.set_num_threads(4)
            cfg,display,metadata=prepare(p);write(record/'display.json',display)
            env=ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None);env.seed(42);obs,_=env.reset()
            actor,critic=networks(75,p,env.device)
            warm=torch.load(ROOT/p['critic_checkpoint'],map_location='cpu',weights_only=True)
            critic.load_state_dict(warm['critic_state_dict'])
            algorithm=MaskedPPO(actor,critic,warm['critic_optimizer_state_dict'],p['learner'])
            startup(env,obs,actor,p,record)
            cpu=lambda v:v.detach().cpu().numpy().copy()
            initial=dict(actor=cpu(obs['actor']),critic=cpu(obs['critic']),qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel),
                imu=cpu(env._imu_misalign_quat),cpu_rng=cpu(torch.get_rng_state()),cuda_rng=cpu(torch.cuda.get_rng_state(env.device)))
            initial.update({'expanded__'+k:cpu(getattr(env.sim.model,k)) for k in sorted(env.sim.expanded_fields)})
            np.savez_compressed(record/'paired-start.npz',**initial);save_checkpoint(record/'model-init.pt',algorithm,env,0)
            if p['arm']=='candidate':
                control=ROOT/'records/experiments'/p['matching_control']
                with np.load(control/'paired-start.npz') as f:compare_arrays(initial,{k:f[k] for k in f.files})
                other=torch.load(control/'model-init.pt',map_location='cpu',weights_only=True)
                for key,network in (('actor_state_dict',actor),('critic_state_dict',critic)):
                    assert all(torch.equal(v.detach().cpu(),other[key][k]) for k,v in network.state_dict().items())
                assert algorithm.actor_optimizer.state_dict()==other['actor_optimizer_state_dict']
                for key,value in algorithm.critic_optimizer.state_dict()['state'].items():
                    for name,v in value.items():torch.testing.assert_close(v.cpu() if torch.is_tensor(v) else v,other['critic_optimizer_state_dict']['state'][key][name],atol=0,rtol=0)
                write(record/'matched-start.json',dict(status='passed',physical_steps=0,observations_physics_imu_rng_exact=True,actor_critic_optimizers_exact=True))
            with np.load(ROOT/p['probe_dataset']) as f:probe=torch.from_numpy(f['actor_obs_sample'][[0,99,100,125,200,300,399]].reshape(-1,61).copy()).to(env.device)
            with torch.no_grad():teacher=actor(td('actor',probe)).clone()
            snapshot=Snapshot(env,record);video=ActionVideo(snapshot,'getup',record);video.sample()
            # Read pre-reset physical state for logging/watchdog. Chain the
            # complete-video hook so terminal footage is also captured first.
            before_reset=env.recorder_manager.record_pre_reset;terminal={}
            def capture(ids):
                terminal.update(ids=ids.clone(),qpos=env.sim.data.qpos[ids].clone(),qvel=env.sim.data.qvel[ids].clone())
                before_reset(ids)
            env.recorder_manager.record_pre_reset=capture
            addresses=torch.tensor(p['qpos_addresses'],device=env.device);ranges=torch.tensor(p['joint_ranges'],device=env.device)
            slots=int(env.sim.wp_data.contact.dist.shape[0]);bad_streak=0;start=time.monotonic();actor_rollouts=0
            for update in range(1,51):
                batch={key:[] for key in ('actor','critic','action','mean','log_prob','value','reward','done','active')}
                for _ in range(24):
                    age=env.episode_length_buf.clone();active=age>=100;terminal.clear()
                    with torch.no_grad():
                        actions=actor(obs,stochastic_output=True);mean=actor.output_mean.clone();logprob=actor.get_output_log_prob(actions).clone()
                        torch.testing.assert_close(actor.output_std,torch.full_like(actor.output_std,.05),atol=1e-8,rtol=0)
                        values=critic(obs).squeeze(-1)
                        for key,v in (('actor',obs['actor']),('critic',obs['critic']),('action',actions),('mean',mean),('log_prob',logprob),('value',values),('active',active)):batch[key].append(v.clone())
                        sent=actions.clone();obs,reward,term,timeout,_=env.step(actions)
                    torch.testing.assert_close(actions,sent,atol=0,rtol=0)
                    done=term|timeout;assert not term.any(),'Numeric termination'
                    assert torch.equal(done,age==399),'Unexpected early timeout/reset'
                    expected=torch.where(active[:,None],sent,torch.zeros_like(sent));expected[done]=0
                    torch.testing.assert_close(obs['actor'][:,34:48],expected,atol=0,rtol=0)
                    batch['reward'].append(reward.clone()/p['reward_scale']);batch['done'].append(done.clone())
                    qpos=env.sim.data.qpos.clone();qvel=env.sim.data.qvel.clone()
                    if done.any():qpos[terminal['ids']]=terminal['qpos'];qvel[terminal['ids']]=terminal['qvel']
                    assert torch.isfinite(qpos).all() and torch.isfinite(qvel).all() and torch.isfinite(reward).all()
                    stable=kinematic_stable(qpos,qvel,addresses,ranges);late=age+1>=350
                    fraction=float(stable.float().mean());bad_streak=bad_streak+1 if late.all() and fraction<p['watch_stable_min'] else 0
                    step=int(env.common_step_counter);count=int(env.sim.wp_data.nacon.numpy()[0]);peak=max(peak,count);assert 0<=count<slots
                    trace.append(dict(step=step,qpos=cpu(qpos[:32]),qvel=cpu(qvel[:32]),
                        stable_fraction=fraction,head_leg_fraction=float(env._getup_contact['contact'].float().mean()),
                        eligible_contact_fraction=float((env._getup_contact['contact']&env._getup_contact['eligible']).float().mean()),
                        active_policy_fraction=float(active.float().mean()),raw_reward_mean=float(reward.mean()),done=bool(done.all())))
                    if step==700 and video.writer is None:video.last_started=None
                    video.sample()
                    assert bad_streak<25,'Late recovery kinematic stability declined for25steps; stop and review'
                    if step%25==0:assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
                tensors={k:torch.stack(v) for k,v in batch.items()}
                with torch.no_grad():last=critic(obs).squeeze(-1)
                metrics=algorithm.update(tensors,last);completed=update;actor_rollouts+=metrics['actor_gradient_steps']>0
                with torch.no_grad():drift=actor(td('actor',probe))-teacher
                metrics.update(completed_updates=update,control_steps=int(env.common_step_counter),
                    raw_action_drift_rms=float(drift.square().mean().sqrt()),raw_action_drift_max=float(drift.abs().max()),
                    current_stable_fraction=fraction,elapsed_seconds=time.monotonic()-start)
                with (record/'learning-metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
                publish(completed_updates=update,actor_updates=actor_rollouts,critic_updates=update,metrics=metrics)
                print('GETUP_PPO',p['arm'],json.dumps(metrics),flush=True)
                if update in (25,50):save_checkpoint(record/('model-final.pt' if update==50 else 'model-25.pt'),algorithm,env,completed)
            assert int(env.common_step_counter)==1200
            video.close();export(actor,metadata,probe,p,record)
            publish(status='budget-complete',completed_updates=50,actor_updates=actor_rollouts,
                actor_optimizer_steps=algorithm.actor_steps,critic_optimizer_steps_this_run=algorithm.critic_steps,
                elapsed_seconds=time.monotonic()-start,finished_at=datetime.now(timezone.utc).isoformat())
        except BaseException as error:
            if algorithm is not None and env is not None:save_checkpoint(record/'stopped-partial.pt',algorithm,env,completed)
            publish(status='stopped-review-required',error=repr(error));traceback.print_exc();raise
        finally:
            if video is not None:video.close()
            if snapshot is not None:snapshot.close()
            if trace:np.savez_compressed(record/'training-trace.npz',**{k:np.asarray([v[k] for v in trace]) for k in trace[0]})
            if env is not None:
                write(record/'runtime.json',dict(control_steps=int(env.common_step_counter),control_transitions=int(env.common_step_counter)*1024,
                    physics_substeps=int(env.common_step_counter)*5,extra_observation_or_render_steps=0,contact_peak=peak,
                    completed_updates=completed,capacity=slots if 'slots' in locals() else None))
                if hasattr(env,'_getup_contact_audits'):write(record/'contact-audits.json',env._getup_contact_audits)
                publish();env.close()
            sys.stdout,sys.stderr=stdout,stderr
