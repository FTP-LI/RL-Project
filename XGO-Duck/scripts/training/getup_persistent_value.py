"""One bounded common value start for the prospective contact-duration pair.

Collect one fixed official0.10rad8s episode batch in the common plant. Capture
each original reward function once, form original/candidate returns on that
same data, and fit one NEW76-input critic to their mean. Both individual
return gates must pass; neither actor nor physics is fitted or resampled.
This is value preparation, not an actor-training or recovery-success claim.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,time,traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/training'),str(ROOT/'scripts/validation')]
from s1_reference_inputs import WORK,read,sha
from record_io import write
from getup_persistent_contact import prepare,networks,startup,clock
from getup_contact_training import split_worlds,POSES
from getup_value_bootstrap import td,fit,discounted_returns
from getup_anchor_runtime import RecordingSnapshot
from action_video import ActionVideo
PLAN=ROOT/'configs/experiments/getup_persistent_value.json'
OUT=ROOT/'records/reports/getup-persistent-contact/value'
STAGE='S3-persistent-contact-value-admission'


def freeze():
    from mjlab.scene import Scene
    from common_scene_contract import model_arrays,compare_arrays
    import yaml
    assert not PLAN.exists() and not OUT.exists()
    proof=read(OUT.parent/'component-preflight.json');assert proof['status']=='passed-cpu-components-only'
    for name,digest in proof['source_sha256'].items():assert sha(ROOT/name)==digest
    p=deepcopy(read(ROOT/'configs/experiments/getup_anchor_scale_recovery.json'))
    for key in ('critic_checkpoint','critic_sha256','source_value_admission','source_value_admission_sha256',
                'ppo_preflight_sha256','source_critic_optimizer_updates','scale_recovery'):
        p.pop(key,None)
    p.update(status='frozen-before-dynamics',stage=STAGE,seed=44,split_seed=4404,split=split_worlds(seed=4404),
        minibatch_seed=444,contact_threshold_seconds=1.2,persistent_penalty_enabled=False,
        candidate_reward_weight=-2.,candidate_reward_enabled=True,anchor_weight_for_future_both=.01,
        critic_width=76,actor_width=61,critic_updates=50,actor_updates=0,learning_updates=0,
        maximum_control_steps=400,maximum_control_transitions=409600,minibatch_size=8192,
        critic_learning_rate=.0003,gradient_norm_max=1.,training_metric_samples=8192,
        evaluation_updates=[0,1,10,25,50],critic_initialization='Fresh seed44; no inherited critic or optimizer',
        reward_scale='Train-world RMS of the mean of two finite discounted return targets; same divisor for both future arms',
        returns='Both genuine finite8s terminal0 targets; common critic fits their mean; gate both separately',
        reset_test='After terminal capture only: reset the first two positive-memory IDs, preserve all other physical/history/memory rows; no further controls',
        scope='Prospective paired-reward critic admission only;0actor updates;full nine-motion goal remains open',
        automatic_retry=False,automatic_extension=False,heldout_used=False,hardware_io=False,
        run_id='duck_getup_contact-duration-value_reference_s44_'+datetime.now().strftime('%Y%m%dT%H%M%S'),
        frozen_at=datetime.now(timezone.utc).isoformat())
    assert not any(f'_s44_' in d.name and 'contact-duration' in d.name for d in (ROOT/'records/experiments').iterdir())
    sources=deepcopy(p['source_sha256'])
    for name in ('scripts/training/getup_persistent_contact.py','scripts/training/getup_persistent_value.py',
        'scripts/validation/check_getup_persistent_contact.py','scripts/validation/verify_getup_persistent_value.py'):
        sources[name]=sha(ROOT/name)
    p['source_sha256']=sources
    p['reference_sha256']={str(path.relative_to(ROOT)):sha(path) for path in
        (OUT.parent/'component-preflight.json',OUT.parent/'cpu-scene-fields.npz',
         ROOT/'records/reports/getup-teacher-anchor-pair/contact-duration/analysis.json')}
    for base,key in ((ROOT,'source_sha256'),(WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
        for name,digest in p[key].items():assert sha(base/name)==digest,name
    cfg,display,_=prepare(p);fields=model_arrays(Scene(cfg.env.scene,device='cpu').compile())
    with np.load(OUT.parent/'cpu-scene-fields.npz') as f:compare_arrays(fields,{k:f[k] for k in f.files})
    assert cfg.env.episode_length_s==8 and cfg.env.is_finite_horizon and not cfg.env.auto_reset
    assert not torch.cuda.is_initialized()
    OUT.mkdir();np.savez_compressed(OUT/'cpu-scene-fields.npz',**fields);write(PLAN,p)
    write(OUT/'preflight.json',dict(status='passed-cpu-only',protocol_sha256=sha(PLAN),source_count=len(sources),
        config=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader),display=display,
        cuda_initialized=False,physics_steps=0,actor_updates=0,critic_width=76))
    print('PERSISTENT_VALUE_FROZEN',sha(PLAN),len(sources),'sources;400controls;50criticupdates;0actor',flush=True)


def checked():
    p=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    assert sha(ROOT/p['model'])==p['model_sha256']
    for base,key in ((ROOT,'source_sha256'),(WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
        for name,digest in p[key].items():assert sha(base/name)==digest,name
    for name,digest in p['reference_sha256'].items():assert sha(ROOT/name)==digest,name
    return p


def sum_rewards(raw,names,weights,dt,potential=None):
    result=torch.zeros_like(raw[0])
    for name,value,weight in zip(names,raw,weights):
        if name=='late_head_leg_contact' and potential is not None:value=potential
        result+=value*weight*dt
    return result


def reset_test(env,obs,p,record):
    state=clock(env,p['contact_threshold_seconds']);ids=torch.nonzero(state.ticks>0).flatten()[:2]
    assert len(ids)==2,'No positive clocks to exercise the frozen reset check'
    selected=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device);selected[ids]=True
    kept=~selected;before=dict(ticks=state.ticks.clone(),last_step=state.last_step.clone(),
        qpos=env.sim.data.qpos.clone(),qvel=env.sim.data.qvel.clone(),episode_age=env.episode_length_buf.clone())
    histories={k:getattr(env.action_manager,k).clone() for k in ('_action','_prev_action','_prev_prev_action')}
    step=env.common_step_counter;after,_=env.reset(env_ids=ids)
    assert env.common_step_counter==step
    torch.testing.assert_close(state.ticks[ids],torch.zeros_like(state.ticks[ids]),atol=0,rtol=0)
    torch.testing.assert_close(state.ticks[kept],before['ticks'][kept],atol=0,rtol=0)
    torch.testing.assert_close(state.last_step[kept],before['last_step'][kept],atol=0,rtol=0)
    for key in ('qpos','qvel'):
        torch.testing.assert_close(getattr(env.sim.data,key)[kept],before[key][kept],atol=0,rtol=0)
    for key,value in histories.items():torch.testing.assert_close(getattr(env.action_manager,key)[kept],value[kept],atol=0,rtol=0)
    torch.testing.assert_close(after['critic'][:,74],state.ticks.float()/60,atol=0,rtol=0)
    cpu=lambda x:x.detach().cpu().numpy().copy()
    arrays={f'before_{key}':cpu(value) for key,value in before.items()}
    arrays.update(ids=cpu(ids),after_ticks=cpu(state.ticks),after_last_step=cpu(state.last_step),
        after_qpos=cpu(env.sim.data.qpos),after_qvel=cpu(env.sim.data.qvel),after_episode_age=cpu(env.episode_length_buf))
    for key,value in histories.items():arrays['before'+key]=cpu(value);arrays['after'+key]=cpu(getattr(env.action_manager,key))
    np.savez_compressed(record/'selected-reset.npz',**arrays)
    proof=dict(status='passed-actual-gpu-reset',control_step=step,ids=cpu(ids).tolist(),
        positive_memory_cleared=True,other_physics_history_memory_exact=True,physics_steps_added=0,
        postterminal_only=True,reset_states_not_used_for_fit=True)
    write(record/'selected-reset.json',proof);return proof


def run():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.os import dump_yaml
    from evaluate_standup_reference import Tee
    assert os.isatty(1),'Use the original visible VS Code shared terminal'
    p=checked();assert not (OUT/'progress.json').exists(),'No retry or extension'
    project=read(ROOT/'project.json');assert project['staged_training_request']['current_stage']==STAGE
    project['staged_training_request'].update(simulation_running=True,learning_running=False);write(ROOT/'project.json',project)
    record=ROOT/'records/experiments'/p['run_id'];record.mkdir();write(record/'protocol.json',p)
    for name in p['source_sha256']:
        target=record/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    state=dict(status='initializing',phase='fixed-policy-value-preparation',run_id=p['run_id'],path=str(record),pid=os.getpid(),
        protocol_sha256=sha(PLAN),actor_updates=0,critic_updates=0,completed_updates=0,control_steps=0,
        heldout_used=False,hardware_io=False,started_at=datetime.now(timezone.utc).isoformat())
    env=snapshot=video=None;data=None;completed=0;peak=0;reset_completed=False;fit_started=False
    def publish(**values):
        state.update(values)
        if env is not None:state['control_steps']=int(env.common_step_counter)
        if video is not None:state.update(clips=video.clips,images=[video.image] if video.image else [])
        write(record/'state.json',state);write(OUT/'progress.json',state)
    publish();write(record/'command.json',dict(argv=sys.argv,pid=os.getpid(),foreground_tty=os.isatty(1)))
    write(ROOT/'records/training-current.json',dict(path=str(record),record=str(record),run_id=p['run_id'],pid=os.getpid(),purpose='contact-duration-shared-critic-admission'))
    with (record/'execution.log').open('x',buffering=1) as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            configure_torch_backends(allow_tf32=False);torch.set_num_threads(4)
            cfg,display,_=prepare(p);write(record/'display.json',display);dump_yaml(record/'env.yaml',asdict(cfg))
            env=ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None);env.seed(p['seed']);obs,_=env.reset()
            cpu=lambda x:x.detach().cpu().numpy().copy()
            rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state(env.device).clone()
            actor,critic=networks(p,env.device)
            assert torch.equal(rng,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state(env.device))
            startup(env,obs,actor,p,record,OUT/'cpu-scene-fields.npz')
            initial={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
            torch.save(dict(actor_state_dict=initial,critic_state_dict=critic.state_dict(),actor_updates=0,critic_updates=0),record/'model-init.pt')
            initial_qpos=cpu(env.sim.data.qpos);initial_qvel=cpu(env.sim.data.qvel)
            names=list(env.reward_manager._term_names);term_cfgs=env.reward_manager._term_cfgs
            weights=[float(c.weight) for c in term_cfgs];raw={}
            assert names[-1]=='late_head_leg_contact' and weights[-1]==-2.
            for name,term_cfg in zip(names,term_cfgs):
                func=term_cfg.func
                def capture(actual_env,*,_func=func,_name=name,**params):
                    value=_func(actual_env,**params);raw[_name]=value.clone();return value
                term_cfg.func=capture
            write(record/'reward-terms.json',dict(names=names,weights=weights,scale_by_dt=True))
            fields=(('critic_obs',(400,1024,76),np.float32),('rewards',(400,1024),np.float32),
                ('candidate_rewards',(400,1024),np.float32),('raw_reward_terms',(400,1024,len(names)),np.float32),
                ('actions',(400,1024,14),np.float32),('raw_history',(400,1024,14),np.float32),
                ('actor_obs_sample',(400,32,61),np.float32),('terminal',(400,1024),bool),
                ('contact',(400,1024),bool),('eligible',(400,1024),bool),('ticks',(400,1024),np.int64),
                ('potential',(400,1024),np.float32),('requested_targets',(400,1024,14),np.float32),
                ('applied_targets',(400,1024,14),np.float32),('qpos',(400,32,21),np.float32),('qvel',(400,32,20),np.float32))
            data={k:np.empty(shape,dtype=dtype) for k,shape,dtype in fields}
            snapshot=RecordingSnapshot(env,record);video=ActionVideo(snapshot,'getup',record);video.sample()
            term=env.action_manager.get_term('joint_pos');slots=int(env.sim.wp_data.contact.dist.shape[0]);start=time.monotonic()
            print('VISIBLE_PERSISTENT_VALUE: 400x1024;freshcritic76;50minibatches;actor0;full8s',flush=True)
            for t in range(400):
                assert env.common_step_counter==t and (env.episode_length_buf==t).all()
                data['critic_obs'][t]=cpu(obs['critic']);data['raw_history'][t]=cpu(obs['actor'][:,34:48]);data['actor_obs_sample'][t]=cpu(obs['actor'][:32])
                with torch.no_grad():actions=actor(obs,stochastic_output=True)
                sampled=actions.clone();raw.clear();obs,reward,terminated,truncated,_=env.step(actions)
                torch.testing.assert_close(actions,sampled,atol=0,rtol=0)
                torch.testing.assert_close(obs['actor'][:,34:48],sampled if t>=100 else torch.zeros_like(sampled),atol=0,rtol=0)
                assert not terminated.any() and (not truncated.any() if t<399 else bool(truncated.all()))
                potential=env._persistent_contact['potential']
                values=[raw[name] if weight else torch.zeros_like(reward) for name,weight in zip(names,weights)]
                reconstructed=sum_rewards(values,names,weights,.02)
                torch.testing.assert_close(reconstructed,reward,atol=0,rtol=0)
                assert not env._persistent_contact['cost'].any()
                candidate=sum_rewards(values,names,weights,.02,potential)
                data['rewards'][t]=cpu(reward);data['candidate_rewards'][t]=cpu(candidate)
                data['raw_reward_terms'][t]=cpu(torch.stack(values,-1));data['actions'][t]=cpu(sampled)
                data['terminal'][t]=cpu(terminated|truncated)
                for key in ('contact','eligible','ticks','potential'):data[key][t]=cpu(env._persistent_contact[key])
                data['requested_targets'][t]=cpu(term.requested_target);data['applied_targets'][t]=cpu(term._processed_actions)
                data['qpos'][t]=cpu(env.sim.data.qpos[:32]);data['qvel'][t]=cpu(env.sim.data.qvel[:32])
                torch.testing.assert_close(obs['critic'][:,74],clock(env,1.2).ticks.float()/60,atol=0,rtol=0)
                assert all(np.isfinite(data[k][t]).all() for k in data)
                count=int(env.sim.wp_data.nacon.numpy()[0]);peak=max(peak,count);assert 0<=count<slots
                completed=t+1;video.sample()
                if completed%25==0:
                    publish(status='sampling',control_transitions=completed*1024,persistent_worlds=int((data['ticks'][t]>=60).sum()),elapsed_seconds=time.monotonic()-start)
                    print('PERSISTENT_VALUE_DATA',completed,'/400;persistent',state['persistent_worlds'],';wall',round(time.monotonic()-start,1),flush=True)
                    assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
            video.close();snapshot.close();snapshot=None
            write(record/'contact-audits.json',env._getup_contact_audits)
            reset_test(env,obs,p,record);reset_completed=True
            np.savez_compressed(record/'value-data.npz',**data,initial_qpos=initial_qpos,initial_qvel=initial_qvel)
            fit_data=dict(data,rewards=(data['rewards'].astype(np.float64)+data['candidate_rewards'].astype(np.float64))*.5)
            fit_started=True
            project=read(ROOT/'project.json');project['staged_training_request'].update(learning_running=True,critic_fitting_running=True);write(ROOT/'project.json',project)
            result=fit(critic,fit_data,p,record,publish)
            with torch.no_grad():
                valid=np.asarray(p['split']['validation_ids']);x=torch.from_numpy(data['critic_obs'][:,valid].reshape(-1,76)).to(env.device)
                prediction=torch.cat([critic(td('critic',v)).squeeze(-1) for v in x.split(4096)]).cpu().numpy().reshape(400,64)
            scores={};codes=np.asarray(p['split']['pose_codes'])[valid];gate=p['value_gate'];admitted=True
            from getup_anchor_value_check import score
            for label,key in [('control','rewards'),('candidate','candidate_rewards')]:
                returns=discounted_returns(torch.from_numpy(data[key][:,valid]),p['gamma']).numpy()/result['reward_scale']
                overall=score(prediction,returns);poses={name:score(prediction[:,codes==i],returns[:,codes==i]) for i,name in enumerate(POSES)}
                okay=overall['explained_variance']>=gate['validation_explained_variance_min'] and overall['normalized_mse']<=gate['validation_normalized_mse_max'] and all(v['explained_variance']>=0 for v in poses.values())
                scores[label]=dict(validation=overall,poses=poses,gate_passed=bool(okay));admitted&=bool(okay)
            assert all(torch.equal(v,actor.state_dict()[k].detach().cpu()) for k,v in initial.items())
            torch.save(dict(actor_state_dict=actor.state_dict(),actor_updates=0),record/'actor-final.pt')
            write(record/'result.json',dict(status='passed-integrity',both_value_gates_passed=bool(admitted),
                scores=scores,common_mean_fit=result,reward_scale=result['reward_scale'],actor_bitwise_unchanged=True,
                actor_updates=0,critic_updates=50,actual_control_transitions=409600,heldout_used=False))
            publish(status='budget-complete-review-required',phase='closed',both_value_gates_passed=bool(admitted),critic_updates=50,
                completed_updates=50,dataset_sha256=sha(record/'value-data.npz'),finished_at=datetime.now(timezone.utc).isoformat())
            print('PERSISTENT_VALUE_CLOSED',json.dumps(scores),'bothgates',bool(admitted),flush=True)
        except BaseException as error:
            publish(status='failed-preserved',error=repr(error));traceback.print_exc()
            if data is not None and completed:np.savez_compressed(record/'partial-data.npz',**{k:v[:completed] for k,v in data.items()})
            raise
        finally:
            if video is not None:video.close()
            if snapshot is not None:snapshot.close()
            if env is not None:
                write(record/'runtime.json',dict(control_steps=int(env.common_step_counter),control_transitions=int(env.common_step_counter)*1024,
                    physics_substeps=int(env.common_step_counter)*5,extra_observation_or_render_steps=0,
                    manual_resets=1+int(reset_completed),postterminal_reset=int(reset_completed),automatic_resets=0,
                    actor_updates=0,critic_updates=50 if (record/'critic-final.pt').exists() else None if fit_started else 0,
                    critic_updates_exact=(record/'critic-final.pt').exists() or not fit_started,
                    contact_peak=peak,contact_slots=slots if 'slots' in locals() else None))
                if state['status']=='failed-preserved':np.savez_compressed(record/'stop-state.npz',qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel))
                env.close()
            project=read(ROOT/'project.json');project['staged_training_request'].update(simulation_running=False,learning_running=False,critic_fitting_running=False);write(ROOT/'project.json',project)
            sys.stdout,sys.stderr=stdout,stderr


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--verify',action='store_true');a=parser.parse_args()
    if a.freeze:freeze()
    elif a.verify:
        from verify_getup_persistent_value import verify
        verify()
    else:
        run();subprocess.run([sys.executable,'-B',str(Path(__file__)),'--verify'],cwd=WORK,check=True)
