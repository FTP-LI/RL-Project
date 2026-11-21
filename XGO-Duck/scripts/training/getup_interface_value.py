"""Fresh value admission for the intended full-contact learning interface.

Changes from the closed value probe: use common walking-level actor noise,
the verified post5s projected2rad/s target processor, and fixed0.10rad action
exploration. Keep official actor coordinates/history, the physical2s freeze,
the8s finite episode and original rewards. Fit a NEW critic; never inherit an
admission obtained with different transitions. No actor fitting in this run.
CPU checks cover the real processor and complete scene; live zero-step checks
must pass before400controls. Capture one full8s episode, not a mid-action clip.
"""
from copy import deepcopy
from dataclasses import replace,asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,time,traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/validation'))
from s1_reference_inputs import WORK,read,sha
from record_io import write
from getup_value_bootstrap import networks,td,fit
from getup_contact_training import prepare as original_prepare,split_worlds
from getup_stand_handoff import apply_common_actor_noise,adapt_action,NOISE
from action_video import ActionVideo

PLAN=ROOT/'configs/experiments/getup_interface_value.json'
OUT=ROOT/'records/reports/getup-interface-value'


def prepare(p):
    cfg,display,meta=original_prepare(p)
    cfg=replace(cfg,env=adapt_action(apply_common_actor_noise(cfg.env),p['standing_home'],
                                   p['joint_limits'],p['handoff_settings']))
    assert cfg.env.episode_length_s==8. and not cfg.env.auto_reset
    assert cfg.env.is_finite_horizon and not cfg.env.curriculum
    assert not cfg.env.rewards['late_head_leg_contact'].params['enabled']
    display.update(recorder='ActionVideo',duration='Complete8s first episode; no added physics',
                   native_live_window=False)
    return cfg,display,meta


def freeze():
    from mjlab.scene import Scene
    from common_scene_contract import model_arrays,compare_arrays
    from check_getup_stand_handoff import real_processor
    from getup_release_coverage import cpu_check
    import yaml
    assert not PLAN.exists() and not OUT.exists(),'Preserve frozen attempts'
    parent=read(ROOT/'configs/experiments/getup_neck_home_order.json')
    old=read(ROOT/'configs/experiments/getup_value_bootstrap.json')
    audit=read(ROOT/'records/reports/getup-target-feasibility/analysis.json')
    assert audit['status']=='reviewed-saved-data-only'
    p=deepcopy(parent['arms']['control'])
    p.update(stage='S3-common-interface-value-admission',status='frozen-before-data',
        seed=42,num_envs=1024,split=split_worlds(seed=4302),split_seed=4302,
        episode_seconds=8.,maximum_control_steps=400,maximum_control_transitions=409600,
        initial_std=.10,minibatch_seed=443,critic_hidden_dims=[512,256,128],
        critic_updates=50,minibatch_size=8192,critic_learning_rate=.0003,
        gamma=.99,gradient_norm_max=1.,training_metric_samples=8192,
        evaluation_updates=[0,1,10,25,50],value_gate=deepcopy(old['value_gate']),
        critic_remaining_time=True,is_finite_horizon=True,head_leg_cost_enabled=False,
        actor_updates=0,actor_frozen=True,handoff_enabled=False,neck_home_enabled=False,
        actor_noise=deepcopy(NOISE),learning_updates=0,
        reward_scale='Train-world RMS of complete finite8s discounted returns',
        returns='Monte Carlo gamma0.99, genuine finite8s terminal0; no bootstrap or resets',
        validation='64development worlds excluded from scale, normalization and fitting; not final heldout',
        source_audit_sha256=sha(ROOT/'records/reports/getup-target-feasibility/analysis.json'),
        automatic_retry=False,automatic_extension=False,heldout_used=False,hardware_io=False,
        scope='Fresh matched-interface critic admission only; no actor or gait improvement')
    cfg,display,meta=prepare(p)
    fields=model_arrays(Scene(cfg.env.scene,device='cpu').compile())
    with np.load(ROOT/'records/reports/getup-value-bootstrap/cpu-scene-fields.npz') as f:
        compare_arrays(fields,{k:f[k] for k in f.files})
    fixture=real_processor(cfg.env,p['model_home'],p['standing_home'])
    numeric=cpu_check(dict(roles={'getup':p,'standing':dict(p,model=p['standing_model'],model_sha256=p['standing_model_sha256'])}))
    sources=deepcopy(parent['source_sha256'])
    for name in ('scripts/training/getup_interface_value.py',
                 'scripts/validation/verify_getup_interface_value.py',
                 'scripts/validation/review_getup_targets.py'):
        sources[name]=sha(ROOT/name)
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name
    for name,digest in p['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    assert not torch.cuda.is_initialized()
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S')
    p.update(source_sha256=sources,run_id=f'duck_getup_interface-value_reference_s42_{stamp}',
             frozen_at=datetime.now(timezone.utc).isoformat())
    OUT.mkdir();np.savez_compressed(OUT/'cpu-scene-fields.npz',**fields);write(PLAN,p)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),fixture=fixture,
        model_import=numeric,config=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader),
        physical_scene_exact=True,actor_width=61,critic_width=75,source_count=len(sources),
        cuda_initialized=False,physical_steps=0,actor_updates=0,display=display))
    print('INTERFACE_VALUE_FROZEN',sha(PLAN),flush=True)


def checked():
    p=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    assert sha(ROOT/p['model'])==p['model_sha256']
    for base,key in ((ROOT,'source_sha256'),(WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
        for name,digest in p[key].items():assert sha(base/name)==digest,name
    return p


def startup(env,obs,actor,p,record):
    from collect_getup_stand_handoff import startup as check_start
    from getup_release_coverage import actor_model
    from common_scene_contract import model_arrays,compare_arrays
    standing=actor_model(dict(p,model=p['standing_model'],model_sha256=p['standing_model_sha256']),env.device)
    check_start(env,obs,dict(getup=actor,standing=standing),p,record)
    with np.load(OUT/'cpu-scene-fields.npz') as f:
        compare_arrays(model_arrays(env.sim.mj_model),{k:f[k] for k in f.files})
    # Inspect the actual stochastic scale while preserving both RNG streams.
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state(env.device).clone()
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        actor(obs,stochastic_output=True)
        torch.testing.assert_close(actor.output_std,torch.full_like(actor.output_std,p['initial_std']),atol=1e-7,rtol=0)
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state(env.device))
    proof=read(record/'startup.json');proof.update(common_noise=NOISE,initial_std=p['initial_std'],
        new_cpu_scene_exact=True,stochastic_probe_rng_restored=True,physics_steps=0)
    write(record/'startup.json',proof)


def run():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from training_watch import Snapshot
    from evaluate_standup_reference import Tee
    assert os.isatty(1),'Use the original visible VS Code shared task terminal'
    p=checked();assert not (OUT/'progress.json').exists(),'No automatic retry'
    record=ROOT/'records/experiments'/p['run_id'];record.mkdir();write(record/'protocol.json',p)
    for name in p['source_sha256']:shutil.copy2(ROOT/name,record/Path(name).name)
    state=dict(status='training',phase='starting',run_id=p['run_id'],path=str(record),pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PLAN),
        control_steps=0,actor_updates=0,critic_updates=0,completed_updates=0,heldout_used=False,hardware_io=False)
    env=snapshot=video=None;peak=0;data=None
    def publish(**values):
        state.update(values)
        if env is not None:state['control_steps']=int(env.common_step_counter)
        if video is not None:state.update(clips=video.clips,images=[video.image] if video.image else [])
        write(record/'state.json',state);write(OUT/'progress.json',state)
    publish();write(ROOT/'records/training-current.json',dict(path=str(record),record=str(record),
        run_id=p['run_id'],pid=os.getpid(),purpose='common-interface-fixed-actor-value-admission'))
    write(record/'command.json',dict(argv=sys.argv,foreground_tty=os.isatty(1),pid=os.getpid(),started_at=state['started_at']))
    with (record/'execution.log').open('x',buffering=1) as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            print('VISIBLE_INTERFACE_VALUE: official actor fixed;400controls x1024;50critic minibatches; full8s clip',flush=True)
            configure_torch_backends(allow_tf32=False);torch.set_num_threads(4)
            cfg,display,_=prepare(p);write(record/'display.json',display)
            env=ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None);env.seed(p['seed']);obs,_=env.reset()
            actor,critic=networks(75,p,env.device);startup(env,obs,actor,p,record)
            initial={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
            torch.save(dict(actor_state_dict=initial,critic_state_dict=critic.state_dict()),record/'model-init.pt')
            snapshot=Snapshot(env,record);video=ActionVideo(snapshot,'getup',record);video.sample()
            n=1024;steps=400;width=75
            fields=(('critic_obs',(steps,n,width),np.float32),('rewards',(steps,n),np.float32),
                ('actions',(steps,n,14),np.float32),('raw_history',(steps,n,14),np.float32),
                ('actor_obs_sample',(steps,32,61),np.float32),('terminal',(steps,n),bool),
                ('head_leg',(steps,n),bool),('eligible',(steps,n),bool),('cost',(steps,n),np.float32),
                ('requested_targets',(steps,n,14),np.float32),('applied_targets',(steps,n,14),np.float32),
                ('qpos',(steps,32,21),np.float32),('qvel',(steps,32,20),np.float32))
            data={key:np.empty(shape,dtype=dtype) for key,shape,dtype in fields}
            cpu=lambda x:x.detach().cpu().numpy().copy()
            start=time.monotonic();slots=int(env.sim.wp_data.contact.dist.shape[0])
            term=env.action_manager.get_term('joint_pos')
            for t in range(steps):
                assert env.common_step_counter==t and (env.episode_length_buf==t).all()
                assert not term.standing_mask.any()
                data['critic_obs'][t]=cpu(obs['critic']);data['raw_history'][t]=cpu(obs['actor'][:,34:48])
                data['actor_obs_sample'][t]=cpu(obs['actor'][:32])
                with torch.no_grad():actions=actor(obs,stochastic_output=True)
                torch.testing.assert_close(actor.output_std,torch.full_like(actor.output_std,.10),atol=1e-7,rtol=0)
                sampled=actions.clone();data['actions'][t]=cpu(sampled)
                obs,reward,terminated,truncated,_=env.step(actions)
                torch.testing.assert_close(actions,sampled,atol=0,rtol=0)
                torch.testing.assert_close(obs['actor'][:,34:48],sampled if t>=100 else torch.zeros_like(sampled),atol=0,rtol=0)
                assert not terminated.any() and (not truncated.any() if t<399 else bool(truncated.all()))
                assert torch.isfinite(reward).all() and torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all()
                data['rewards'][t]=cpu(reward);data['terminal'][t]=cpu(terminated|truncated)
                for key,name in (('head_leg','contact'),('eligible','eligible'),('cost','cost')):data[key][t]=cpu(env._getup_contact[name])
                data['requested_targets'][t]=cpu(term.requested_target);data['applied_targets'][t]=cpu(term.processed_actions)
                data['qpos'][t]=cpu(env.sim.data.qpos[:32]);data['qvel'][t]=cpu(env.sim.data.qvel[:32])
                count=int(env.sim.wp_data.nacon.numpy()[0]);peak=max(peak,count);assert 0<=count<slots
                video.sample()
                if (t+1)%25==0:
                    assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
                    publish(control_transitions=(t+1)*n,elapsed_seconds=time.monotonic()-start,
                            current_head_leg_worlds=int(data['head_leg'][t].sum()))
                    print('INTERFACE_VALUE_DATA',t+1,'/400; head-leg',int(data['head_leg'][t].sum()),
                          '; wall',round(time.monotonic()-start,1),flush=True)
            video.close();write(record/'contact-audits.json',env._getup_contact_audits)
            assert len(env._getup_contact_audits)==16 and any(x['contact_worlds']>0 for x in env._getup_contact_audits)
            publish(phase='archive-samples',control_transitions=409600,contact_peak=peak,contact_slots=slots)
            temp=record/'value-data.npz.partial'
            with temp.open('xb') as f:np.savez_compressed(f,**data)
            temp.replace(record/'value-data.npz')
            result=fit(critic,data,p,record,publish)
            assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in actor.state_dict().items())
            torch.save(dict(actor_state_dict=actor.state_dict(),actor_updates=0),record/'actor-final.pt')
            write(record/'result.json',dict(status='passed',actor_bitwise_unchanged=True,**result))
            publish(status='budget-complete-review-required',phase='closed',critic_updates=50,completed_updates=50,
                actor_updates=0,value_gate_passed=result['value_gate_passed'],actor_bitwise_unchanged=True,
                dataset_sha256=sha(record/'value-data.npz'),elapsed_seconds=time.monotonic()-start,
                finished_at=datetime.now(timezone.utc).isoformat())
            print('INTERFACE_VALUE_COMPLETE',json.dumps(result),flush=True)
        except BaseException as error:
            publish(status='failed-preserved',phase='stopped',error=repr(error));traceback.print_exc();raise
        finally:
            if video is not None:video.close()
            if snapshot is not None:snapshot.close()
            if env is not None:
                write(record/'runtime.json',dict(control_steps=int(env.common_step_counter),
                    control_transitions=int(env.common_step_counter)*1024,physics_substeps=int(env.common_step_counter)*5,
                    extra_observation_or_render_steps=0,manual_resets=1,automatic_resets=0,actor_updates=0,contact_peak=peak))
                if state['status']=='failed-preserved':
                    np.savez_compressed(record/'stop-state.npz',qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel))
                env.close()
            sys.stdout,sys.stderr=stdout,stderr


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:freeze()
    else:
        run()
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validation/verify_getup_interface_value.py')],check=True)
