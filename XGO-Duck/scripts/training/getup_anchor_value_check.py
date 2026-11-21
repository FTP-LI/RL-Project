"""One visible new-seed admission of the existing common-interface critic.

Before the new anchored pair, test the frozen official0.10rad behavior and
unchanged existing critic on400x1024new transitions. Do not fit either model,
update normalization, recompute its reward scale or consume final heldouts.
An unsuitable value model closes this protocol; it is not silently refit.
Capture a complete8s first episode and require actualGPUzero-step admission.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,time,traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/validation'))
from s1_reference_inputs import WORK,read,sha
from record_io import write
from getup_interface_value import prepare
from getup_value_bootstrap import networks,td
from getup_contact_training import split_worlds,discounted_returns,POSES
from getup_anchor_runtime import startup,RecordingSnapshot
from action_video import ActionVideo

PLAN=ROOT/'configs/experiments/getup_anchor_value_check.json'
OUT=ROOT/'records/reports/getup-anchor-value-check'
STAGE='S3-anchor-value-admission'


def score(prediction,target):
    error=prediction.astype(np.float64)-target
    variance=np.var(target);assert variance>1e-10
    return dict(normalized_mse=float(np.mean(error**2)),explained_variance=float(1-np.var(error)/variance),bias=float(error.mean()))


def freeze():
    from mjlab.scene import Scene
    from common_scene_contract import model_arrays,compare_arrays
    from check_getup_stand_handoff import real_processor
    import yaml
    assert not PLAN.exists() and not OUT.exists()
    proofpath=ROOT/'records/reports/getup-teacher-anchor/ppo-preflight.json';proof=read(proofpath)
    assert proof['status']=='passed-cpu-optimizer-fixture' and proof['zero_coefficient_actor_critic_and_adam_bitwise_equal']
    assert proof['ppo_sha256']==sha(ROOT/'scripts/training/getup_anchor_ppo.py')
    admittedpath=ROOT/'records/reports/getup-interface-value/readback-recovery/analysis.json';admitted=read(admittedpath)
    assert admitted['status']=='passed' and admitted['value_gate_passed']
    parent=read(ROOT/'configs/experiments/getup_late_feasibility_pair.json')
    origin=ROOT/admitted['record'];p=deepcopy(read(origin/'protocol.json'))
    p.pop('readback_recovery',None)
    p.update(status='frozen-before-new-seed-data',stage=STAGE,seed=43,split_seed=4303,
        split=split_worlds(seed=4303),run_id='duck_getup_anchor-value_reference_s43_'+datetime.now().strftime('%Y%m%dT%H%M%S'),
        actor_updates=0,critic_updates=0,learning_updates=0,maximum_control_steps=400,maximum_control_transitions=409600,
        critic_checkpoint=str((origin/'critic-final.pt').relative_to(ROOT)),critic_sha256=admitted['critic_sha256'],
        reward_scale=admitted['reward_scale'],source_value_admission=str(admittedpath.relative_to(ROOT)),
        source_value_admission_sha256=sha(admittedpath),ppo_preflight_sha256=sha(proofpath),
        source_critic_optimizer_updates=50,heldout_used=False,hardware_io=False,
        automatic_retry=False,automatic_extension=False,frozen_at=datetime.now(timezone.utc).isoformat(),
        scope='Newdevelopmentseed43 fixedactor and fixedcritic admission only; no learning or gait improvement')
    assert sha(ROOT/p['critic_checkpoint'])==p['critic_sha256'] and p['initial_std']==.1
    sources=dict(parent['source_sha256'])
    for name in ('scripts/training/getup_teacher_anchor.py','scripts/training/getup_anchor_ppo.py',
        'scripts/validation/check_getup_anchor_ppo.py','scripts/training/getup_anchor_runtime.py',
        'scripts/training/getup_anchor_value_check.py','scripts/validation/verify_getup_anchor_value.py'):
        sources[name]=sha(ROOT/name)
    p['source_sha256']=sources
    reference_paths=[str(proofpath.relative_to(ROOT)),str(admittedpath.relative_to(ROOT)),
        'records/reports/getup-teacher-anchor/preparation.json',
        'records/reports/getup-interface-value/readback-recovery/cpu-scene-fields.npz']
    p['reference_sha256']={name:sha(ROOT/name) for name in reference_paths}
    for root,key in ((ROOT,'source_sha256'),(WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
        for name,expected in p[key].items():assert sha(root/name)==expected,name
    cfg,display,_=prepare(p)
    assert not cfg.env.auto_reset and cfg.env.is_finite_horizon and cfg.env.episode_length_s==8. and cfg.env.scene.num_envs==1024
    fields=model_arrays(Scene(cfg.env.scene,device='cpu').compile())
    with np.load(ROOT/'records/reports/getup-interface-value/readback-recovery/cpu-scene-fields.npz') as f:
        compare_arrays(fields,{k:f[k] for k in f.files})
    processor=real_processor(cfg.env,p['model_home'],p['standing_home'])
    assert not torch.cuda.is_initialized()
    OUT.mkdir();np.savez_compressed(OUT/'cpu-scene-fields.npz',**fields);write(PLAN,p)
    write(OUT/'preflight.json',dict(status='passed-cpu-only',protocol_sha256=sha(PLAN),source_count=len(sources),
        physical_scene_exact=True,processor=processor,actor_width=61,critic_width=75,cuda_initialized=False,
        actor_updates=0,critic_updates=0,source_critic_sha256=p['critic_sha256'],parent_scale_unchanged=p['reward_scale'],
        config=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader),display=display))
    print('ANCHOR_VALUE_CHECK_FROZEN',sha(PLAN),len(sources),'sources;400controls,0gradients',flush=True)


def checked():
    p=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    for root,key in ((ROOT,'source_sha256'),(WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
        for name,expected in p[key].items():assert sha(root/name)==expected,name
    for name,expected in p['reference_sha256'].items():assert sha(ROOT/name)==expected,name
    assert sha(ROOT/p['model'])==p['model_sha256'] and sha(ROOT/p['critic_checkpoint'])==p['critic_sha256']
    return p


def run():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from evaluate_standup_reference import Tee
    assert os.isatty(1),'Use the original one visible VS Code shared task terminal'
    p=checked();assert not (OUT/'progress.json').exists(),'No restart or extension'
    project=read(ROOT/'project.json');assert project['staged_training_request']['current_stage']==STAGE
    project['staged_training_request'].update(simulation_running=True,learning_running=False);write(ROOT/'project.json',project)
    record=ROOT/'records/experiments'/p['run_id'];record.mkdir();write(record/'protocol.json',p)
    for name in p['source_sha256']:
        target=record/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    state=dict(status='initializing',phase='fixed-value-check',run_id=p['run_id'],path=str(record),pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PLAN),
        control_steps=0,actor_updates=0,critic_updates=0,completed_updates=0,heldout_used=False,hardware_io=False)
    env=snapshot=video=None;data=None;completed=0;peak=0
    def publish(**values):
        state.update(values)
        if env is not None:state['control_steps']=int(env.common_step_counter)
        if video is not None:state.update(clips=video.clips,images=[video.image] if video.image else [])
        write(record/'state.json',state);write(OUT/'progress.json',state)
    publish();write(record/'command.json',dict(argv=sys.argv,pid=os.getpid(),foreground_tty=os.isatty(1),started_at=state['started_at']))
    write(ROOT/'records/training-current.json',dict(path=str(record),record=str(record),run_id=p['run_id'],pid=os.getpid(),purpose='frozen-actor-and-critic-new-seed-value-check'))
    with (record/'execution.log').open('x',buffering=1) as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            configure_torch_backends(allow_tf32=False);torch.set_num_threads(4)
            cfg,display,_=prepare(p);write(record/'display.json',display)
            from mjlab.utils.os import dump_yaml
            dump_yaml(record/'env.yaml',asdict(cfg))
            env=ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None);env.seed(p['seed']);obs,_=env.reset()
            cpu=lambda x:x.detach().cpu().numpy().copy()
            before_cpu=torch.get_rng_state().clone();before_gpu=torch.cuda.get_rng_state(env.device).clone()
            actor,critic=networks(75,p,env.device)
            warm=torch.load(ROOT/p['critic_checkpoint'],map_location='cpu',weights_only=True)
            assert warm['reward_scale']==p['reward_scale'];critic.load_state_dict(warm['critic_state_dict'])
            actor.eval().requires_grad_(False);critic.eval().requires_grad_(False)
            startup(env,obs,actor,p,record,OUT/'cpu-scene-fields.npz')
            assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_gpu,torch.cuda.get_rng_state(env.device))
            write(record/'complete-import-rng.json',dict(status='passed',cpu_and_cuda_rng_unchanged=True,physical_steps=0))
            initial={key:{k:v.detach().cpu().clone() for k,v in net.state_dict().items()} for key,net in [('actor',actor),('critic',critic)]}
            torch.save(initial,record/'model-init.pt')
            initial_qpos=cpu(env.sim.data.qpos);initial_qvel=cpu(env.sim.data.qvel)
            snapshot=RecordingSnapshot(env,record);video=ActionVideo(snapshot,'getup',record);video.sample()
            n=1024;steps=400
            sizes=(('critic_obs',(steps,n,75),np.float32),('critic_predictions',(steps,n),np.float32),
                ('rewards',(steps,n),np.float32),('actions',(steps,n,14),np.float32),('raw_history',(steps,n,14),np.float32),
                ('actor_obs_sample',(steps,32,61),np.float32),('terminal',(steps,n),bool),
                ('requested_targets',(steps,n,14),np.float32),('applied_targets',(steps,n,14),np.float32),
                ('qpos',(steps,32,21),np.float32),('qvel',(steps,32,20),np.float32))
            data={key:np.empty(shape,dtype=dtype) for key,shape,dtype in sizes};term=env.action_manager.get_term('joint_pos')
            slots=int(env.sim.wp_data.contact.dist.shape[0]);start=time.monotonic();publish(status='sampling')
            for t in range(steps):
                assert env.common_step_counter==t and (env.episode_length_buf==t).all() and not term.standing_mask.any()
                data['critic_obs'][t]=cpu(obs['critic']);data['raw_history'][t]=cpu(obs['actor'][:,34:48]);data['actor_obs_sample'][t]=cpu(obs['actor'][:32])
                with torch.no_grad():
                    actions=actor(obs,stochastic_output=True);data['critic_predictions'][t]=cpu(critic(obs).squeeze(-1))
                torch.testing.assert_close(actor.output_std,torch.full_like(actor.output_std,.1),atol=1e-7,rtol=0)
                sampled=actions.clone();data['actions'][t]=cpu(sampled)
                obs,reward,terminated,truncated,_=env.step(actions)
                torch.testing.assert_close(actions,sampled,atol=0,rtol=0)
                torch.testing.assert_close(obs['actor'][:,34:48],sampled if t>=100 else torch.zeros_like(sampled),atol=0,rtol=0)
                assert not terminated.any() and (not truncated.any() if t<399 else bool(truncated.all()))
                assert torch.isfinite(reward).all() and torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all()
                data['rewards'][t]=cpu(reward);data['terminal'][t]=cpu(terminated|truncated)
                data['requested_targets'][t]=cpu(term.requested_target);data['applied_targets'][t]=cpu(term._processed_actions)
                data['qpos'][t]=cpu(env.sim.data.qpos[:32]);data['qvel'][t]=cpu(env.sim.data.qvel[:32]);completed=t+1
                count=int(env.sim.wp_data.nacon.numpy()[0]);peak=max(peak,count);assert 0<=count<slots
                video.sample()
                if completed%25==0:
                    assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
                    publish(control_transitions=completed*n,elapsed_seconds=time.monotonic()-start)
                    print('ANCHOR_VALUE_CHECK',completed,'/400;0actor/criticupdates;wall',round(time.monotonic()-start,1),flush=True)
            video.close();snapshot.close();snapshot=None
            np.savez_compressed(record/'value-data.npz',**data,initial_qpos=initial_qpos,initial_qvel=initial_qvel)
            with torch.no_grad():returns=discounted_returns(torch.from_numpy(data['rewards']).to(env.device),p['gamma']).cpu().numpy()
            target=returns/p['reward_scale'];valid=np.asarray(p['split']['validation_ids']);codes=np.asarray(p['split']['pose_codes'])
            overall=score(data['critic_predictions'][:,valid],target[:,valid])
            poses={name:score(data['critic_predictions'][:,valid[codes[valid]==i]],target[:,valid[codes[valid]==i]]) for i,name in enumerate(POSES)}
            gate=p['value_gate'];admitted=overall['explained_variance']>=gate['validation_explained_variance_min'] and overall['normalized_mse']<=gate['validation_normalized_mse_max'] and all(v['explained_variance']>=0 for v in poses.values())
            final={key:{k:v.detach().cpu().clone() for k,v in net.state_dict().items()} for key,net in [('actor',actor),('critic',critic)]}
            for role in initial:
                for k,v in initial[role].items():torch.testing.assert_close(v,final[role][k],atol=0,rtol=0)
            torch.save(final,record/'networks-final.pt');write(record/'contact-audits.json',env._getup_contact_audits)
            result=dict(status='passed-integrity',value_gate_passed=bool(admitted),validation=overall,poses=poses,
                reward_scale=p['reward_scale'],actor_and_critic_bitwise_unchanged=True,source_critic_sha256=p['critic_sha256'],
                actual_control_transitions=409600,actor_updates=0,critic_updates=0,heldout_used=False)
            write(record/'result.json',result)
            publish(status='budget-complete-review-required',phase='closed',value_gate_passed=bool(admitted),dataset_sha256=sha(record/'value-data.npz'),
                actor_and_critic_bitwise_unchanged=True,elapsed_seconds=time.monotonic()-start,finished_at=datetime.now(timezone.utc).isoformat())
            print('ANCHOR_VALUE_CHECK_COMPLETE',json.dumps(result),flush=True)
        except BaseException as error:
            publish(status='failed-preserved',phase='stopped',error=repr(error));traceback.print_exc()
            if data is not None and completed:np.savez_compressed(record/'partial-data.npz',**{k:v[:completed] for k,v in data.items()})
            raise
        finally:
            if video is not None:video.close()
            if snapshot is not None:snapshot.close()
            if env is not None:
                write(record/'runtime.json',dict(control_steps=int(env.common_step_counter),control_transitions=int(env.common_step_counter)*1024,
                    physics_substeps=int(env.common_step_counter)*5,extra_observation_or_render_steps=0,automatic_resets=0,manual_resets=1,
                    actor_updates=0,critic_updates=0,contact_peak=peak,contact_slots=slots if 'slots' in locals() else None))
                if state['status']=='failed-preserved':np.savez_compressed(record/'stop-state.npz',qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel))
                env.close()
            project=read(ROOT/'project.json');project['staged_training_request'].update(simulation_running=False,learning_running=False);write(ROOT/'project.json',project)
            sys.stdout,sys.stderr=stdout,stderr


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--verify',action='store_true');a=parser.parse_args()
    if a.freeze:freeze()
    elif a.verify:
        from verify_getup_anchor_value import verify
        verify()
    else:
        run();subprocess.run([sys.executable,'-B',str(Path(__file__)),'--verify'],cwd=WORK,check=True)
