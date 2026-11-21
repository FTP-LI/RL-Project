"""One finite-episode value-fit probe with the official getup actor frozen.

Why: fresh, badly scaled value estimates can destabilize the first PPO actor
step. Fit only a fresh critic on complete returns; never manufacture an
official critic or optimizer history. Validation worlds do not affect fitting,
normalization or reward scaling. This probe cannot improve the fixed actor.
"""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/validation'))
from s1_reference_inputs import WORK,read,sha
from record_io import write
from getup_contact_training import split_worlds,prepare,discounted_returns,train_reward_scale,POSES

PLAN=ROOT/'configs/experiments/getup_value_bootstrap.json'
OUT=ROOT/'records/reports/getup-value-bootstrap'


def make_plan():
    old=read(ROOT/'configs/experiments/common_plant/getup.json')
    report=read(ROOT/'records/reports/getup-contact-release-recovery/analysis.json')
    physical=read(ROOT/report['arms']['full']['record']/'evaluation.json')['simulation_home']
    return dict(status='frozen-before-data',stage='S3-getup-value-bootstrap',model=old['model'],
        model_sha256=old['model_sha256'],task=old['task'],seed=42,num_envs=1024,initial_std=.05,
        split=split_worlds(),split_seed=4301,minibatch_seed=442,simulation_home=physical,
        episode_seconds=8.,maximum_control_steps=400,maximum_control_transitions=409600,
        critic_updates=50,minibatch_size=8192,critic_learning_rate=.0003,gradient_norm_max=1.,
        gamma=.99,actor_updates=0,actor_frozen=True,critic_hidden_dims=[512,256,128],activation='elu',
        reward_scale='RMS of complete discounted returns on train worlds only; positive common divisor',
        returns='Finite8s Monte Carlo, zero terminal value, no bootstrap and no resets',
        critic_remaining_time=True,is_finite_horizon=True,head_leg_cost_enabled=False,
        value_gate=dict(validation_explained_variance_min=.5,validation_normalized_mse_max=.25,
            each_pose_validation_explained_variance_min=0.),
        evaluation_updates=[0,1,10,25,50],training_metric_samples=8192,
        automatic_retry=False,automatic_extension=False,heldout_used=False,hardware_io=False,
        scope='Development value-fit admission only; official actor plus explicit exploration; no gait-improvement claim')


def freeze():
    from dataclasses import asdict
    import yaml
    from mjlab.scene import Scene
    from common_scene_contract import model_arrays
    from check_getup_value_bootstrap import check
    assert not PLAN.exists() and not OUT.exists(),'Existing frozen attempt must be preserved'
    p=make_plan();proof=check(p);cfg,display,_=prepare(p)
    model=Scene(cfg.env.scene,device='cpu').compile()
    old=read(ROOT/'configs/experiments/common_plant/getup.json')
    sources=deepcopy(old['source_sha256'])
    for name in ('scripts/training/getup_contact_training.py','scripts/training/getup_model_coordinates.py',
        'scripts/training/getup_value_bootstrap.py','scripts/validation/check_getup_value_bootstrap.py',
        'scripts/training/metadata_actor_import.py','scripts/training/training_watch.py',
        'scripts/training/display_preferences.py','scripts/validation/verify_getup_value_bootstrap.py'):
        sources[name]=sha(ROOT/name)
    for name,digest in old['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in old['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    import inspect
    from rsl_rl.models import MLPModel
    from mjlab.sensor import ContactSensor
    from mjlab.envs import ManagerBasedRlEnv
    libraries={str(Path(inspect.getfile(cls))):sha(Path(inspect.getfile(cls))) for cls in (MLPModel,ContactSensor,ManagerBasedRlEnv)}
    p.update(source_sha256=sources,training_source_sha256=old['training_source_sha256'],library_source_sha256=libraries,
        frozen_at=datetime.now(timezone.utc).isoformat())
    assert not torch.cuda.is_initialized()
    OUT.mkdir();np.savez_compressed(OUT/'cpu-scene-fields.npz',**model_arrays(model));write(PLAN,p)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),fixture=proof,
        config=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader),display=display,
        cpu_scene_sha256=sha(OUT/'cpu-scene-fields.npz'),cuda_initialized=False))
    print('VALUE_PROBE_FROZEN',sha(PLAN),flush=True)


def checked():
    p=read(PLAN);assert read(OUT/'preflight.json')['protocol_sha256']==sha(PLAN)
    assert sha(ROOT/p['model'])==p['model_sha256']
    for base,key in ((ROOT,'source_sha256'),(WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
        for name,digest in p[key].items():assert sha(base/name)==digest,name
    return p


def td(key,x):
    from tensordict import TensorDict
    return TensorDict({key:x},batch_size=[len(x)])


def networks(width,p,device):
    from rsl_rl.models import MLPModel
    from metadata_actor_import import import_actor
    # Preserve both RNG streams used by the already initialized environment.
    with torch.random.fork_rng(devices=[0]),torch.device('cpu'):
        torch.manual_seed(p['seed'])
        actor=MLPModel(td('actor',torch.zeros(1,61)),{'actor':['actor']},'actor',14,
            hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
            distribution_cfg=dict(class_name='GaussianDistribution',init_std=p['initial_std'],std_type='scalar'))
        import_actor(actor,ROOT/p['model'],p['model_sha256'])
        critic=MLPModel(td('critic',torch.zeros(1,width)),{'critic':['critic']},'critic',1,
            hidden_dims=p['critic_hidden_dims'],activation='elu',obs_normalization=True)
    actor.eval();actor.requires_grad_(False)
    return actor.to(device),critic.to(device)


def startup(env,obs,actor,p,record):
    import onnxruntime as ort
    from common_scene_contract import model_arrays,compare_arrays,nominal_primary
    cpu=lambda x:x.detach().cpu().numpy().copy()
    assert env.common_step_counter==0 and (env.episode_length_buf==0).all()
    actual=model_arrays(env.sim.mj_model)
    with np.load(OUT/'cpu-scene-fields.npz') as f:compare_arrays(actual,{k:f[k] for k in f.files})
    np.savez_compressed(record/'scene-static-fields.npz',**actual)
    fields={k:cpu(getattr(env.sim.model,k)) for k in sorted(env.sim.expanded_fields)}
    primary=nominal_primary(fields,env.num_envs)
    with np.load(ROOT/'records/reports/common-plant-screen/startup/getup/nominal-primary-fields.npz') as f:
        compare_arrays(primary,{k:f[k] for k in f.files})
    np.savez_compressed(record/'expanded-fields-before-physics.npz',**fields)
    a=env.action_manager.get_term('joint_pos');physical=np.array(p['simulation_home'],np.float32)
    import onnx
    meta={v.key:v.value for v in onnx.load(ROOT/p['model']).metadata_props}
    home=np.asarray(meta['default_joint_pos'].split(','),np.float32)
    np.testing.assert_allclose(cpu(a._offset),np.broadcast_to(home,cpu(a._offset).shape),atol=1e-7,rtol=0)
    robot=env.scene['robot'];ids,names=robot.find_joints(meta['joint_names'].split(','),preserve_order=True)
    assert list(a._target_names)==names
    np.testing.assert_allclose(cpu(robot.data.default_joint_pos[:,ids]),np.broadcast_to(physical,(env.num_envs,14)),atol=1e-7,rtol=0)
    assert obs['actor'].shape==(env.num_envs,61) and obs['critic'].shape[0]==env.num_envs
    np.testing.assert_array_equal(cpu(obs['actor'][:,34:]),0.)
    np.testing.assert_array_equal(cpu(obs['critic'][:,-1]),1.)
    selected=cpu(obs['actor'][:32]);options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
    with torch.no_grad():predicted=cpu(actor(td('actor',obs['actor'][:32])))
    expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in selected])
    np.testing.assert_allclose(predicted,expected,atol=2e-5,rtol=2e-5)
    rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state(env.device).clone()
    saved={key:getattr(a,key).clone() for key in ('_raw_actions','_processed_actions')}
    histories={key:getattr(env.action_manager,key).clone() for key in ('_action','_prev_action','_prev_prev_action')}
    history=histories['_action'];counter=env.episode_length_buf.clone()
    qpos=env.sim.data.qpos.clone();qvel=env.sim.data.qvel.clone()
    probe=torch.full_like(history,.01);original=probe.clone()
    for count in (0,99,100):
        env.episode_length_buf.fill_(count);env.action_manager.process_action(probe)
        target=torch.as_tensor(physical if count<100 else home+.01,device=env.device).expand_as(a._processed_actions)
        torch.testing.assert_close(a._processed_actions,target,atol=1e-7,rtol=0)
        torch.testing.assert_close(probe,original,atol=0,rtol=0)
        torch.testing.assert_close(env.action_manager._action,torch.zeros_like(probe) if count<100 else probe,atol=0,rtol=0)
    for key,value in saved.items():getattr(a,key).copy_(value)
    for key,value in histories.items():getattr(env.action_manager,key).copy_(value)
    env.episode_length_buf.copy_(counter)
    torch.testing.assert_close(qpos,env.sim.data.qpos,atol=0,rtol=0);torch.testing.assert_close(qvel,env.sim.data.qvel,atol=0,rtol=0)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state(env.device))
    np.savez_compressed(record/'live-startup.npz',actor_obs=cpu(obs['actor']),critic_obs=cpu(obs['critic']),
        qpos=cpu(qpos),qvel=cpu(qvel),torch_actions=predicted,onnx_actions=expected)
    proof=dict(status='passed',physics_steps=0,actor_width=61,critic_width=obs['critic'].shape[1],
        maximum_import_error=float(abs(predicted-expected).max()),full_physical_fields_exact=True,nominal_primary_exact=True,
        freeze_and_release_checked=True,caller_actions_preserved=True,raw_history_checked=True,rng_and_physical_state_unchanged=True)
    write(record/'startup.json',proof);return proof


def fit(critic,data,p,record,publish):
    x=torch.from_numpy(data['critic_obs']).to('cuda:0');raw=torch.from_numpy(data['rewards']).to('cuda:0')
    returns=discounted_returns(raw,p['gamma']);scale=train_reward_scale(returns,p['split']['train_ids'])
    target=(returns/scale).float();train=torch.tensor(p['split']['train_ids'],device=x.device)
    valid=torch.tensor(p['split']['validation_ids'],device=x.device)
    xtrain=x[:,train].reshape(-1,x.shape[-1]);ytrain=target[:,train].reshape(-1)
    xvalid=x[:,valid].reshape(-1,x.shape[-1]);yvalid=target[:,valid].reshape(-1)
    critic.train()
    with torch.no_grad():critic.update_normalization(td('critic',xtrain))
    critic.obs_normalizer.until=0;critic.eval()
    assert int(critic.obs_normalizer.count)==400*960
    generator=torch.Generator(device='cpu').manual_seed(p['minibatch_seed'])
    indices=torch.randint(len(xtrain),(p['critic_updates'],p['minibatch_size']),generator=generator)
    probe=torch.randperm(len(xtrain),generator=generator)[:p['training_metric_samples']].to(x.device)
    optimizer=torch.optim.Adam(critic.parameters(),lr=p['critic_learning_rate'])
    torch.save(dict(critic_state_dict=critic.state_dict(),reward_scale=scale),record/'critic-fit-start.pt')
    rows=[]
    def score(xx,yy):
        with torch.no_grad():pred=torch.cat([critic(td('critic',v)).squeeze(-1) for v in xx.split(4096)])
        error=pred-yy;variance=yy.double().var(unbiased=False)
        assert variance>1e-10
        return dict(normalized_mse=float(error.double().square().mean()),
            explained_variance=float(1.-error.double().var(unbiased=False)/variance),bias=float(error.double().mean()))
    def evaluate(update,loss=None,grad=None):
        row=dict(update=update,train=score(xtrain[probe],ytrain[probe]),validation=score(xvalid,yvalid),
            poses={},loss=loss,gradient_norm=grad)
        codes=np.asarray(p['split']['pose_codes'])[p['split']['validation_ids']]
        for i,name in enumerate(POSES):
            mask=torch.tensor(codes==i,device=x.device)
            row['poses'][name]=score(x[:,valid][:,mask].reshape(-1,x.shape[-1]),target[:,valid][:,mask].reshape(-1))
        rows.append(row);write(record/'value-metrics.json',rows)
        publish(phase='critic-fit',completed_updates=update,value_metrics=row,reward_scale=scale)
        print('VALUE_FIT',json.dumps(row),flush=True)
    evaluate(0)
    for k,idx in enumerate(indices,1):
        idx=idx.to(x.device);pred=critic(td('critic',xtrain[idx])).squeeze(-1)
        loss=(pred-ytrain[idx]).square().mean();assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(critic.parameters(),p['gradient_norm_max'])
        assert torch.isfinite(grad);optimizer.step()
        assert all(torch.isfinite(v).all() for v in critic.state_dict().values())
        if k in p['evaluation_updates']:evaluate(k,float(loss),float(grad))
    np.savez_compressed(record/'fit-sampling.npz',minibatch_indices=indices.numpy(),metric_probe_indices=probe.cpu().numpy())
    torch.save(dict(critic_state_dict=critic.state_dict(),critic_optimizer_state_dict=optimizer.state_dict(),
        reward_scale=scale,gamma=p['gamma'],critic_updates=50,actor_updates=0,split=p['split']),record/'critic-final.pt')
    gate=p['value_gate'];last=rows[-1]
    passed=last['validation']['explained_variance']>=gate['validation_explained_variance_min']
    passed&=last['validation']['normalized_mse']<=gate['validation_normalized_mse_max']
    passed&=all(r['explained_variance']>=gate['each_pose_validation_explained_variance_min'] for r in last['poses'].values())
    return dict(value_gate_passed=bool(passed),reward_scale=scale,final=last,critic_updates=50,actor_updates=0)


def run():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from training_watch import Snapshot,SparseVideo
    from evaluate_standup_reference import Tee
    assert os.isatty(1),'Use the one visible VS Code shared task terminal'
    p=checked();assert not (OUT/'progress.json').exists(),'No automatic retry'
    run_id='duck_getup_value-bootstrap_reference_s42_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    record=ROOT/'records/experiments'/run_id;record.mkdir();write(record/'protocol.json',p)
    state=dict(status='training',phase='starting',run_id=run_id,path=str(record),pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PLAN),completed_updates=0,
        actor_updates=0,critic_updates=0,control_steps=0,heldout_used=False,hardware_io=False)
    def publish(**values):
        state.update(values)
        if video is not None:state.update(clips=video.clips,image=video.image)
        write(record/'state.json',state);write(OUT/'progress.json',state)
    env=snapshot=video=None
    publish();write(ROOT/'records/training-current.json',dict(path=str(record),run_id=run_id,pid=os.getpid()))
    write(record/'command.json',dict(argv=sys.argv,foreground_tty=os.isatty(1),pid=os.getpid(),
        started_at=state['started_at'],scope=p['scope']))
    with (record/'execution.log').open('x',buffering=1) as log:
        oldout,olderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(oldout,log),Tee(olderr,log)
        try:
            print('VISIBLE_FIXED_ACTOR_VALUE_PROBE',run_id,'max_controls=400 critic_steps=50 actor_steps=0 tty=',os.isatty(1),flush=True)
            configure_torch_backends(allow_tf32=False);torch.set_num_threads(4)
            cfg,display,metadata=prepare(p);write(record/'display.json',display)
            env=ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None)
            env.seed(p['seed']);obs,_=env.reset()
            actor,critic=networks(obs['critic'].shape[1],p,env.device);startup(env,obs,actor,p,record)
            initial={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
            torch.save(dict(actor_state_dict=initial,critic_state_dict=critic.state_dict(),actor_updates=0,critic_updates=0),record/'model-init.pt')
            snapshot=Snapshot(env,record);video=SparseVideo(snapshot,read(ROOT/'configs/experiments/training_watch.json'),record)
            video.sample();publish(phase='collecting-value-data')
            n=p['num_envs'];steps=p['maximum_control_steps'];width=obs['critic'].shape[1]
            data={k:np.empty(shape,dtype=dtype) for k,shape,dtype in (
                ('critic_obs',(steps,n,width),np.float32),('rewards',(steps,n),np.float32),
                ('actions',(steps,n,14),np.float32),('raw_history',(steps,n,14),np.float32),
                ('actor_obs_sample',(steps,32,61),np.float32),('terminal',(steps,n),bool),
                ('head_leg',(steps,n),bool),('eligible',(steps,n),bool),('cost',(steps,n),np.float32),
                ('qpos',(steps,32,env.sim.data.qpos.shape[1]),np.float32),('qvel',(steps,32,env.sim.data.qvel.shape[1]),np.float32))}
            start=time.monotonic();peak=0;slots=int(env.sim.wp_data.contact.dist.shape[0])
            cpu=lambda v:v.detach().cpu().numpy()
            for t in range(steps):
                assert env.common_step_counter==t and (env.episode_length_buf==t).all()
                data['critic_obs'][t]=cpu(obs['critic']);data['raw_history'][t]=cpu(obs['actor'][:,34:48])
                data['actor_obs_sample'][t]=cpu(obs['actor'][:32])
                assert np.isfinite(data['critic_obs'][t]).all()
                with torch.no_grad():actions=actor(obs,stochastic_output=True)
                sampled=actions.clone();data['actions'][t]=cpu(actions)
                assert torch.isfinite(actions).all()
                obs,reward,terminated,truncated,_=env.step(actions)
                torch.testing.assert_close(actions,sampled,atol=0,rtol=0)
                expected=torch.zeros_like(sampled) if t<100 else sampled
                torch.testing.assert_close(obs['actor'][:,34:48],expected,atol=0,rtol=0)
                assert torch.isfinite(reward).all() and torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all()
                assert not terminated.any() and (not truncated.any() if t<399 else truncated.all())
                data['rewards'][t]=cpu(reward);data['terminal'][t]=cpu(terminated|truncated)
                for key,term in (('head_leg','contact'),('eligible','eligible'),('cost','cost')):data[key][t]=cpu(env._getup_contact[term])
                data['qpos'][t]=cpu(env.sim.data.qpos[:32]);data['qvel'][t]=cpu(env.sim.data.qvel[:32])
                count=int(env.sim.wp_data.nacon.numpy()[0]);peak=max(peak,count);assert 0<=count<slots
                if t+1==250 and video.writer is None:video.last_started=None
                video.sample()
                if (t+1)%25==0:
                    assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
                    publish(control_steps=t+1,control_transitions=(t+1)*n,elapsed_seconds=time.monotonic()-start)
                    print('VALUE_DATA',t+1,'/',steps,'head_leg_worlds=',int(data['head_leg'][t].sum()),'elapsed_s=',round(time.monotonic()-start,2),flush=True)
            video.finish();write(record/'contact-audits.json',env._getup_contact_audits)
            assert any(a['contact_worlds']>0 for a in env._getup_contact_audits)
            publish(phase='archive-samples',control_steps=400,control_transitions=409600,contact_peak=peak,contact_slots=slots)
            temp=record/'value-data.npz.partial'
            with temp.open('xb') as file:np.savez_compressed(file,**data)
            temp.replace(record/'value-data.npz')
            result=fit(critic,data,p,record,publish)
            assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in actor.state_dict().items())
            torch.save(dict(actor_state_dict=actor.state_dict(),actor_updates=0),record/'actor-final.pt')
            write(record/'result.json',dict(status='passed',actor_bitwise_unchanged=True,**result))
            publish(status='budget-complete-review-required',phase='closed',critic_updates=50,completed_updates=50,
                actor_updates=0,actor_bitwise_unchanged=True,value_gate_passed=result['value_gate_passed'],
                finished_at=datetime.now(timezone.utc).isoformat(),dataset_sha256=sha(record/'value-data.npz'))
            print('VALUE_PROBE_COMPLETE',json.dumps(result),flush=True)
        except BaseException as error:
            publish(status='failed-preserved',phase='stopped',error=repr(error));traceback.print_exc();raise
        finally:
            if video is not None:video.finish()
            if snapshot is not None:snapshot.close()
            if env is not None:
                write(record/'runtime.json',dict(control_steps=int(env.common_step_counter),
                    physics_substeps=int(env.common_step_counter)*5,extra_observation_or_render_steps=0,
                    manual_resets=1,automatic_resets=0,actor_updates=0))
                if state['status']=='failed-preserved':
                    np.savez_compressed(record/'stop-state.npz',qpos=env.sim.data.qpos.detach().cpu().numpy(),
                        qvel=env.sim.data.qvel.detach().cpu().numpy())
                env.close()
            sys.stdout,sys.stderr=oldout,olderr


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    freeze() if args.freeze else run()
