"""Bounded PPO comparison with a training-only successful-state mean anchor.

Both arms retain the admitted common interface and original rewards. The only
objective difference is replay coefficient0 vs0.01. After each complete PPO
update, compare all2360fixed successful teacher inputs; save and stop when RMS
exceeds0.05rad or any joint error exceeds0.25rad. Candidate cannot exceed the
actual control budget. An earlier stop closes that arm; never extend or retry.
Keep full8s clips when available, label interrupted clips and add no physics.
"""
from datetime import datetime,timezone
from pathlib import Path
import json,os,shutil,sys,time,traceback
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/validation'))
from s1_reference_inputs import read,sha
from record_io import write
from getup_interface_value import prepare as value_prepare
from getup_value_bootstrap import networks,td
from getup_feasibility_loss import interval_distance
from getup_anchor_ppo import AnchorPPO
from getup_anchor_runtime import startup,RecordingSnapshot
from getup_contact_pilot import export,kinematic_stable
from action_video import ActionVideo


def prepare(p):
    cfg,display,meta=value_prepare(p)
    cfg.env.auto_reset=True
    display.update(recorder='Complete8s learning episodes; periodic180s at boundaries; early stop clips marked partial',
                   native_live_window=False)
    return cfg,display,meta


def masks(ages):
    assert ages.dtype==torch.long
    active=ages>=100;late=ages>=250
    assert not (late&~active).any()
    return active,late


def save_checkpoint(path,algorithm,env,completed):
    # Include both independent optimizer samplers as well as physical RNG.
    # Closed pilot checkpoints are evidence, not permission to resume a run.
    state=dict(actor_state_dict=algorithm.actor.state_dict(),critic_state_dict=algorithm.critic.state_dict(),
        actor_optimizer_state_dict=algorithm.actor_optimizer.state_dict(),critic_optimizer_state_dict=algorithm.critic_optimizer.state_dict(),
        completed_updates=completed,actor_steps_this_run=algorithm.actor_steps,critic_steps_this_run=algorithm.critic_steps,
        control_steps=int(env.common_step_counter),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(env.device),
        minibatch_rng=algorithm.generator.get_state(),anchor_rng=algorithm.anchor_generator.get_state())
    temporary=path.with_suffix('.partial');torch.save(state,temporary);temporary.replace(path)


def run(p,master_path,out):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from common_scene_contract import model_arrays,compare_arrays
    from evaluate_standup_reference import Tee
    assert os.isatty(1),'Use the original visible VS Code shared terminal'
    progress=out/p['arm']/'progress.json';assert not progress.exists(),'No repeated or extended attempt'
    record=ROOT/'records/experiments'/p['run_id'];record.mkdir();write(record/'protocol.json',p)
    for name in p['source_sha256']:shutil.copy2(ROOT/name,record/Path(name).name)
    state=dict(status='training',run_id=p['run_id'],path=str(record),pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),completed_updates=0,actor_updates=0,critic_updates=0,
        control_steps=0,planned_updates=p['training_updates'],num_envs=1024,protocol_sha256=sha(master_path),hardware_io=False)
    env=snapshot=video=algorithm=None;trace=[];completed=0;peak=0
    def publish(**values):
        state.update(values)
        if env is not None:state['control_steps']=int(env.common_step_counter)
        if video is not None:state.update(clips=video.clips,images=[video.image] if video.image else [])
        write(record/'state.json',state);write(progress,state)
    publish();write(ROOT/'records/training-current.json',dict(path=str(record),record=str(record),run_id=p['run_id'],
        pid=os.getpid(),purpose='training-only-successful-state-mean-anchor-PPO'))
    write(record/'command.json',dict(argv=sys.argv,foreground_tty=os.isatty(1),pid=os.getpid(),started_at=state['started_at']))
    with (record/'execution.log').open('x',buffering=1) as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            print('VISIBLE_TEACHER_ANCHOR',p['arm'],p['training_updates'],'updates x24 x1024; weight',p['anchor_weight'],'; full8s; perupdate drift stop',flush=True)
            configure_torch_backends(allow_tf32=False);torch.set_num_threads(4)
            cfg,display,meta=prepare(p);write(record/'display.json',display)
            env=ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None);env.seed(p['seed']);obs,_=env.reset()
            actor,critic=networks(75,p,env.device)
            warm=torch.load(ROOT/p['critic_checkpoint'],map_location='cpu',weights_only=True)
            critic.load_state_dict(warm['critic_state_dict'])
            assert warm['reward_scale']==p['reward_scale']
            with np.load(ROOT/p['anchor_data']) as f:
                anchor_inputs=f['observations'].copy();anchor_targets=f['teacher_targets'].copy()
            cells=read(ROOT/p['anchor_cells'])
            algorithm=AnchorPPO(actor,critic,warm['critic_optimizer_state_dict'],p['learner'],
                anchor_inputs=anchor_inputs,anchor_targets=anchor_targets,anchor_cells=cells,
                anchor_weight=p['anchor_weight'],anchor_seed=p['anchor_seed'])
            startup(env,obs,actor,p,record,out/'cpu-scene-fields.npz')
            with np.load(out/'cpu-scene-fields.npz') as f:
                compare_arrays(model_arrays(env.sim.mj_model),{k:f[k] for k in f.files})
            cpu=lambda x:x.detach().cpu().numpy().copy()
            term=env.action_manager.get_term('joint_pos')
            assert term._processed_actions.shape==(1024,14) and not term.standing_mask.any()
            initial=dict(actor=cpu(obs['actor']),critic=cpu(obs['critic']),qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel),
                imu=cpu(env._imu_misalign_quat),cpu_rng=cpu(torch.get_rng_state()),cuda_rng=cpu(torch.cuda.get_rng_state(env.device)))
            initial.update({'expanded__'+k:cpu(getattr(env.sim.model,k)) for k in sorted(env.sim.expanded_fields)})
            np.savez_compressed(record/'paired-start.npz',**initial);save_checkpoint(record/'model-init.pt',algorithm,env,0)
            if p['arm']=='candidate':
                control=ROOT/'records/experiments'/p['matching_control']
                with np.load(control/'paired-start.npz') as f:compare_arrays(initial,{k:f[k] for k in f.files})
                parent=torch.load(control/'model-init.pt',map_location='cpu',weights_only=True)
                from verify_getup_late_feasibility import exact
                exact(parent['actor_state_dict'],actor.state_dict());exact(parent['critic_state_dict'],critic.state_dict())
                exact(parent['actor_optimizer_state_dict'],algorithm.actor_optimizer.state_dict())
                exact(parent['critic_optimizer_state_dict'],algorithm.critic_optimizer.state_dict())
                write(record/'matched-start.json',dict(status='passed',physics_steps=0,
                    physical_observations_imu_rng_exact=True,actor_critic_optimizers_exact=True))
            with np.load(ROOT/p['probe_dataset']) as f:
                probe=torch.from_numpy(f['actor_obs_sample'][[0,99,100,125,200,300,399]].reshape(-1,61).copy()).to(env.device)
            with torch.no_grad():
                initial_anchor=actor(td('actor',algorithm.anchor_inputs))
                torch.testing.assert_close(initial_anchor,algorithm.anchor_targets,atol=2e-5,rtol=2e-5)
            snapshot=RecordingSnapshot(env,record);video=ActionVideo(snapshot,'getup',record);video.sample()
            pre_reset=env.recorder_manager.record_pre_reset;terminal={}
            def capture(ids):
                terminal.update(ids=ids.clone(),qpos=env.sim.data.qpos[ids].clone(),qvel=env.sim.data.qvel[ids].clone(),
                    requested=term.requested_target[ids].clone(),applied=term._processed_actions[ids].clone())
                pre_reset(ids)
            env.recorder_manager.record_pre_reset=capture
            addresses=torch.tensor(p['qpos_addresses'],device=env.device);ranges=torch.tensor(p['joint_limits'],device=env.device)
            slots=int(env.sim.wp_data.contact.dist.shape[0]);bad_streak=0;actor_rollouts=0;start=time.monotonic();stop_reason='planned-budget-complete'
            for update in range(1,p['training_updates']+1):
                batch={k:[] for k in ('actor','critic','action','mean','log_prob','value','reward','done','active','late')}
                for _ in range(24):
                    age=env.episode_length_buf.clone();active,late=masks(age);terminal.clear()
                    assert not term.standing_mask.any()
                    observed32=cpu(obs['actor'][:32]);pre_qpos=cpu(env.sim.data.qpos[:32])
                    with torch.no_grad():
                        actions=actor(obs,stochastic_output=True);mean=actor.output_mean.clone();logprob=actor.get_output_log_prob(actions).clone()
                        torch.testing.assert_close(actor.output_std,torch.full_like(actor.output_std,.10),atol=1e-7,rtol=0)
                        values=critic(obs).squeeze(-1)
                        for key,v in (('actor',obs['actor']),('critic',obs['critic']),('action',actions),('mean',mean),
                            ('log_prob',logprob),('value',values),('active',active),('late',late)):batch[key].append(v.clone())
                        sampled=actions.clone();obs,reward,terminals,timeouts,_=env.step(actions)
                    torch.testing.assert_close(actions,sampled,atol=0,rtol=0)
                    done=terminals|timeouts;assert not terminals.any() and torch.equal(done,age==399)
                    history=torch.where(active[:,None],sampled,torch.zeros_like(sampled));history[done]=0
                    torch.testing.assert_close(obs['actor'][:,34:48],history,atol=0,rtol=0)
                    batch['reward'].append(reward.clone()/p['reward_scale']);batch['done'].append(done.clone())
                    qpos=env.sim.data.qpos.clone();qvel=env.sim.data.qvel.clone()
                    requested=term.requested_target.clone();applied=term._processed_actions.clone()
                    if done.any():
                        ids=terminal['ids'];qpos[ids]=terminal['qpos'];qvel[ids]=terminal['qvel']
                        requested[ids]=terminal['requested'];applied[ids]=terminal['applied']
                    assert all(torch.isfinite(x).all() for x in (qpos,qvel,reward,requested,applied,mean,sampled))
                    stable=kinematic_stable(qpos,qvel,addresses,ranges);fraction=float(stable.float().mean())
                    bad_streak=bad_streak+1 if (age+1>=350).all() and fraction<p['watch_stable_min'] else 0
                    count=int(env.sim.wp_data.nacon.numpy()[0]);peak=max(peak,count);assert 0<=count<slots
                    step=int(env.common_step_counter)
                    excess=interval_distance(mean+torch.as_tensor(p['model_home'],device=env.device),ranges).square().sum(-1)
                    trace.append(dict(step=step,age=cpu(age[:32]),actor_inputs=observed32,actions=cpu(sampled[:32]),
                        means=cpu(mean[:32]),requested_targets=cpu(requested[:32]),applied_targets=cpu(applied[:32]),
                        pre_qpos=pre_qpos,qpos=cpu(qpos[:32]),qvel=cpu(qvel[:32]),stable_fraction=fraction,
                        raw_mean_squared_excess=float(excess.mean()),late_policy_fraction=float(late.float().mean()),
                        head_leg_fraction=float(env._getup_contact['contact'].float().mean()),
                        raw_reward_mean=float(reward.mean()),done=bool(done.all())))
                    if step==800 and video.writer is None:video.last_started=None
                    video.sample()
                    assert bad_streak<25,'Late all-world kinematic decline for25controls; stop and review'
                    if step%25==0:assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
                tensors={k:torch.stack(v) for k,v in batch.items()}
                with torch.no_grad():last=critic(obs).squeeze(-1)
                metrics=algorithm.update(tensors,last);completed=update;actor_rollouts+=metrics['actor_gradient_steps']>0
                with torch.no_grad():drift=actor(td('actor',algorithm.anchor_inputs))-algorithm.anchor_targets
                metrics.update(completed_updates=update,control_steps=int(env.common_step_counter),
                    raw_action_drift_rms=float(drift.square().mean().sqrt()),raw_action_drift_max=float(drift.abs().max()),
                    eligible_late_fraction=float(tensors['late'].float().mean()),
                    current_stable_fraction=fraction,elapsed_seconds=time.monotonic()-start)
                with (record/'learning-metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
                publish(completed_updates=update,actor_updates=actor_rollouts,critic_updates=update,metrics=metrics)
                print(f'TEACHER_ANCHOR {p["arm"]} {update:02}/{p["training_updates"]} | steps={int(env.common_step_counter)} '
                    f'| stable={fraction:.1%} | aux={metrics["auxiliary_loss"]:.4f} '
                    f'| drift={metrics["raw_action_drift_rms"]:.4f}/{metrics["raw_action_drift_max"]:.4f}rad | KL={metrics["kl"]:.4f} | actor_steps={algorithm.actor_steps} '
                    f'| wall={metrics["elapsed_seconds"]:.1f}s',flush=True)
                if update==25:save_checkpoint(record/'model-25.pt',algorithm,env,completed)
                if (metrics['raw_action_drift_rms']>p['tripwire']['success_replay_rms_rad_max'] or
                    metrics['raw_action_drift_max']>p['tripwire']['success_replay_abs_rad_max']):
                    stop_reason='successful-state-drift-tripwire'
                    print('ANCHOR_TRIPWIRE',p['arm'],update,'save and stop; no extension',flush=True)
                    break
            assert int(env.common_step_counter)==completed*24
            save_checkpoint(record/'model-final.pt',algorithm,env,completed)
            video.close();export(actor,meta,probe,p,record)
            publish(status='budget-complete-review-required',completed_updates=completed,actor_updates=actor_rollouts,stop_reason=stop_reason,
                actor_optimizer_steps=algorithm.actor_steps,critic_optimizer_steps_this_run=algorithm.critic_steps,
                elapsed_seconds=time.monotonic()-start,finished_at=datetime.now(timezone.utc).isoformat())
        except BaseException as error:
            if algorithm is not None and env is not None:save_checkpoint(record/'stopped-partial.pt',algorithm,env,completed)
            publish(status='stopped-review-required',error=repr(error));traceback.print_exc();raise
        finally:
            if video is not None:video.close()
            if snapshot is not None:snapshot.close()
            if trace:np.savez_compressed(record/'training-trace.npz',**{k:np.asarray([row[k] for row in trace]) for k in trace[0]})
            if env is not None:
                write(record/'runtime.json',dict(control_steps=int(env.common_step_counter),
                    control_transitions=int(env.common_step_counter)*1024,physics_substeps=int(env.common_step_counter)*5,
                    extra_observation_or_render_steps=0,contact_peak=peak,capacity=slots if 'slots' in locals() else None,
                    completed_updates=completed,automatic_resets=int(env.common_step_counter)//400,
                    recorded_replay_inputs=2360,anchor_weight=p['anchor_weight']))
                if hasattr(env,'_getup_contact_audits'):write(record/'contact-audits.json',env._getup_contact_audits)
                publish();env.close()
            sys.stdout,sys.stderr=stdout,stderr
