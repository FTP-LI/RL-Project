"""Bounded PPO with common successful-state anchoring and contact-age input.

Both arms restore the same official actor and admitted76-input critic with its
Adam state. Only the candidate pays for eligible continuous head-leg contact.
Save reward-stage clock and post-reset critic input separately to preserve the
terminal contact when the next observation belongs to a new episode.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/validation'))
from s1_reference_inputs import read, sha
from record_io import write
from getup_persistent_contact import clock, networks, prepare as value_prepare, startup
from getup_value_bootstrap import td
from getup_feasibility_loss import interval_distance
from getup_anchor_ppo import AnchorPPO
from getup_anchor_runtime import RecordingSnapshot
from getup_contact_pilot import export, kinematic_stable
from getup_teacher_anchor_train import masks, save_checkpoint
from action_video import ActionVideo

CONTACT_FIELDS = ('contact', 'eligible', 'ticks', 'potential', 'cost',
                  'pre_age_input', 'post_age_input', 'manager_rate')


def prepare(p):
    cfg, display, meta = value_prepare(p)
    cfg.env.auto_reset = True
    display.update(recorder='Complete8s with periodic180s sampling at episode boundaries',
                   native_live_window=False)
    return cfg, display, meta


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


def initialize(p, out, record):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg, display, meta = prepare(p)
    write(record / 'display.json', display)
    env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
    env.seed(p['seed'])
    obs, _ = env.reset()
    actor, critic = networks(p, env.device)
    warm = torch.load(ROOT / p['critic_checkpoint'], map_location='cpu', weights_only=True)
    assert warm['reward_scale'] == p['reward_scale']
    critic.load_state_dict(warm['critic_state_dict'])
    with np.load(ROOT / p['anchor_data']) as data:
        anchor_inputs = data['observations'].copy()
        anchor_targets = data['teacher_targets'].copy()
    algorithm = AnchorPPO(actor, critic, warm['critic_optimizer_state_dict'], p['learner'],
                         anchor_inputs=anchor_inputs, anchor_targets=anchor_targets,
                         anchor_cells=read(ROOT / p['anchor_cells']),
                         anchor_weight=p['anchor_weight'], anchor_seed=p['anchor_seed'])
    startup(env, obs, actor, p, record, out / 'cpu-scene-fields.npz')
    reward = env.reward_manager
    index = reward._term_names.index('late_head_leg_contact')
    cfg_term = reward._term_cfgs[index]
    assert reward._scale_by_dt and abs(env.step_dt - .02) < 1e-8
    assert cfg_term.weight == -2 and cfg_term.params['enabled'] is p['persistent_penalty_enabled']
    assert cfg_term.params['threshold_seconds'] == p['contact_threshold_seconds']
    term = env.action_manager.get_term('joint_pos')
    assert term._processed_actions.shape == (1024, 14) and not term.standing_mask.any()
    memory = clock(env, p['contact_threshold_seconds'])
    initial = dict(actor=cpu(obs['actor']), critic=cpu(obs['critic']), qpos=cpu(env.sim.data.qpos),
                   qvel=cpu(env.sim.data.qvel), imu=cpu(env._imu_misalign_quat),
                   clock_ticks=cpu(memory.ticks), clock_last_step=cpu(memory.last_step),
                   episode_age=cpu(env.episode_length_buf),
                   cpu_rng=cpu(torch.get_rng_state()), cuda_rng=cpu(torch.cuda.get_rng_state(env.device)))
    initial.update({'expanded__' + name: cpu(getattr(env.sim.model, name))
                    for name in sorted(env.sim.expanded_fields)})
    np.savez_compressed(record / 'paired-start.npz', **initial)
    save_checkpoint(record / 'model-init.pt', algorithm, env, 0)
    write(record / 'reward-interface.json', dict(enabled=cfg_term.params['enabled'],
          threshold_seconds=cfg_term.params['threshold_seconds'], weight=cfg_term.weight,
          control_dt=env.step_dt, manager_rate_saved=True))
    return env, obs, algorithm, meta, index


def zero_step(p, out):
    from evaluate_standup_reference import Tee

    assert os.isatty(1), 'Use the original visible VS Code shared terminal'
    record = out / 'zero-step' / p['arm']
    record.mkdir(parents=True)
    write(record / 'command.json', dict(argv=sys.argv, foreground_tty=True, pid=os.getpid(),
          started_at=datetime.now(timezone.utc).isoformat()))
    with (record / 'execution.log').open('x', buffering=1) as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        env = None
        try:
            print('PERSISTENT_CONTACT_GPU_ZERO_STEP', p['arm'], flush=True)
            env, _, _, _, _ = initialize(p, out, record)
            assert env.common_step_counter == 0
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            if env is not None:
                env.close()
            sys.stdout, sys.stderr = stdout, stderr


def match_start(record, reference):
    from common_scene_contract import compare_arrays
    from verify_getup_teacher_anchor_train import exact

    with np.load(record / 'paired-start.npz') as actual, np.load(reference / 'paired-start.npz') as expected:
        compare_arrays({k: actual[k] for k in actual.files}, {k: expected[k] for k in expected.files})
    actual = torch.load(record / 'model-init.pt', map_location='cpu', weights_only=True)
    expected = torch.load(reference / 'model-init.pt', map_location='cpu', weights_only=True)
    exact(actual, expected)
    assert actual['control_steps'] == actual['actor_steps_this_run'] == actual['critic_steps_this_run'] == 0
    assert not actual['actor_optimizer_state_dict']['state']
    assert {int(v['step']) for v in actual['critic_optimizer_state_dict']['state'].values()} == {50}


def record_contact(env, memory, critic_before, critic_after, done, index, enabled):
    stage = env._persistent_contact
    applied_rate = env.reward_manager._step_reward[:, index]
    torch.testing.assert_close(applied_rate, -2 * stage['cost'], atol=0, rtol=0)
    expected_cost = stage['potential'] if enabled else torch.zeros_like(stage['cost'])
    torch.testing.assert_close(stage['cost'], expected_cost, atol=0, rtol=0)
    next_ticks = torch.where(done, torch.zeros_like(stage['ticks']), stage['ticks'])
    torch.testing.assert_close(memory.ticks, next_ticks, atol=0, rtol=0)
    torch.testing.assert_close(critic_after[:, 74], memory.observation()[:, 0], atol=0, rtol=0)
    # One transfer per control. Integer/bool clock fields are exactly
    # representable as float32 in the recorded range0..60.
    return cpu(torch.stack([*(stage[k] for k in CONTACT_FIELDS[:5]),
                            critic_before[:, 74], critic_after[:, 74], applied_rate], dim=-1))


def run(p,master_path,out):
    from evaluate_standup_reference import Tee
    assert os.isatty(1),'Use the original visible VS Code shared terminal'
    progress=out/p['arm']/'progress.json';assert not progress.exists(),'No repeated or extended attempt'
    record=ROOT/'records/experiments'/p['run_id'];record.mkdir();write(record/'protocol.json',p)
    for name in p['source_sha256']:
        target = record / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    state=dict(status='training',run_id=p['run_id'],path=str(record),pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),completed_updates=0,actor_updates=0,critic_updates=0,
        control_steps=0,planned_updates=p['training_updates'],num_envs=1024,protocol_sha256=sha(master_path),hardware_io=False)
    env = snapshot = video = algorithm = None
    trace, contact_trace = [], []
    completed = peak = 0
    def publish(**values):
        state.update(values)
        if env is not None:state['control_steps']=int(env.common_step_counter)
        if video is not None:state.update(clips=video.clips,images=[video.image] if video.image else [])
        write(record/'state.json',state);write(progress,state)
    publish();write(ROOT/'records/training-current.json',dict(path=str(record),record=str(record),run_id=p['run_id'],
        pid=os.getpid(),purpose='persistent-contact-paired-PPO'))
    write(record/'command.json',dict(argv=sys.argv,foreground_tty=os.isatty(1),pid=os.getpid(),started_at=state['started_at']))
    with (record/'execution.log').open('x',buffering=1) as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            print('VISIBLE_PERSISTENT_CONTACT', p['arm'], p['training_updates'],
                  'updates x24 x1024; penalty', p['persistent_penalty_enabled'],
                  '; both anchors .01; full8s; per-update drift stop', flush=True)
            env, obs, algorithm, meta, contact_index = initialize(p, out, record)
            actor, critic = algorithm.actor, algorithm.critic
            term = env.action_manager.get_term('joint_pos')
            memory = clock(env, p['contact_threshold_seconds'])
            match_start(record, out / 'zero-step' / p['arm'])
            if p['arm'] == 'candidate':
                control = ROOT / 'records/experiments' / p['matching_control']
                match_start(record, control)
            write(record / 'matched-start.json', dict(status='passed', physics_steps=0,
                  physical_observations_imu_rng_exact=True, actor_critic_optimizers_exact=True,
                  exact_against_gpu_zero_step=True, exact_against_control=p['arm']=='candidate'))
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
                    contact_trace.append(record_contact(env, memory, batch['critic'][-1],
                                                        obs['critic'], done, contact_index,
                                                        p['persistent_penalty_enabled']))
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
                    # A final episode boundary is for the terminal frame only.
                    # Do not start a new one-frame clip from the reset state.
                    if step in p['forced_video_boundaries'] and video.writer is None:
                        video.last_started = None
                    if step < 24 * p['training_updates'] or video.writer is not None:
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
                    current_stable_fraction=fraction,
                    eligible_persistent_fraction=float(env._persistent_contact['potential'].mean()),
                    elapsed_seconds=time.monotonic()-start)
                with (record/'learning-metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
                publish(completed_updates=update,actor_updates=actor_rollouts,critic_updates=update,metrics=metrics)
                print(f'PERSISTENT_PPO {p["arm"]} {update:02}/{p["training_updates"]} | steps={int(env.common_step_counter)} '
                    f'| stable={fraction:.1%} | persistent={metrics["eligible_persistent_fraction"]:.1%} '
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
            if contact_trace:
                contact_array = np.asarray(contact_trace)
                np.savez_compressed(record / 'contact-training-trace.npz',
                                    **{key: contact_array[..., i] for i, key in enumerate(CONTACT_FIELDS)})
            if env is not None:
                write(record/'runtime.json',dict(control_steps=int(env.common_step_counter),
                    control_transitions=int(env.common_step_counter)*1024,physics_substeps=int(env.common_step_counter)*5,
                    extra_observation_or_render_steps=0,contact_peak=peak,capacity=slots if 'slots' in locals() else None,
                    completed_updates=completed,automatic_resets=int(env.common_step_counter)//400,
                    recorded_replay_inputs=2360,anchor_weight=p['anchor_weight']))
                if hasattr(env,'_getup_contact_audits'):write(record/'contact-audits.json',env._getup_contact_audits)
                publish();env.close()
            sys.stdout,sys.stderr=stdout,stderr
