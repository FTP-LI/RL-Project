"""Shared local PPO learner for the XGO-Duck roll experiment.

Capacity runs use the official uniform distribution. The later paired
experiment can use the same learner with only the startup friction draw changed.
The official network, critic initialization and Adam groups are loaded from
one verified checkpoint before any environment step.
"""

from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import file_digest
import json
import math
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.scripts.train import TrainConfig
from mjlab.sim import Simulation
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from tensordict import TensorDict

from action_video import ActionVideo
from record_io import write
from roll_friction_exposure import apply as expose
from roll_top_contact import adapt as adopt_home
from roll_top_contact_pilot import roll_score
from run_local_getup_ppo import MemoryWatch
from selective_friction_guard import resolve_foot_geometry_ids
from training_watch import Snapshot

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with path.open('rb') as file:
        return file_digest(file, 'sha256').hexdigest()


def demand(condition, message):
    if not condition:
        raise RuntimeError(message)


def prepare(plan, arm, metadata):
    recipe = plan['recipe']
    cfg = TrainConfig.from_task(recipe['task'])
    # This helper corrects coordinates to the model HOME. Preserve the original
    # head-pivot reward object so even its callable stays the same in both arms.
    pivot = cfg.env.rewards['roulade_head_pivot']
    env_cfg = adopt_home(cfg.env, metadata, False)
    env_cfg.rewards['roulade_head_pivot'] = pivot
    expose(env_cfg, arm['friction_role'])
    cfg = replace(cfg, env=env_cfg)
    cfg.env.scene.num_envs = cfg.env.scene.terrain.num_envs = arm['environments']
    cfg.env.seed = cfg.agent.seed = recipe['seed']
    cfg.agent.run_name = arm['run_id']
    cfg.agent.max_iterations = arm['updates']
    cfg.agent.save_interval = 10**9
    cfg.agent.logger = 'tensorboard'
    cfg.agent.upload_model = cfg.agent.resume = False
    cfg.agent.actor.distribution_cfg['init_std'] = recipe['initial_std']
    cfg.agent.algorithm.learning_rate = recipe['learning_rate']
    cfg.agent.algorithm.schedule = 'fixed'
    cfg.agent.algorithm.entropy_coef = 0.0
    demand(cfg.agent.algorithm.num_mini_batches == 4 and cfg.agent.algorithm.num_learning_epochs == 5
           and cfg.agent.clip_actions is None and cfg.agent.algorithm.symmetry_cfg is None
           and cfg.agent.num_steps_per_env == recipe['rollout_controls']
           and cfg.env.auto_reset and cfg.env.episode_length_s == 7
           and cfg.env.decimation == 5 and cfg.env.sim.mujoco.timestep == .004, 'Official learner settings changed')
    return cfg


def resources(watch):
    rows = list(watch.rows)
    demand(rows and all(row.get('used_mib') is not None for row in rows), 'Device telemetry unavailable')
    return dict(rows=rows, peak_used_mib=max(x['used_mib'] for x in rows),
                median_utilization=statistics.median(x['gpu_percent'] for x in rows))


def check_exact(actual, expected, label):
    demand(set(actual)==set(expected), label+' tensor keys')
    for key in actual:
        torch.testing.assert_close(actual[key].detach().cpu(), expected[key], atol=0, rtol=0, msg=label+'/'+key)


def initial_setup(raw, wrapped, runner, parent, session, metadata, arm, record):
    from verify_roll_exposure_parent import finite
    demand(raw.common_step_counter == raw._sim_step_counter == 0, 'Initialization advanced physics')
    demand(parent['iter']==-1 and parent['infos']['env_state']['common_step_counter']==0
           and not parent['optimizer_state_dict']['state'] and finite(parent), 'Parent is not the shared initialization')
    runner.alg.load(parent, load_cfg=None, strict=True)
    check_exact(runner.alg.actor.state_dict(), parent['actor_state_dict'], 'actor')
    check_exact(runner.alg.critic.state_dict(), parent['critic_state_dict'], 'critic')
    opt = runner.alg.optimizer.state_dict()
    demand(opt==parent['optimizer_state_dict'] and not opt['state'], 'Adam initialization differs from parent')
    runner.alg.actor.obs_normalizer.until = 0
    runner.alg.actor.requires_grad_(False)
    term = raw.action_manager.get_term('joint_pos')
    names = metadata['joint_names'].split(',')
    demand(list(term.target_names)==names and float(metadata['action_scale'])==1., 'Joint names or action scale differs')
    home = np.asarray(metadata['default_joint_pos'].split(','),np.float32)
    offset = term._offset.detach().cpu().numpy()
    np.testing.assert_allclose(offset, np.broadcast_to(home,offset.shape), atol=1e-7, rtol=0)
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        obs = wrapped.get_observations()
        demand(obs['actor'].shape==(arm['environments'],61) and obs['critic'].shape==(arm['environments'],74),
               'Actor or critic input width changed')
        x = obs['actor'][:32]
        actual = runner.alg.actor(TensorDict({'actor':x},batch_size=[len(x)])).detach().cpu().numpy()
        recorded = x.cpu().numpy()
    np.testing.assert_allclose(recorded[:,34:48],0.,atol=1e-7,rtol=0)
    expected = np.concatenate([session.run(None,{'obs':x[None]})[0] for x in recorded])
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    feet = resolve_foot_geometry_ids(raw.sim.mj_model)
    sliding = raw.sim.model.geom_friction[:,feet,0].detach().cpu().numpy()
    demand(sliding.shape==(arm['environments'],2) and np.isfinite(sliding).all()
           and np.all((sliding >= .7) & (sliding <= 1.3)), 'Actual sliding coefficients are outside the plan')
    np.testing.assert_array_equal(sliding[:,0],sliding[:,1])
    demand(raw.common_step_counter == raw._sim_step_counter == 0, 'Zero-step evidence changed the counters')
    proof = dict(status='passed-learner-reset-only', controls=0, physics_substeps=0, learning_updates=0,
                 matched_actor_critic_and_adam=True, matching_actor_inputs=len(x), max_actor_error=float(abs(actual-expected).max()),
                 sliding_mu_below_0p85=int((sliding[:,0]<.85).sum()), min_mu=float(sliding.min()), max_mu=float(sliding.max()))
    path = record/'startup-inputs.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, actor=recorded, policy=actual, sliding_mu=sliding[:,0],
                            cpu_rng=torch.get_rng_state().numpy(), gpu_rng=torch.cuda.get_rng_state().numpy(),
                            qpos=raw.sim.data.qpos.detach().cpu().numpy(), qvel=raw.sim.data.qvel.detach().cpu().numpy())
    proof['startup_inputs_sha256']=sha(path)
    write(record/'zero-step.json',proof)
    return proof


def collect(plan, arm, record, observer):
    recipe = plan['recipe']
    n, updates = arm['environments'], arm['updates']
    critic_only = recipe['critic_only_updates']
    rollout = recipe['rollout_controls']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/recipe['official_onnx']),sess_options=options,providers=['CPUExecutionProvider'])
    metadata = session.get_modelmeta().custom_metadata_map
    parent = torch.load(ROOT/recipe['parent_checkpoint'],map_location='cpu',weights_only=True)
    cfg = prepare(plan,arm,metadata)
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    watch = MemoryWatch(recipe['hard_gpu_fraction'])
    rows = []
    stage_clock, cpu_clock = None,None
    component = dict(checkpoint_write_seconds=0.,publication_seconds=0.,snapshot_seconds=0.,ppo_update_seconds=0.)
    progress = dict(run_id=arm['run_id'], environment_count=n, planned_updates=updates,
                    scope=plan['scope'], friction_role=arm['friction_role'])
    raw = snapshot = video = runner = None
    checkpoint = record/'profile-final.pt'
    measured_from = time.perf_counter()

    def guard():
        watch.check()
        observer.require_active()

    def publish(status, marker=None):
        start = time.perf_counter()
        current = raw.common_step_counter
        clips = video.clips if video is not None else []
        complete = [clip for clip in clips if clip['full_action']]
        observation = dict(run_id=arm['run_id'],status=status,updated_at=now(),
            controls=current,transitions=current*n,environments=n,planned_updates=updates,
            completed_updates=len(rows),rows=list(rows),frame=marker['file'] if marker else None,
            latest_clip=complete[-1]['file'] if complete else None,
            checkpoint=checkpoint.name if checkpoint.exists() else None)
        written = dict(progress, status=status, controls=current, physical_substeps=raw._sim_step_counter,
            completed_updates=len(rows), clips=clips, images=[marker] if marker else [])
        write(record/'state.json',written)
        write(record/'observation.json',observation)
        observer.publish(record,observation)
        component['publication_seconds']+=time.perf_counter()-start

    try:
        with watch.active():
            guard()
            def forbidden(*_args, **_kwargs):
                raise RuntimeError('An environment step attempted before learner zero-step approval')
            with patch.object(ManagerBasedRlEnv,'step',forbidden), patch.object(Simulation,'step',forbidden):
                raw = ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None)
                env = RslRlVecEnvWrapper(raw,clip_actions=cfg.agent.clip_actions)
                env.seed(recipe['seed'])
                env.reset()
                dump_yaml(record/'environment.yaml',asdict(cfg.env))
                dump_yaml(record/'learner.yaml',asdict(cfg.agent))
                runner = MjlabOnPolicyRunner(env,asdict(cfg.agent),str(record/'tensorboard'),'cuda:0')
                runner.logger.git_status_repos = []
                proof = initial_setup(raw,env,runner,parent,session,metadata,arm,record)
                demand(runner.alg.learning_rate==recipe['learning_rate'], 'Learning rate differs from frozen recipe')
                watch.sample()
                if plan['scope']=='formal-paired-training':
                    from verify_roll_exposure_parent import finite
                    demand(finite(runner.alg.save()), 'Invalid initial learner')

            publish('gpu-zero-step')
            guard()
            snapshot = Snapshot(raw,record)
            video = ActionVideo(snapshot,'roll',record)
            first_frame = None
            update_raw, step_raw, log_raw, save_raw = runner.alg.update, env.step, runner.logger.log, runner.save
            anchor_score = None
            reward_sum = torch.zeros((),device='cuda:0')
            batch_clock = None
            nsteps = 0
            latest_marker = None
            fixed_probe = parent['actor_state_dict']

            def step(actions):
                nonlocal reward_sum,nsteps
                guard()
                demand(raw.common_step_counter < rollout*updates, 'Frozen rollout budget reached')
                # Existing official learn loop checks state/reward; guard inputs too.
                if int(raw.common_step_counter)%rollout==0:
                    demand(bool(torch.isfinite(actions).all()),'Nonfinite sampled action')
                sample = step_raw(actions)
                reward_sum += sample[1].mean()
                nsteps += 1
                video.sample()
                return sample

            def update():
                guard()
                demand(raw.common_step_counter == (len(rows)+1)*rollout and nsteps==rollout,'Wrong rollout phase')
                torch.cuda.synchronize()
                start=time.perf_counter()
                loss=update_raw()
                torch.cuda.synchronize()
                component['ppo_update_seconds']+=time.perf_counter()-start
                guard()
                return loss

            def save(_path, infos=None):
                if not rows or (len(rows)!=updates and len(rows) not in arm.get('extra_checkpoints',[])):
                    return
                path = checkpoint if len(rows)==updates else record/f'checkpoint-update-{len(rows)}.pt'
                # The installed runner also requests update-zero and final
                # snapshots. Write only the declared complete-update files.
                if path.exists():
                    return
                start=time.perf_counter()
                tmp=path.with_suffix('.partial')
                save_raw(str(tmp), infos={**(infos or {}),'roll_capacity':dict(
                    plan_scope=plan['scope'], run_id=arm['run_id'], completed_updates=len(rows), parent=recipe['parent_sha256'])})
                with tmp.open('rb') as stream:
                    os.fsync(stream.fileno())
                tmp.replace(path)
                component['checkpoint_write_seconds']+=time.perf_counter()-start

            def log(*args, **kw):
                nonlocal nsteps,reward_sum,batch_clock,latest_marker,anchor_score
                guard()
                iteration = int(kw['it'])+1
                demand(iteration==len(rows)+1, 'Unexpected PPO iteration')
                losses={key:float(value) for key,value in kw['loss_dict'].items()}
                demand(losses and all(math.isfinite(x) for x in losses.values()),'Nonfinite PPO loss')
                elapsed=time.perf_counter()-batch_clock
                reward=float(reward_sum.item()/nsteps)
                demand(math.isfinite(reward), 'Nonfinite rollout reward')
                if iteration in recipe['late_monitor_updates']:
                    quality=roll_score(raw,recipe['nominal_height_m'])
                    write(record/f'late-roll-update-{iteration}.json', quality)
                    if iteration==recipe['late_monitor_updates'][0]:
                        anchor_score=quality['qualified_stable_fraction']
                        # Capacity trials always use the same fixed episode clock.
                        demand(anchor_score is not None and anchor_score >= recipe['minimum_late_stability'],
                               'Teacher did not retain sufficient late roll stability')
                    elif quality['qualified_stable_fraction'] is not None:
                        demand(quality['qualified_stable_fraction'] >= max(.1,anchor_score-recipe['maximum_late_drop']),
                               'Late roll holding fell beyond the guard')
                if iteration==critic_only:
                    check_exact(runner.alg.actor.state_dict(),fixed_probe,'critic-only actor')
                    write(record/'critic-only.json',dict(status='actor-byte-unchanged',updates=iteration,
                          controls=raw.common_step_counter, critic_normalizer_count=int(runner.alg.critic.obs_normalizer.count)))
                    runner.alg.actor.requires_grad_(True)
                row=dict(update=iteration, control=raw.common_step_counter, phase='critic-only' if iteration<=critic_only else 'full-ppo',
                    timestamp=now(), environment_reward_mean=reward,
                    value_loss=losses.get('value_function',losses.get('value')),
                    policy_loss=losses.get('surrogate'), loss=losses, rollout_wall_seconds=float(kw['collect_time']),
                    returns_and_update_wall_seconds=float(kw['learn_time']), measured_batch_wall_seconds=elapsed,
                    net_transitions_per_second=rollout*n/elapsed)
                with (record/'learning-metrics.jsonl').open('a',encoding='utf-8',newline='\n') as file:
                    file.write(json.dumps(row,allow_nan=False)+'\n')
                rows.append(row)
                if iteration%4==0 or iteration==updates:
                    tick=time.perf_counter()
                    latest_marker=snapshot.capture()
                    component['snapshot_seconds']+=time.perf_counter()-tick
                publish('training',latest_marker)
                log_raw(*args,**kw)
                if iteration in arm.get('extra_checkpoints',[]):
                    save(None)
                nsteps=0
                reward_sum.zero_()
                batch_clock=time.perf_counter()

            runner.alg.update, env.step, runner.logger.log, runner.save=update,step,log,save
            setup_elapsed = time.perf_counter()-measured_from
            stage_clock,cpu_clock=time.perf_counter(),time.process_time()
            video.sample()
            if video.image:
                first_frame=video.image
            latest_marker=first_frame
            publish('training',latest_marker)
            batch_clock=time.perf_counter()
            runner.learn(num_learning_iterations=updates,init_at_random_ep_len=False)
            torch.cuda.synchronize()
            video.close()
            publish('training',latest_marker)
            watch.sample()
            elapsed=time.perf_counter()-stage_clock
            cpu=time.process_time()-cpu_clock
            demand(checkpoint.is_file() and len(rows)==updates and raw.common_step_counter==rollout*updates
                   and raw._sim_step_counter==5*rollout*updates, 'Physical or optimizer budget differs')
            demand(any(clip['full_action'] and clip['frames']==176 for clip in video.clips), 'Missing complete roll episode')
            info=dict(progress,status='collected-review-required',ended_at=now(),init_seconds=setup_elapsed,
                controls=raw.common_step_counter,physics_substeps=raw._sim_step_counter,transitions=n*rollout*updates,
                completed_updates=updates,critic_only_updates=critic_only,actor_updates=updates-critic_only,
                measured_wall_seconds=elapsed,process_cpu_seconds=cpu,
                measured_transitions_per_second=n*rollout*updates/elapsed,
                component_wall_seconds={**component,'render_and_video_seconds':video.elapsed_seconds},
                device=resources(watch),checkpoint=checkpoint.name,checkpoint_sha256=sha(checkpoint),
                metrics_sha256=sha(record/'learning-metrics.jsonl'),zero_step_sha256=sha(record/'zero-step.json'),
                parent_sha256=recipe['parent_sha256'],clips=video.clips,hardware_io=False)
            write(record/'online-result.json',info)
            print('ROLL_PPO_PROFILE',arm['run_id'],info['measured_transitions_per_second'],flush=True)
            return info
    except BaseException as exc:
        write(record/'worker-stop.json',dict(progress,status='stopped-review-required',reason=f'{type(exc).__name__}: {exc}',
              actual_controls=None if raw is None else raw.common_step_counter,
              actual_physics_substeps=None if raw is None else raw._sim_step_counter,completed_updates=len(rows),
              device=resources(watch) if watch.rows else None,ended_at=now()))
        raise
    finally:
        if video is not None:
            video.close()
        if snapshot is not None:
            snapshot.close()
        if raw is not None:
            raw.close()
