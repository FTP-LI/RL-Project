"""Run measured forward PPO batches with the common parent and local observer."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import time
from unittest.mock import patch

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.sim import Simulation
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from action_video import ActionVideo
from record_io import write
from repair_forward_yaw_zero_single_reset import SingleResetForwardRollout
from roll_friction_ppo import check_exact, demand, resources
from run_forward_yaw_reward_zero import learner_config
from run_local_getup_ppo import MemoryWatch
from selective_friction_guard import RESET_FIELDS, compare_reset_fields, resolve_foot_geometry_ids
from training_watch import Snapshot
from verify_roll_exposure_parent import finite


def save_checkpoint(path, runner, env, arm, parent_sha, update):
    checkpoint = runner.alg.save()
    checkpoint.update(iter=update - 1, infos=dict(
        env_state={'common_step_counter': env.common_step_counter}, parent_sha256=parent_sha,
        run_id=arm['run_id'], actor_normalizer_frozen=True, completed_updates=update,
        resume_scope='Learner and optimizer from a fresh simulator reset'))
    demand(finite(checkpoint), 'Invalid learner or optimizer state')
    temporary = path.with_suffix('.partial')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def collect(root, plan, arm, record, observer):
    from s1_reference_inputs import read, sha

    draft, route = (read(root / plan[key]) for key in ('draft', 'route'))
    budget = draft['pair_budget']
    worlds, updates = arm['worlds'], arm['updates']
    rollout, warmup = budget['controls_per_update'], arm['critic_only_updates']
    role = arm['role']
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg, agent, weight = learner_config(route, draft, worlds, role, updates)
    agent.run_name = arm['run_id']
    dump_yaml(record / 'environment.yaml', asdict(cfg))
    dump_yaml(record / 'learner.yaml', asdict(agent))
    parent_path = root / plan['parent']
    demand(sha(parent_path) == plan['parent_sha256'], 'The shared parent changed')
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    watch = MemoryWatch(draft['capacity']['gpu_memory_fraction_max'])
    rows, raw, video, snapshot = [], None, None, None
    latest_frame = None
    metrics = record / 'learning-metrics.jsonl'
    checkpoint = record / 'model-final.pt'
    components = dict(optimizer_seconds=0., snapshot_seconds=0., publication_seconds=0., checkpoint_seconds=0.)

    def guard():
        watch.check()
        observer.require_active()

    def publish(status):
        start = time.perf_counter()
        clips = [] if video is None else video.clips
        complete = [clip for clip in clips if clip['full_action']]
        state = dict(status=status, run_id=arm['run_id'], controls=raw.common_step_counter,
                     completed_updates=len(rows), clips=clips, images=[latest_frame] if latest_frame else [])
        view = dict(status=status, run_id=arm['run_id'], updated_at=datetime.now(timezone.utc).isoformat(),
                    controls=state['controls'], transitions=state['controls']*worlds, environments=worlds,
                    planned_updates=updates, completed_updates=len(rows), rows=list(rows),
                    frame=latest_frame['file'] if latest_frame else None, latest_clip=complete[-1]['file'] if complete else None,
                    checkpoint=checkpoint.name if checkpoint.exists() else None)
        write(record / 'state.json', state)
        write(record / 'observation.json', view)
        observer.publish(record, view)
        components['publication_seconds'] += time.perf_counter() - start

    def no_step(*_, **__):
        raise RuntimeError('Physical step before verification of the training initial state')

    try:
        with watch.active():
            guard()
            with patch.object(ManagerBasedRlEnv, 'step', no_step), patch.object(Simulation, 'step', no_step):
                raw = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode=None)
                env = SingleResetForwardRollout(raw, route, draft, read(root / route['feedback_screen']))
                env.seed(plan['seed'])
                env.reset()
                runner = MjlabOnPolicyRunner(env, asdict(agent), str(record / 'tensorboard'), 'cuda:0')
                runner.logger.git_status_repos = []
                runner.alg.load(parent, load_cfg=None, strict=True)
                env.actor = runner.alg.actor
                env.actor.obs_normalizer.until = 0
                env.actor.requires_grad_(False)
                for field in ('actor', 'critic'):
                    check_exact(getattr(runner.alg, field).state_dict(), parent[field + '_state_dict'], field)
                demand(runner.alg.optimizer.state_dict() == parent['optimizer_state_dict'], 'Optimizer differs')
                fields = {k: getattr(raw.sim.model, k).cpu().numpy() for k in RESET_FIELDS}
                with np.load(root / plan['nominal_root'] / 'nominal-primary-fields.npz', allow_pickle=False) as f:
                    expected = {k: np.broadcast_to(f[k], fields[k].shape) for k in RESET_FIELDS}
                compare_reset_fields(fields, expected, foot_geom_ids=resolve_foot_geometry_ids(raw.sim.mj_model),
                                     friction=draft['foot_sliding_friction'], worlds=worlds)
                demand(raw.common_step_counter == raw._sim_step_counter == 0, 'Unexpected zero-step state')
                watch.sample()
                write(record / 'starting-state.json', dict(controls=0, worlds=worlds, exact_shared_parent=True,
                      parent_sha256=plan['parent_sha256'], yaw_cost_weight=-weight if role == 'candidate' else 0.))
            publish('gpu-zero-step')
            snapshot = Snapshot(raw, record)
            video = ActionVideo(snapshot, 'switch', record, sequence_seconds=plan['video_seconds'])
            video.plan['skill'] = 'forward'
            video.sample()
            latest_frame = video.image
            publish('training')
            robot = raw.scene['robot'].data
            obs_stats = torch.zeros(5, device=env.device)
            step_env, ppo_update, report_log = env.step, runner.alg.update, runner.logger.log
            runner.save = lambda *_args, **_kwargs: None
            recording_starts = {round(cfg.episode_length_s/.02)} if arm.get('record_late') else set()
            started = batch_start = time.perf_counter()
            cpu_start = time.process_time()

            def step(actions):
                guard()
                demand(raw.common_step_counter < rollout*updates, 'Training sample budget exhausted')
                obs, reward, done, extras = step_env(actions)
                alive = ~done.bool()
                obs_stats[0] += reward.sum()
                obs_stats[1] += (robot.root_link_lin_vel_b[:, 0] * alive).sum()
                obs_stats[2] += (robot.root_link_ang_vel_b[:, 2].abs() * alive).sum()
                obs_stats[3] += ((robot.projected_gravity_b[:, 2] > -0.75) & alive).sum()
                obs_stats[4] += alive.sum()
                if raw.common_step_counter in recording_starts:
                    video.last_started = None
                    video.sample()
                elif video.writer is not None:
                    video.sample()
                return obs, reward, done, extras

            def optimizer():
                guard()
                demand(raw.common_step_counter == (len(rows)+1)*rollout, 'Wrong sampling phase')
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = ppo_update()
                torch.cuda.synchronize()
                components['optimizer_seconds'] += time.perf_counter() - start
                guard()
                return result

            def log(*args, **kwargs):
                nonlocal batch_start, latest_frame
                update = int(kwargs['it']) + 1
                demand(update == len(rows) + 1, 'Unexpected PPO update number')
                losses = {k: float(v) for k, v in kwargs['loss_dict'].items()}
                reward, velocity, yaw, tipped, valid = obs_stats.tolist()
                elapsed = time.perf_counter() - batch_start
                demand(valid > 0 and all(math.isfinite(v) for v in (*losses.values(), reward, velocity, yaw)),
                       'Nonfinite PPO update or missing physical samples')
                row = dict(update=update, control=raw.common_step_counter,
                           environment_reward_mean=reward/(worlds*rollout), body_forward_m_s=velocity/valid,
                           body_abs_yaw_rad_s=yaw/valid, inclined_fraction=tipped/valid,
                           reset_fraction=1-valid/(worlds*rollout), value_loss=losses.get('value_function'),
                           policy_loss=losses.get('surrogate'), loss=losses, measured_batch_wall_seconds=elapsed,
                           net_transitions_per_second=worlds*rollout/elapsed)
                with metrics.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                rows.append(row)
                if update == warmup:
                    check_exact(env.actor.state_dict(), parent['actor_state_dict'], 'critic-only actor')
                    env.actor.requires_grad_(True)
                if update % 4 == 0 or update == updates:
                    t0 = time.perf_counter()
                    latest_frame = snapshot.capture()
                    components['snapshot_seconds'] += time.perf_counter() - t0
                if update in (warmup, updates):
                    t0 = time.perf_counter()
                    path = checkpoint if update == updates else record / f'critic-{update}.pt'
                    save_checkpoint(path, runner, raw, arm, plan['parent_sha256'], update)
                    components['checkpoint_seconds'] += time.perf_counter() - t0
                publish('training')
                print('FORWARD_PPO_UPDATE', arm['run_id'], update, updates, row['body_forward_m_s'],
                      row['body_abs_yaw_rad_s'], row['net_transitions_per_second'], flush=True)
                quality = plan['training_quality']
                recent = rows[-quality['consecutive_updates']:]
                if update >= warmup + quality['consecutive_updates']:
                    demand(not all(r['body_forward_m_s'] < quality['minimum_training_speed_m_s']
                               or r['body_abs_yaw_rad_s'] > quality['maximum_training_abs_yaw_rad_s']
                               or r['inclined_fraction'] > quality['maximum_inclined_fraction'] for r in recent),
                           'Forward gait left the prespecified training envelope')
                report_log(*args, **kwargs)
                obs_stats.zero_()
                batch_start = time.perf_counter()

            env.step, runner.alg.update, runner.logger.log = step, optimizer, log
            runner.learn(num_learning_iterations=updates, init_at_random_ep_len=False)
            torch.cuda.synchronize()
            video.close()
            publish('training')
            measured = time.perf_counter() - started
            demand(raw.common_step_counter == updates*rollout and raw._sim_step_counter == updates*rollout*5
                   and len(rows) == updates and checkpoint.is_file(), 'Incomplete PPO sample or update budget')
            return dict(status='collected-review-required', worlds=worlds, updates=len(rows), role=role,
                        transitions=worlds*rollout*updates, controls=raw.common_step_counter,
                        wall_seconds=measured, process_cpu_seconds=time.process_time()-cpu_start,
                        net_transitions_per_second=worlds*rollout*updates/measured, device=resources(watch),
                        components=dict(components, video_seconds=video.elapsed_seconds), clips=video.clips,
                        checkpoint_sha256=sha(checkpoint), metrics_sha256=sha(metrics), hardware_io=False)
    except BaseException as error:
        write(record / 'worker-stop.json', dict(reason=f'{type(error).__name__}: {error}',
                   controls=raw.common_step_counter if raw is not None else 0, complete_updates=len(rows)))
        raise
    finally:
        if video is not None:
            video.close()
        if snapshot is not None:
            snapshot.close()
        if raw is not None:
            raw.close()
