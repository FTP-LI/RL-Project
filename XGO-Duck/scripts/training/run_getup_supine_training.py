"""Visible, bounded continuation from the same trained getup parent."""
from functools import partial
import json
import time

import numpy as np
import torch

from action_video import ActionVideo
from getup_anchor_runtime import RecordingSnapshot, startup
from getup_contact_pilot import export, kinematic_stable
from getup_supine_learning import actual_supine_at_release, in_early_supine_window, load_parent
from getup_supine_ppo import SupineWeightedPPO
from getup_teacher_anchor_train import masks, prepare, save_checkpoint
from getup_value_bootstrap import td
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_getup_shared_release import load_arrays, same_bytes
from verify_getup_teacher_anchor_train import exact


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


def check_matched_start(record, reference):
    same_bytes(load_arrays(record/'paired-start.npz'), load_arrays(reference/'paired-start.npz'))
    current = torch.load(record/'model-init.pt', map_location='cpu', weights_only=True)
    previous = torch.load(reference/'model-init.pt', map_location='cpu', weights_only=True)
    exact(current, previous)


def initialize(p, record, scene):
    from mjlab.envs import ManagerBasedRlEnv

    cfg, display, meta = prepare(p)
    write(record/'display.json', display)
    env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
    try:
        env.seed(p['seed'])
        obs, _ = env.reset()
        before = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(env.device).clone())
        with np.load(ROOT/p['anchor_data']) as data:
            replay = data['observations'].copy(), data['teacher_targets'].copy(), read(ROOT/p['anchor_cells'])
        optimizer = partial(SupineWeightedPPO, focus_weight=p['focus_weight'])
        algorithm, parent = load_parent(ROOT/p['parent_checkpoint'], p, replay, env.device, optimizer)
        for role in ('actor', 'critic'):
            exact(getattr(algorithm, role).state_dict(), parent[f'{role}_state_dict'])
            exact(getattr(algorithm, f'{role}_optimizer').state_dict(), parent[f'{role}_optimizer_state_dict'])
        startup(env, obs, algorithm.actor, p, record, scene)
        assert torch.equal(before[0], torch.get_rng_state())
        assert torch.equal(before[1], torch.cuda.get_rng_state(env.device))
        assert env.common_step_counter == env._sim_step_counter == 0
        assert env.num_envs == p['num_envs'] and env.step_dt == .02 and env.max_episode_length == 400
        reward = env.reward_manager
        index = reward._term_names.index('late_head_leg_contact')
        assert reward._scale_by_dt and not reward._term_cfgs[index].params['enabled']
        term = env.action_manager.get_term('joint_pos')
        assert not term.standing_mask.any() and term._processed_actions.shape == (env.num_envs, 14)
        with torch.no_grad():
            drift = algorithm.actor(td('actor', algorithm.anchor_inputs))-algorithm.anchor_targets
        drift_rms, drift_max = float(drift.square().mean().sqrt()), float(drift.abs().max())
        assert drift_rms <= p['tripwire']['success_replay_rms_rad_max']
        assert drift_max <= p['tripwire']['success_replay_abs_rad_max']
        initial = dict(actor=cpu(obs['actor']), critic=cpu(obs['critic']), qpos=cpu(env.sim.data.qpos),
                       qvel=cpu(env.sim.data.qvel), actuator_ctrl=cpu(env.sim.data.ctrl),
                       imu=cpu(env._imu_misalign_quat), episode_age=cpu(env.episode_length_buf),
                       cpu_rng=cpu(before[0]), cuda_rng=cpu(before[1]))
        initial.update({'expanded__'+name: cpu(getattr(env.sim.model, name)) for name in sorted(env.sim.expanded_fields)})
        np.savez_compressed(record/'paired-start.npz', **initial)
        save_checkpoint(record/'model-init.pt', algorithm, env, 0)
        write(record/'parent-readback.json', dict(
            physical_substeps=0, matched_networks_and_adam=True, parent_sha256=p['parent_checkpoint_sha256'],
            actor_adam_steps=parent['actor_steps_this_run'],
            critic_adam_steps=next(iter(parent['critic_optimizer_state_dict']['state'].values()))['step'].item(),
            parent_drift_rms_rad=drift_rms, parent_drift_max_rad=drift_max, physical_rng_unchanged=True,
            start_archive_sha256=sha(record/'paired-start.npz'), initial_checkpoint_sha256=sha(record/'model-init.pt'),
        ))
        return env, obs, algorithm, meta
    except BaseException:
        env.close()
        raise


def zero_step(p, record, scene):
    record.mkdir(parents=True)
    env, _, _, _ = initialize(p, record, scene)
    env.close()


def train(p, record, scene, zero_record, protocol_sha, limit, check_active):
    assert p['training_updates'] * limit['rollout_controls'] == limit['controls']
    assert limit['controls'] * p['num_envs'] == limit['transitions']
    record.mkdir()
    write(record/'protocol.json', p)
    write(record/'command.json', dict(visible_tty=True, protocol_sha256=protocol_sha))
    state = dict(status='gpu-zero-step', run_id=p['run_id'], arm=p['arm'], completed_updates=0,
                 controls=0, substeps=0, protocol_sha256=protocol_sha, clips=[])
    env = algorithm = snapshot = video = None
    trace, releases, masks_at_release, release_steps = [], [], [], []
    counts = dict(controls=0, attempted_controls=0, substeps=0, attempted_substeps=0, terminal_resets=0)
    began = time.monotonic()

    def publish(**fields):
        state.update(fields, **counts)
        if video is not None:
            state.update(clips=video.clips, images=[video.image] if video.image else [])
        write(record/'state.json', state)

    try:
        publish()
        env, obs, algorithm, meta = initialize(p, record, scene)
        check_matched_start(record, zero_record)
        publish(status='training', paired_zero_start_verified=True)
        actor, critic = algorithm.actor, algorithm.critic
        term = env.action_manager.get_term('joint_pos')
        snapshot = RecordingSnapshot(env, record)
        video = ActionVideo(snapshot, 'getup', record)
        video.sample()
        terminal = {}
        video_hook = env.recorder_manager.record_pre_reset
        old_physics = env.sim.step
        addresses = torch.tensor(p['qpos_addresses'], device=env.device)
        ranges = torch.tensor(p['joint_limits'], device=env.device)
        slots = int(env.sim.wp_data.contact.dist.shape[0])
        peak, bad_streak, completed = 0, 0, 0
        release_supine = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        def capture_terminal(ids):
            terminal.update(qpos=env.sim.data.qpos[ids].clone(), qvel=env.sim.data.qvel[ids].clone(),
                            requested=term.requested_target[ids].clone(), applied=term._processed_actions[ids].clone())
            assert torch.equal(ids, torch.arange(env.num_envs, device=env.device))
            counts['terminal_resets'] += 1
            video_hook(ids)

        def count_physics():
            assert counts['attempted_substeps'] < limit['substeps']
            counts['attempted_substeps'] += 1
            old_physics()
            counts['substeps'] += 1

        env.recorder_manager.record_pre_reset = capture_terminal
        env.sim.step = count_physics
        with (record/'learning-metrics.jsonl').open('x', buffering=1) as log:
            for update in range(1, p['training_updates']+1):
                check_active()
                batch = []
                for _ in range(limit['rollout_controls']):
                    step = counts['controls']
                    age = env.episode_length_buf.clone()
                    control_age = step % 400
                    assert (age == control_age).all() and not term.standing_mask.any()
                    assert counts['attempted_controls'] < limit['controls']
                    active, _ = masks(age)
                    if control_age == p['actual_release_control']:
                        quaternion = env.sim.data.qpos[:, 3:7]
                        release_supine = actual_supine_at_release(quaternion, p['release_pose_minimum_alignment'])
                        releases.append(cpu(quaternion))
                        masks_at_release.append(cpu(release_supine))
                        release_steps.append(step)
                    focus = in_early_supine_window(release_supine, age, *p['focus_interval'])
                    terminal.clear()
                    before = cpu(env.sim.data.qpos[:32])
                    with torch.no_grad():
                        action = actor(obs, stochastic_output=True)
                        torch.testing.assert_close(actor.output_std, torch.full_like(actor.output_std, p['learner']['initial_std']),
                                                   atol=1e-7, rtol=0)
                        row = dict(actor=obs['actor'].clone(), critic=obs['critic'].clone(), action=action.clone(),
                                   mean=actor.output_mean.clone(), log_prob=actor.get_output_log_prob(action).clone(),
                                   value=critic(obs).squeeze(-1).clone(), active=active, early_supine=focus)
                        counts['attempted_controls'] += 1
                        obs, reward, terminated, timeout, _ = env.step(action)
                        counts['controls'] += 1
                    done = terminated | timeout
                    assert not terminated.any() and torch.equal(done, age == 399) and torch.equal(action, row['action'])
                    history = torch.where((active & ~done)[:, None], row['action'], torch.zeros_like(action))
                    torch.testing.assert_close(obs['actor'][:, 34:48], history, atol=0, rtol=0)
                    row.update(reward=reward.clone()/p['reward_scale'], done=done.clone())
                    batch.append(row)
                    qpos = terminal['qpos'] if terminal else env.sim.data.qpos
                    qvel = terminal['qvel'] if terminal else env.sim.data.qvel
                    requested = terminal['requested'] if terminal else term.requested_target
                    applied = terminal['applied'] if terminal else term._processed_actions
                    assert all(torch.isfinite(v).all() for v in (qpos, qvel, reward, requested, applied, row['mean'], action))
                    stable = float(kinematic_stable(qpos, qvel, addresses, ranges).float().mean())
                    bad_streak = bad_streak+1 if control_age >= 349 and stable < p['watch_stable_min'] else 0
                    contact = int(env.sim.wp_data.nacon.numpy()[0])
                    peak = max(peak, contact)
                    assert 0 <= contact < slots and bad_streak < 25, 'Contact capacity or late stability guard'
                    trace.append(dict(pre_qpos=before, qpos=cpu(qpos[:32]), qvel=cpu(qvel[:32]),
                        inputs=cpu(row['actor'][:32]), actions=cpu(action[:32]), requested=cpu(requested[:32]),
                        applied=cpu(applied[:32]), focused=cpu(focus[:32]), focus_count=int(focus.sum()), stable=stable,
                        done=bool(done.all())))
                    if control_age == 399:
                        release_supine.zero_()
                    if counts['controls'] < limit['controls'] or video.writer is not None:
                        video.sample()
                    if counts['controls'] % 24 == 0:
                        check_active()
                        assert time.monotonic()-began < limit['wall_seconds_per_arm'], 'Training wall-time cap'
                tensors = {key: torch.stack([row[key] for row in batch]) for key in batch[0]}
                with torch.no_grad():
                    last = critic(obs).squeeze(-1)
                metrics = algorithm.update(tensors, last)
                completed = update
                with torch.no_grad():
                    drift = actor(td('actor', algorithm.anchor_inputs))-algorithm.anchor_targets
                metrics.update(update=update, controls=counts['controls'], elapsed=time.monotonic()-began,
                               stable=stable, drift_rms=float(drift.square().mean().sqrt()), drift_max=float(drift.abs().max()))
                log.write(json.dumps(metrics, allow_nan=False)+'\n')
                publish(completed_updates=completed, metrics=metrics)
                print(f'SUPINE_PPO {p["arm"]} {update:02}/{p["training_updates"]} | controls={counts["controls"]} '
                      f'| focus={metrics["early_supine_samples"]} | stable={stable:.1%} '
                      f'| drift={metrics["drift_rms"]:.4f}/{metrics["drift_max"]:.4f} rad', flush=True)
                assert (metrics['drift_rms'] <= p['tripwire']['success_replay_rms_rad_max'] and
                        metrics['drift_max'] <= p['tripwire']['success_replay_abs_rad_max']), 'Successful-state drift guard'
                if update == 25:
                    save_checkpoint(record/'model-25.pt', algorithm, env, update)
        assert counts['controls'] == env.common_step_counter == limit['controls']
        assert counts['substeps'] == env._sim_step_counter == limit['substeps']
        assert counts['terminal_resets'] == limit['controls']//400
        save_checkpoint(record/'model-final.pt', algorithm, env, completed)
        video.close()
        with np.load(ROOT/p['probe_dataset']) as data:
            probe = torch.from_numpy(data['actor_obs_sample'][[0, 99, 100, 125, 200, 300, 399]].reshape(-1, 61).copy())
        export(actor, meta, probe, p, record)
        publish(status='collected-review-required', contact_peak=peak, contact_capacity=slots)
    except BaseException as error:
        if algorithm is not None and env is not None:
            save_checkpoint(record/'stopped-partial.pt', algorithm, env, state['completed_updates'])
        publish(status='stopped-review-required', error=repr(error))
        raise
    finally:
        if video is not None:
            video.close()
        if snapshot is not None:
            snapshot.close()
        if env is not None:
            if hasattr(env, '_getup_contact_audits'):
                write(record/'contact-audits.json', env._getup_contact_audits)
            env.close()
        if trace:
            np.savez_compressed(record/'training-trace.npz', **{key: np.asarray([row[key] for row in trace]) for key in trace[0]})
        if releases:
            np.savez_compressed(record/'actual-release.npz', quaternion=np.stack(releases),
                                supine=np.stack(masks_at_release), controls=np.asarray(release_steps))
        publish(elapsed_seconds=time.monotonic()-began)
    return state
