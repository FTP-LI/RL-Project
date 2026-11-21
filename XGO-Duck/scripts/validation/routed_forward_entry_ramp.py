"""Pair two startup commands at one verified standing state."""

from contextlib import ExitStack
from hashlib import sha256
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from action_video import ActionVideo
from command_input_mapping import transform
from evaluate_getup_dual_pose_route import base, guarded_actor, routes
from getup_release_fork_state import digest, signature
from getup_release_scene_state import SceneReleaseSnapshot
from routed_getup_sequence import FORWARD, GETUP, STAND, ForwardFeedback, SequenceController
from standup_metrics import classify_release, projected_gravity
from training_watch import Snapshot


ROLES = ('step-command', 'ramped-command')
cpu, write, sha = base.shared.cpu, base.write, base.sha


def command_table(branch, controls):
    ramp = branch['ramp_controls']
    fraction = np.minimum(np.arange(1, controls + 1) / max(ramp, 1), 1.)
    commands = np.zeros((controls, 3), dtype=np.float32)
    commands[:, 0] = fraction * branch['external_forward_m_s']
    return transform(commands)


def capture_controller(controller, actor):
    return dict(mode=cpu(controller.mode), dwell=cpu(controller.dwell), history=cpu(controller.history),
        feedback_previous=controller.feedback.previous.copy(), feedback_history=controller.feedback.history.copy(),
        release_gravity=cpu(actor.first_gravity), right=cpu(actor.right),
        supine=cpu(actor.parent.supine_at_release), getup_calls=np.asarray(actor.calls))


def restore_controller(controller, actor, saved):
    for key in ('mode', 'dwell', 'history'):
        getattr(controller, key).copy_(torch.from_numpy(saved[key]))
    for key in ('previous', 'history'):
        np.copyto(getattr(controller.feedback, key), saved['feedback_' + key])
    actor.calls = int(saved['getup_calls'])
    for owner, attr, key in ((actor, 'first_gravity', 'release_gravity'), (actor, 'right', 'right'),
                              (actor.parent, 'supine_at_release', 'supine')):
        getattr(owner, attr).copy_(torch.from_numpy(saved[key]))


def collect(env, actors, walking, memory, spec, record, state, publish, guard):
    p, s, d = (spec[key] for key in ('environment', 'sequence', 'pair'))
    release, stand_at, forward_at = s['boundary_controls']
    n, dt = env.num_envs, env.step_dt
    horizon = forward_at + d['forward_controls_per_branch']
    homes = [p['model_home'], *[spec['switch_policies'][key]['home'] for key in ('stand', 'forward')]]
    controller = SequenceController(n, env.device, s['controller'], homes, p['joint_limits'],
        ForwardFeedback(spec['walking'], n), release=release, stand_at=stand_at, forward_at=forward_at)
    fallback = guarded_actor(spec['common_fallback'], env.device)
    getup = routes(actors, fallback, lambda: env.common_step_counter, spec)['candidate']
    networks = {GETUP: getup, STAND: walking['stand'], FORWARD: walking['forward']}
    weights = lambda: {role: base.shared.signature(net.state_dict()) for role, net in networks.items()}
    original_weights = weights()
    robot, term = env.scene['robot'], env.action_manager.get_term('joint_pos')
    joints, names = robot.find_joints(spec['walking']['joint_names'], preserve_order=True)
    assert names == spec['walking']['joint_names']
    assert not term.standing_mask.any() and not term.neck_home_mask.any()
    capacity = int(env.sim.wp_data.contact.dist.shape[0])
    counts = state['actual']
    started = monotonic()
    sim_step, process = env.sim.step, term.process_actions
    motor = []
    contact_peak = 0
    control = -1

    def physical_step():
        assert counts['attempted_substeps'] < spec['maximum_physics_substeps']
        if control == forward_at:
            motor.append(cpu(env.sim.data.ctrl))
        counts['attempted_substeps'] += 1
        sim_step()
        counts['substeps'] += 1

    def forbid_reset(*args, **kwargs):
        raise AssertionError('The shared sequence permits one initial reset')

    def process_actions(actions):
        previous = term.previous_target.clone()
        process(actions)
        if control >= stand_at:
            controller.apply(term, previous, actions, alternate, bridge, control, dt)

    def step(k, command=None):
        nonlocal control, alternate, bridge, contact_peak
        control = k
        assert env.common_step_counter == k and bool((env.episode_length_buf == k).all())
        sensed = env.obs_buf['actor'].clone()
        actor_input, bridge = controller.advance(k, sensed)
        if command is not None:
            actor_input[controller.mode == FORWARD, 48:51] = command
        raw, yaw, alternate = controller.infer(k, actor_input, networks)
        counts['attempted_controls'] += 1
        obs, reward, terminated, truncated, _ = env.step(raw)
        counts['controls'] += 1
        assert env.common_step_counter == k + 1 and bool((env.episode_length_buf == k + 1).all())
        assert not (terminated | truncated).any() and torch.isfinite(reward).all()
        torch.testing.assert_close(obs['actor'][:, 34:48], raw, atol=0, rtol=0)
        count = int(env.sim.wp_data.nacon.numpy()[0])
        assert 0 <= count < capacity
        contact_peak = max(count, contact_peak)
        if command is not None:
            return dict(control=k + 1, qpos=cpu(env.sim.data.qpos), qvel=cpu(env.sim.data.qvel),
                angular_velocity=cpu(robot.data.root_link_ang_vel_b), velocity=cpu(robot.data.root_link_lin_vel_b[:, :2]),
                mode=cpu(controller.mode), sensed=cpu(sensed), actor_input=cpu(actor_input), raw=cpu(raw),
                feedback_yaw=cpu(yaw), bridge=cpu(bridge), target=cpu(term._processed_actions),
                motor_joint_target=cpu(robot.data.joint_pos_target[:, joints]))

    def progress(role, step_number):
        guard()
        assert monotonic() - started <= spec['maximum_sampling_wall_seconds']
        name = f'frame-{role}-{step_number:04d}.jpg'
        Image.fromarray(snap.render()).save(record / name, quality=88)
        state.update(frame=name, phase=role, sequence_control=step_number)
        publish()
        print('ROUTED_ENTRY_RAMP', role, step_number, counts['controls'], flush=True)

    alternate = bridge = None
    try:
        with ExitStack() as stack:
            for scope in (torch.inference_mode(), patch.object(env, '_reset_idx', forbid_reset),
                          patch.object(env.sim, 'step', physical_step), patch.object(term, 'process_actions', process_actions)):
                stack.enter_context(scope)
            snap = Snapshot(env, record)
            stack.callback(snap.close)
            render = snap.render
            frame_key, pixels = None, None
            frame_hashes = {'shared': [], **{role: [] for role in ROLES}}
            active = 'shared'

            def source_frame():
                nonlocal frame_key, pixels
                key = (active, env.common_step_counter)
                if key != frame_key:
                    qpos = cpu(env.sim.data.qpos[0])
                    pixels = render()
                    np.testing.assert_array_equal(snap.data.qpos.astype(qpos.dtype), qpos)
                    frame_hashes[active].append([key[1], sha256(qpos.tobytes()).hexdigest()])
                    frame_key = key
                return pixels

            snap.render = source_frame
            videos = {}
            for role in ROLES:
                folder = record / role
                folder.mkdir()
                videos[role] = ActionVideo(snap, 'switch', folder, horizon * dt)
                stack.callback(videos[role].close)
                videos[role].sample()
            for k in range(forward_at):
                step(k)
                for video in videos.values():
                    video.sample()
                if k + 1 == release:
                    poses = np.asarray(classify_release(projected_gravity(cpu(robot.data.root_link_quat_w)),
                        p['criteria']['release_alignment_min']))
                    coverage = {pose: int((poses == pose).sum()) for pose in (*base.shared.POSES, 'other')}
                    np.savez_compressed(record / 'release-poses.npz', poses=poses)
                    state.update(release_coverage=coverage, release_poses_sha256=sha(record / 'release-poses.npz'))
                    assert coverage['right-side'] >= d['minimum_actual_right_coverage'], coverage
                    assert min(coverage[pose] for pose in base.shared.POSES) >= d['minimum_other_actual_pose_coverage'], coverage
                if (k + 1) % 50 == 0:
                    progress('shared', k + 1)

            saved = capture_controller(controller, getup)
            saved_hash = digest(signature(saved))
            np.savez_compressed(record / 'fork-controller.npz', **saved)
            # Save the exact plant, sensor, delay-ring and RNG fields without replacing CUDA graph addresses.
            fork = SceneReleaseSnapshot(env, memory, spec['maximum_warp_snapshot_bytes'])
            captured = fork.verify()
            assert captured['counters'] == [forward_at, forward_at * 5]
            np.savez_compressed(record / 'fork-input.npz', sensed=cpu(env.obs_buf['actor']),
                previous=cpu(term.previous_target), qpos=cpu(env.sim.data.qpos))
            write(record / 'fork.json', dict(plant=captured, fields=fork.expected, controller_sha256=saved_hash,
                controller_file_sha256=sha(record / 'fork-controller.npz'), input_sha256=sha(record / 'fork-input.npz')))
            state['fork_sha256'] = sha(record / 'fork.json')
            state['branches'] = {}
            guard()

            for role in ROLES:
                active = role
                folder = record / role
                restored = fork.verify() if role == ROLES[0] else fork.restore()
                restore_controller(controller, getup, saved)
                assert restored == captured and digest(signature(capture_controller(controller, getup))) == saved_hash
                write(folder / 'restored-fork.json', dict(plant=restored, controller_sha256=saved_hash))
                state['branches'][role] = dict(restore_sha256=sha(folder / 'restored-fork.json'))
                commands = torch.as_tensor(command_table(d['branches'][role], horizon - forward_at), device=env.device)
                rows = []
                motor.clear()
                try:
                    for k in range(forward_at, horizon):
                        rows.append(step(k, commands[k - forward_at]))
                        if k == forward_at:
                            assert len(motor) == 5
                            ready = int((controller.mode == FORWARD).sum())
                            state['branches'][role]['forward_ready'] = ready
                            np.savez_compressed(folder / 'first-motor.npz', substeps=np.stack(motor))
                            assert ready >= n * d['minimum_forward_ready_fraction'], ready
                        videos[role].sample()
                        if (k + 1 - forward_at) % 25 == 0:
                            progress(role, k + 1)
                finally:
                    if rows:
                        trace = folder / 'trace.npz'
                        np.savez_compressed(trace, **{key: np.stack([r[key] for r in rows]) for key in rows[0]})
                        state['branches'][role].update(recorded_controls=len(rows), trace_sha256=sha(trace))
                clip, = videos[role].clips
                hashes = sorted(dict(frame_hashes['shared'] + frame_hashes[role]).items())
                recorded = set(clip['common_steps'])
                selected = [[k, value] for k, value in hashes if k in recorded]
                assert clip['complete_clip'] and [k for k, _ in selected] == clip['common_steps']
                movie = folder / clip['file']
                state['latest_clip'] = movie.relative_to(record).as_posix()
                state['branches'][role].update(video=dict(clip, path=state['latest_clip'], sha256=sha(movie),
                    qpos_sha256_by_frame=selected), motor_sha256=sha(folder / 'first-motor.npz'))
                publish()
            assert weights() == original_weights
            state['actor_weights_unchanged'] = True
    finally:
        state.update(contact_capacity=capacity, contact_peak=contact_peak, sampling_wall_seconds=monotonic()-started)
