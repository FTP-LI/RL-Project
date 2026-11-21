"""Collect one complete routed-getup, stand and forward trajectory on CUDA."""

from contextlib import ExitStack
from hashlib import sha256
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from action_video import ActionVideo
from evaluate_getup_dual_pose_route import base, guarded_actor, routes
from routed_getup_sequence import FORWARD, GETUP, STAND, ForwardFeedback, SequenceController
from standup_metrics import classify_release, projected_gravity
from training_watch import Snapshot


cpu, write, sha = base.shared.cpu, base.write, base.sha


def collect(env, actors, walking_actors, meta, spec, record, state, publish, check_active):
    p, s = spec['environment'], spec['sequence']
    controls, boundaries = spec['maximum_unique_controls'], s['boundary_controls']
    release, stand_at, forward_at = boundaries
    n, dt = env.num_envs, env.step_dt
    feedback = ForwardFeedback(spec['walking'], n)
    homes = [p['model_home'], *[spec['switch_policies'][role]['home'] for role in ('stand', 'forward')]]
    controller = SequenceController(n, env.device, s['controller'], homes, p['joint_limits'], feedback,
        release=release, stand_at=stand_at, forward_at=forward_at)
    fallback = guarded_actor(spec['common_fallback'], env.device)
    getup = routes(actors, fallback, lambda: env.common_step_counter, spec)['candidate']
    networks = {GETUP: getup, STAND: walking_actors['stand'], FORWARD: walking_actors['forward']}
    signatures = {key: base.shared.signature(model.state_dict()) for key, model in networks.items()}
    robot, manager = env.scene['robot'], env.action_manager
    joints, names = robot.find_joints(meta['joint_names'].split(','), preserve_order=True)
    feet, sites = robot.find_sites(('left_foot', 'right_foot'), preserve_order=True)
    assert names == spec['walking']['joint_names'] and list(sites) == ['left_foot', 'right_foot']
    model = env.sim.mj_model
    addresses = [int(model.joint('robot/' + name).qposadr[0]) for name in names]
    term = manager.get_term('joint_pos')
    assert not term.standing_mask.any() and not term.neck_home_mask.any()
    action_method, sim_step = term.process_actions, env.sim.step
    rows, chunks, frame_hashes, boundary_motor = [], [], [], []
    counts = state['actual']
    started, peak = monotonic(), 0
    contact_capacity = int(env.sim.wp_data.contact.dist.shape[0])
    record_window = s['record_window_controls']
    last_frame, frame_pixels = -1, None
    state.update(qpos_joint_addresses=addresses, contact_capacity=contact_capacity, record_chunks=chunks)
    np.savez_compressed(record / 'initial.npz', qpos=cpu(env.sim.data.qpos), qvel=cpu(env.sim.data.qvel),
        requested_pose=np.asarray([base.shared.POSES[i] for i in p['split']['pose_codes']]))

    def flush():
        if rows:
            folder = record / f'trace-{rows[0]["control"]:04d}-{rows[-1]["control"]:04d}.npz'
            np.savez_compressed(folder, **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
            chunks.append(dict(first=rows[0]['control'], last=rows[-1]['control'], file=folder.name, sha256=sha(folder)))
            rows.clear()

    def physical_step():
        if control in (stand_at, forward_at):
            boundary_motor.append(cpu(env.sim.data.ctrl))
        assert counts['attempted_substeps'] < spec['maximum_physics_substeps']
        counts['attempted_substeps'] += 1
        sim_step()
        counts['substeps'] += 1

    def forbid_reset(*args, **kwargs):
        raise AssertionError('A continuous trajectory permits only the initial physical reset')

    def process(actions):
        previous = term.previous_target.clone()
        action_method(actions)
        if control >= stand_at:
            controller.apply(term, previous, actions, alternate, bridge, control, dt)

    def coverage():
        poses = np.asarray(classify_release(projected_gravity(cpu(robot.data.root_link_quat_w)),
            p['criteria']['release_alignment_min']))
        actual = {pose: int((poses == pose).sum()) for pose in (*base.shared.POSES, 'other')}
        np.savez_compressed(record / 'release-poses.npz', poses=poses)
        write(record / 'release-coverage.json', dict(actual=actual, control=release, physical_substeps=release * 5,
            poses_sha256=sha(record / 'release-poses.npz')))
        state['release_coverage'] = actual
        publish()
        assert actual['right-side'] >= s['minimum_actual_right_coverage'], actual
        assert min(actual[pose] for pose in base.shared.POSES) >= s['minimum_other_actual_pose_coverage'], actual

    try:
        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            stack.enter_context(patch.object(env, '_reset_idx', forbid_reset))
            stack.enter_context(patch.object(env.sim, 'step', physical_step))
            stack.enter_context(patch.object(term, 'process_actions', process))
            snap = Snapshot(env, record)
            stack.callback(snap.close)
            render = snap.render

            def source_frame():
                nonlocal last_frame, frame_pixels
                k = env.common_step_counter
                if k != last_frame:
                    q = cpu(env.sim.data.qpos[0])
                    frame_pixels = render()
                    np.testing.assert_array_equal(snap.data.qpos.astype(q.dtype), q)
                    frame_hashes.append(sha256(q.tobytes()).hexdigest())
                    last_frame = k
                return frame_pixels

            snap.render = source_frame
            video = ActionVideo(snap, 'switch', record, controls * dt)
            stack.callback(video.close)
            video.sample()
            for control in range(controls):
                assert env.common_step_counter == control and bool((env.episode_length_buf == control).all())
                observed = env.obs_buf['actor'].clone()
                x, bridge = controller.advance(control, observed)
                raw, yaw, alternate = controller.infer(control, x, networks)
                if control in (stand_at, forward_at):
                    boundary_motor.clear()
                counts['attempted_controls'] += 1
                obs, reward, terminated, truncated, _ = env.step(raw)
                step = control + 1
                counts['controls'] += 1
                assert env.common_step_counter == step and bool((env.episode_length_buf == step).all())
                assert not terminated.any() and bool(truncated.all() if step == controls else ~truncated.any())
                assert torch.isfinite(reward).all()
                torch.testing.assert_close(obs['actor'][:, 34:48], raw, atol=0, rtol=0)
                qpos, qvel = cpu(env.sim.data.qpos), cpu(env.sim.data.qvel)
                applied = cpu(term._processed_actions)
                assert all(np.isfinite(a).all() for a in (qpos, qvel, applied))
                contact_count = int(env.sim.wp_data.nacon.numpy()[0])
                assert 0 <= contact_count < contact_capacity
                peak = max(peak, contact_count)
                rows.append(dict(control=step, qpos=qpos, qvel=qvel,
                    velocity=cpu(robot.data.root_link_lin_vel_b[:, :2]), angular_velocity=cpu(robot.data.root_link_ang_vel_b),
                    foot_contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(n, 2, -1).any(2),
                    foot_positions=cpu(robot.data.site_pos_w[:, feet]),
                    foot_speed=np.linalg.norm(cpu(robot.data.site_lin_vel_w[:, feet, :2]), axis=-1),
                    observed=cpu(observed), actor_input=cpu(x), raw=cpu(raw), mode=cpu(controller.mode),
                    feedback_yaw=cpu(yaw), alternate=cpu(alternate), bridge=cpu(bridge),
                    requested=cpu(term.requested_target), projected=cpu(term.projected_destination), applied=applied,
                    motor_joint_target=cpu(robot.data.joint_pos_target[:, joints]),
                    terminated=cpu(terminated), truncated=cpu(truncated)))
                if control in (stand_at, forward_at):
                    assert len(boundary_motor) == 5
                    np.savez_compressed(record / f'boundary-motor-{step}.npz', substeps=np.stack(boundary_motor))
                video.sample()
                if step == release:
                    coverage()
                if step % 50 == 0 or step == controls:
                    check_active()
                    assert monotonic() - started <= spec['maximum_sampling_wall_seconds']
                    frame = f'frame-sequence-{step:04d}.jpg'
                    Image.fromarray(source_frame()).save(record / frame, quality=88)
                    state.update(frame=frame, phase=('fall' if step <= release else 'getup' if step <= stand_at
                        else 'stand' if step <= forward_at else 'forward'),
                        standing_worlds=int((controller.mode == STAND).sum()),
                        forward_worlds=int((controller.mode == FORWARD).sum()))
                    publish()
                    print('ROUTED_SEQUENCE', step, controls, state['phase'], state['forward_worlds'], flush=True)
                if step % record_window == 0:
                    flush()
            assert signatures == {key: base.shared.signature(model.state_dict()) for key, model in networks.items()}
            assert env._manual_reset_pending.all()
            np.savez_compressed(record / 'route-masks.npz', right=cpu(getup.right), supine=cpu(getup.parent.supine_at_release))
            state['route_masks_sha256'] = sha(record / 'route-masks.npz')
            state['actor_weights_unchanged'] = True
            clip, = video.clips
            assert clip['complete_clip'] and clip['frames'] == len(frame_hashes)
            state['video'] = dict(clip, sha256=sha(record / clip['file']), environment_index=0,
                qpos_sha256_by_frame=frame_hashes)
    finally:
        flush()
        state.update(contact_peak=peak, sampling_wall_seconds=monotonic()-started,
            actual_control_transitions=counts['controls'] * n)
