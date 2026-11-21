"""Collect a shared recovery prefix and two full seven-command suffixes."""

from contextlib import ExitStack
from hashlib import sha256
from time import monotonic
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from PIL import Image

from action_video import ActionVideo
from evaluate_getup_dual_pose_route import base, guarded_actor, routes
from getup_release_fork_state import digest, signature
from getup_release_scene_state import SceneReleaseSnapshot
from routed_forward_entry_ramp import capture_controller, restore_controller
from routed_getup_sequence import FORWARD, GETUP, STAND, ForwardFeedback
from routed_seven_command_controller import SevenCommandSequenceController
from routed_seven_turn_feedback_controller import SevenCommandTurnFeedbackController
from standup_metrics import classify_release, projected_gravity
from training_watch import Snapshot


ROLES = ('static-turn', 'restored-gyro')
cpu, write, sha = base.shared.cpu, base.write, base.sha


def save_trace(folder, rows, level):
    """Keep the same NPY payload and hash the closed, atomically named chunk."""
    first, last = rows[0]['control'], rows[-1]['control']
    path = folder/f'trace-{first:04d}-{last:04d}.npz'
    temporary = path.with_suffix('.tmp')
    assert not path.exists() and not temporary.exists()
    try:
        with ZipFile(temporary, 'x', compression=ZIP_DEFLATED, compresslevel=level) as archive:
            for name in rows[0]:
                with archive.open(name+'.npy', 'w', force_zip64=True) as member:
                    np.lib.format.write_array(member, np.stack([row[name] for row in rows]), allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(first=first, last=last, file=path.name, sha256=sha(path))


def controllers(spec, device):
    p, s, seven = (spec[name] for name in ('environment', 'sequence', 'seven_command'))
    homes = [p['model_home'], *[spec['switch_policies'][name]['home'] for name in ('stand', 'forward')]]
    release, stand_at, command_at = s['boundary_controls']
    return {role: cls(p['num_envs'], device, s['controller'], homes, p['joint_limits'],
            ForwardFeedback(spec['walking'], p['num_envs']), commands=seven['commands'],
            groups=seven['groups'], routes=seven['routes'], ramp=spec['ramped_forward']['ramp'],
            release=release, stand_at=stand_at, forward_at=command_at)
            for role, cls in zip(ROLES, (SevenCommandSequenceController, SevenCommandTurnFeedbackController), strict=True)}


def collect(env, actors, walking, memory, spec, record, state, publish, guard):
    p, s, d, seven = (spec[name] for name in ('environment', 'sequence', 'turn_pair', 'seven_command'))
    release, stand_at, command_at = s['boundary_controls']
    horizon, n, dt = command_at+d['terminal_controls'], env.num_envs, env.step_dt
    banks = controllers(spec, env.device)
    controller = banks[ROLES[0]]
    fallback = guarded_actor(spec['common_fallback'], env.device)
    getup = routes(actors, fallback, lambda: env.common_step_counter, spec)['candidate']
    networks = {GETUP:getup, STAND:walking['stand'], FORWARD:walking['forward']}

    def weights():
        return {role: base.shared.signature(net.state_dict()) for role, net in networks.items()}

    original_weights = weights()
    robot, term = env.scene['robot'], env.action_manager.get_term('joint_pos')
    names = spec['walking']['joint_names']
    joints, order = robot.find_joints(names, preserve_order=True)
    feet, sites = robot.find_sites(('left_foot', 'right_foot'), preserve_order=True)
    assert names == order and list(sites) == ['left_foot', 'right_foot']
    assert not term.standing_mask.any() and not term.neck_home_mask.any()
    addresses = [int(env.sim.mj_model.joint('robot/'+name).qposadr[0]) for name in names]
    turn_ids = sorted(seven['groups']['turn-left']+seven['groups']['turn-right'])
    turn_device = torch.as_tensor(turn_ids, device=env.device)
    sim_step, process = env.sim.step, term.process_actions
    capacity = int(env.sim.wp_data.contact.dist.shape[0])
    counts = state['actual']
    started, control, peak = monotonic(), -1, 0
    alternate = bridge = None
    motor = []
    state.update(qpos_joint_addresses=addresses, turn_worlds=turn_ids, shared_chunks=[], branches={})
    np.savez_compressed(record/'initial.npz', qpos=cpu(env.sim.data.qpos), qvel=cpu(env.sim.data.qvel),
        requested_pose=np.asarray([base.shared.POSES[i] for i in p['split']['pose_codes']]))

    def physical_step():
        assert counts['attempted_substeps'] < spec['maximum_physics_substeps']
        if control == command_at:
            motor.append(cpu(env.sim.data.ctrl))
        counts['attempted_substeps'] += 1
        sim_step()
        counts['substeps'] += 1

    def forbid_reset(*args, **kwargs):
        raise AssertionError('Only the original scene reset is permitted')

    def process_actions(actions):
        previous = term.previous_target.clone()
        process(actions)
        if control >= stand_at:
            controller.apply(term, previous, actions, alternate, bridge, control, dt)

    def step(k, role):
        nonlocal control, alternate, bridge, peak
        control = k
        assert env.common_step_counter == k and bool((env.episode_length_buf == k).all())
        sensed = env.obs_buf['actor'].clone()
        x, bridge = controller.advance(k, sensed)
        raw, _, alternate = controller.infer(k, x, networks)
        mode = cpu(controller.mode)
        if k == command_at:
            entered = {name:int((mode[ids] == FORWARD).sum()) for name, ids in seven['groups'].items()}
            write(record/role/'admission.json', entered)
            assert all(count >= seven['minimum_by_command'][name] for name, count in entered.items()), entered
            assert sum(entered.values()) >= seven['minimum_overall']
        counts['attempted_controls'] += 1
        obs, reward, terminated, truncated, _ = env.step(raw)
        counts['controls'] += 1
        t = k+1
        assert env.common_step_counter == t and bool((env.episode_length_buf == t).all())
        assert not terminated.any() and bool(truncated.all() if t == horizon else ~truncated.any())
        assert torch.isfinite(reward).all()
        torch.testing.assert_close(obs['actor'][:, 34:48], raw, atol=0, rtol=0)
        contacts = int(env.sim.wp_data.nacon.numpy()[0])
        assert 0 <= contacts < capacity
        peak = max(peak, contacts)
        qpos, qvel = cpu(env.sim.data.qpos), cpu(env.sim.data.qvel)
        assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
        row = dict(control=t, qpos=qpos, qvel=qvel, mode=mode, terminated=cpu(terminated),
            velocity=cpu(robot.data.root_link_lin_vel_b[:, :2]), angular_velocity=cpu(robot.data.root_link_ang_vel_b),
            foot_contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(n, 2, -1).any(2),
            foot_positions=cpu(robot.data.site_pos_w[:, feet]),
            foot_speed=np.linalg.norm(cpu(robot.data.site_lin_vel_w[:, feet, :2]), axis=-1))
        if k >= command_at:
            row.update(turn_gyro=cpu(sensed[turn_device, 2]), turn_yaw_input=cpu(x[turn_device, 50]))
        if k == command_at:
            assert len(motor) == 5
            np.savez_compressed(record/role/'first-control.npz', sensed=cpu(sensed), actor_input=cpu(x),
                raw=cpu(raw), bridge=cpu(bridge), mode=mode, target=cpu(term._processed_actions),
                motor_joint_target=cpu(robot.data.joint_pos_target[:, joints]), delayed_motor=np.stack(motor))
        return row

    def progress(role, t):
        guard()
        assert monotonic()-started <= spec['maximum_sampling_wall_seconds']
        name = f'frame-{role}-{t:04d}.jpg'
        Image.fromarray(snap.render()).save(record/name, quality=88)
        state.update(frame=name, phase=role, sequence_control=t)
        publish()
        print('ROUTED_TURN_PAIR', role, t, counts['controls'], flush=True)

    def range_trace(role, first, end, writers, chunks):
        folder = record if role == 'shared' else record/role
        rows = []
        try:
            for k in range(first, end):
                rows.append(step(k, role))
                for video in writers:
                    video.sample()
                t = k+1
                if t == release:
                    poses = np.asarray(classify_release(projected_gravity(cpu(robot.data.root_link_quat_w)),
                        p['criteria']['release_alignment_min']))
                    coverage = {name:int((poses == name).sum()) for name in (*base.shared.POSES, 'other')}
                    np.savez_compressed(record/'release-poses.npz', poses=poses)
                    state.update(release_coverage=coverage, release_poses_sha256=sha(record/'release-poses.npz'))
                    assert coverage['right-side'] >= s['minimum_actual_right_coverage']
                    assert min(coverage[name] for name in base.shared.POSES) >= s['minimum_other_actual_pose_coverage']
                if t % 50 == 0:
                    progress(role, t)
                if len(rows) == s['record_window_controls']:
                    chunks.append(save_trace(folder, rows, d['trace_zip_level']))
                    rows.clear()
        finally:
            if rows:
                chunks.append(save_trace(folder, rows, d['trace_zip_level']))

    try:
        with ExitStack() as stack:
            for scope in (torch.inference_mode(), patch.object(env, '_reset_idx', forbid_reset),
                          patch.object(env.sim, 'step', physical_step), patch.object(term, 'process_actions', process_actions)):
                stack.enter_context(scope)
            snap = Snapshot(env, record)
            stack.callback(snap.close)
            render, frame_key, pixels, active = snap.render, None, None, 'shared'
            frame_hashes = {role:{} for role in ('shared', *ROLES)}

            def source_frame():
                nonlocal frame_key, pixels
                k = env.common_step_counter
                if (active, k) != frame_key:
                    qpos = cpu(env.sim.data.qpos[0])
                    pixels = render()
                    np.testing.assert_array_equal(snap.data.qpos.astype(qpos.dtype), qpos)
                    frame_hashes[active][k] = sha256(qpos.tobytes()).hexdigest()
                    frame_key = active, k
                return pixels

            snap.render = source_frame
            videos = {}
            for role in ROLES:
                (record/role).mkdir()
                videos[role] = ActionVideo(snap, 'switch', record/role, horizon*dt)
                stack.callback(videos[role].close)
                videos[role].sample()
            range_trace('shared', 0, command_at, videos.values(), state['shared_chunks'])
            saved = capture_controller(controller, getup)
            saved_hash = digest(signature(saved))
            np.savez_compressed(record/'fork-controller.npz', **saved)
            fork = SceneReleaseSnapshot(env, memory, spec['maximum_warp_snapshot_bytes'])
            captured = fork.verify()
            assert captured['counters'] == [command_at, command_at*5]
            np.savez_compressed(record/'fork-input.npz', sensed=cpu(env.obs_buf['actor']),
                previous=cpu(term.previous_target), qpos=cpu(env.sim.data.qpos))
            write(record/'fork.json', dict(plant=captured, fields=fork.expected, controller_sha256=saved_hash,
                controller_file_sha256=sha(record/'fork-controller.npz'), input_sha256=sha(record/'fork-input.npz')))
            state['fork_sha256'] = sha(record/'fork.json')
            guard()
            for role in ROLES:
                active, controller = role, banks[role]
                restored = fork.verify() if role == ROLES[0] else fork.restore()
                restore_controller(controller, getup, saved)
                assert restored == captured and digest(signature(capture_controller(controller, getup))) == saved_hash
                folder = record/role
                write(folder/'restored-fork.json', dict(plant=restored, controller_sha256=saved_hash))
                branch = dict(restore_sha256=sha(folder/'restored-fork.json'), chunks=[])
                state['branches'][role] = branch
                motor.clear()
                range_trace(role, command_at, horizon, (videos[role],), branch['chunks'])
                clip, = videos[role].clips
                hashes = frame_hashes['shared'] | frame_hashes[role]
                assert clip['complete_clip'] and clip['common_steps'] == list(range(0, horizon+1, 2))
                state['latest_clip'] = f"{role}/{clip['file']}"
                branch.update(video=dict(clip, path=state['latest_clip'], sha256=sha(record/state['latest_clip']),
                    qpos_sha256_by_frame=[hashes[k] for k in clip['common_steps']]),
                    first_control_sha256=sha(folder/'first-control.npz'))
                publish()
            assert weights() == original_weights
            state['actor_weights_unchanged'] = True
    finally:
        state.update(contact_capacity=capacity, contact_peak=peak, sampling_wall_seconds=monotonic()-started)
