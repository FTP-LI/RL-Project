"""Collect three distinct getup policies from one frozen release.

The saved anchor collector stays byte-identical. Reuse its camera and keep
its physical/readback guards. The first policy outputs may differ after the
identical observation and delayed motor command.
"""
from contextlib import ExitStack
import time
from types import SimpleNamespace

import numpy as np
import torch

import getup_shared_release as shared
from action_video import ActionVideo
from common_scene_contract import compare_arrays
from collect_getup_anchor_same_release import FixedGetupCamera
from getup_release_scene_state import SceneReleaseSnapshot
from record_io import write
from s1_reference_inputs import sha
from training_watch import Snapshot


cpu = shared.cpu


def collect(env, actors, memory, meta, spec, record, state, publish, check_active):
    p, roles = spec['environment'], spec['branches']
    release, end, n = spec['prefix_controls'], p['maximum_control_steps'], env.num_envs
    robot, manager = env.scene['robot'], env.action_manager
    joint_ids, names = robot.find_joints(meta['joint_names'].split(','), preserve_order=True)
    assert names == meta['joint_names'].split(',')
    model = env.sim.mj_model
    joint_indices = [model.joint('robot/' + name).id for name in names]
    limits = model.jnt_range[joint_indices]
    np.testing.assert_allclose(limits, p['joint_limits'], atol=1e-7, rtol=0)
    qpos_indices = model.jnt_qposadr[joint_indices].tolist()
    tolerance = np.deg2rad(p['applied_joint_limit_tolerance_deg'])
    term = manager.get_term('joint_pos')
    initial = dict(qpos=cpu(env.sim.data.qpos), qvel=cpu(env.sim.data.qvel),
                   requested_pose=np.array([shared.POSES[i] for i in p['split']['pose_codes']]))
    np.savez_compressed(record / 'initial.npz', **initial)
    state.update(qpos_joint_addresses=qpos_indices, contact_peak=0,
                 contact_capacity=int(env.sim.wp_data.contact.dist.shape[0]))
    counts = state['actual']
    traces = {role: [] for role in ('shared-prefix', *roles)}
    movies, first, release_receipts = {}, {}, {}
    weights = {role: shared.signature(actor.state_dict()) for role, actor in actors.items()}
    old_step, old_reset = env.sim.step, env._reset_idx
    phase = 'shared-prefix'
    motor = []
    began = time.monotonic()

    def count_physics():
        if phase in roles and env.common_step_counter == release:
            motor.append(cpu(env.sim.data.ctrl))
        assert counts['attempted_substeps'] < spec['maximum_physics_substeps']
        counts['attempted_substeps'] += 1
        old_step()
        counts['substeps'] += 1

    def forbid_reset(*args, **kwargs):
        raise AssertionError('The sole physical reset is the original fall')

    def advance(k):
        assert env.common_step_counter == k-1 and counts['attempted_controls'] < spec['maximum_unique_controls']
        sensed = env.obs_buf['actor']
        source, _ = memory.inputs(sensed, env.episode_length_buf, env.step_dt)
        zeros = torch.zeros_like(memory.source_history)
        # Neither standing nor neck correction participates in this protocol.
        # During the shared fall every policy would have been masked to zero.
        model_action = actors[phase](shared.td('actor', source)) if k > release else zeros
        raw = memory.select(model_action, zeros, env.episode_length_buf, env.step_dt)
        sent, observations, source_array = cpu(raw), cpu(sensed), cpu(source)
        if k <= release:
            assert not sent.any()
        assert not memory.standing.any() and not memory.neck_home_active.any()
        for name, value in memory.action_flags().items():
            setattr(term, name, value.clone())
        if k == release+1:
            motor.clear()
        counts['attempted_controls'] += 1
        obs, reward, terminated, truncated, _ = env.step(raw)
        counts['controls'] += 1
        np.testing.assert_array_equal(cpu(obs['actor'][:, 34:48]), sent)
        assert torch.isfinite(reward).all() and (env.episode_length_buf == k).all()
        assert not terminated.any() and bool(truncated.all() if k == end else ~truncated.any())
        qpos, qvel = cpu(env.sim.data.qpos), cpu(env.sim.data.qvel)
        joints = cpu(robot.data.joint_pos[:, joint_ids])
        valid = ((joints >= limits[:, 0]-tolerance) & (joints <= limits[:, 1]+tolerance)).all(axis=1)
        requested, applied = cpu(term.requested_target), cpu(term._processed_actions)
        count = int(env.sim.wp_data.nacon.numpy()[0])
        assert 0 <= count < state['contact_capacity']
        state['contact_peak'] = max(state['contact_peak'], count)
        traces[phase].append(dict(time=k*.02, active=np.ones(n, bool), qpos=qpos, qvel=qvel,
            quaternion=cpu(robot.data.root_link_quat_w), height=cpu(robot.data.root_link_pos_w[:, 2]),
            angular_velocity=cpu(robot.data.root_link_ang_vel_b),
            foot_contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(n, 2, -1).any(axis=2),
            joint_positions=joints, joint_envelope_ok=valid,
            native_head_leg_contact=cpu(env._getup_contact['contact']), observations=observations,
            actor_input=source_array, raw_policy_actions=sent, requested_targets=requested,
            applied_targets=applied, terminated=cpu(terminated), truncated=cpu(truncated)))
        assert all(np.isfinite(a).all() for a in (qpos, qvel, requested, applied, observations))
        if k == release+1:
            assert len(motor) == env.cfg.decimation
            first[phase] = dict(actor_input=source_array, raw_policy_actions=sent, requested_targets=requested,
                                applied_targets=applied, motor=np.stack(motor))
            if phase != roles[0]:
                np.testing.assert_array_equal(first[phase]['actor_input'], first[roles[0]]['actor_input'])
                np.testing.assert_array_equal(first[phase]['motor'][0], first[roles[0]]['motor'][0])
            np.savez_compressed(record / f'first-control-{phase}.npz', **first[phase])
        if k % 25 == 0:
            check_active()
            assert time.monotonic() - began <= spec['maximum_sampling_wall_seconds'], 'Sampling time limit'
        if k % 50 == 0 or k == end:
            state.update(phase=phase, episode_control=k)
            publish()
            print('SUPINE_EVAL', phase, k, '/', end, 'physical controls', counts['controls'], flush=True)

    env.sim.step, env._reset_idx = count_physics, forbid_reset
    try:
        with ExitStack() as stack, torch.no_grad():
            renderer = Snapshot(env, record)
            stack.callback(renderer.close)
            for role, cameras in spec['cameras'].items():
                for pose, index in cameras.items():
                    folder = record / f'evaluation-{role}-{pose}'
                    folder.mkdir()
                    camera = FixedGetupCamera(renderer, role, pose, index)
                    video = ActionVideo(camera, 'getup', folder)
                    video.media = SimpleNamespace(VideoWriter=camera.writer(video.media.VideoWriter))
                    assert video.horizon == end and video.stride == 2
                    stack.callback(video.close)
                    movies[role, pose] = camera, video
                    video.sample()
            for k in range(1, release+1):
                advance(k)
                for _, video in movies.values():
                    video.sample()
            payload = shared.release_payload(env, memory)
            snapshot = SceneReleaseSnapshot(env, memory, spec['maximum_warp_snapshot_bytes'])
            captured = snapshot.verify()
            assert captured['counters'] == [release, release*5]
            write(record / 'release-state-fields.json', snapshot.expected)
            for phase in roles:
                release_receipts[phase] = snapshot.verify() if phase == roles[0] else snapshot.restore()
                assert release_receipts[phase] == captured
                exact = shared.release_payload(env, memory)
                compare_arrays(exact, payload)
                np.savez_compressed(record / f'release-{phase}.npz', **exact)
                write(record / 'release-readback.json', release_receipts)
                print('SUPINE_RELEASE_READBACK', phase, 'matching fields', captured['fields'], flush=True)
                for k in range(release+1, end+1):
                    advance(k)
                    for pose in spec['cameras'][phase]:
                        movies[phase, pose][1].sample()
                assert env._manual_reset_pending.all()
                write(record / f'contact-audits-{phase}.json', env._getup_contact_audits)
            assert weights == {role: shared.signature(actor.state_dict()) for role, actor in actors.items()}
            state['actor_weights_unchanged'] = True
    finally:
        env.sim.step, env._reset_idx = old_step, old_reset
        for role, rows in traces.items():
            if rows:
                np.savez_compressed(record / f'trace-{role}.npz',
                                    **{name: np.stack([row[name] for row in rows]) for name in rows[0]})
        for (role, pose), (camera, video) in movies.items():
            if not video.clips:
                continue
            clip = video.clips[0]
            path = video.record / clip['file']
            state['conditions'].append(dict(condition=pose, role=role, video=dict(
                path=path.relative_to(record).as_posix(), sha256=sha(path), environment_index=camera.index,
                fps=clip['fps'], complete_clip=clip['complete_clip'], frame_times_seconds=clip['frame_times_seconds'],
                qpos_sha256_by_frame=camera.frames, extra_physics_steps=0, source='actual sampled GPU states')))
        state.update(actual_control_transitions=counts['controls']*n, sampling_wall_seconds=time.monotonic()-began)
