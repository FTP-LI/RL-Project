"""Collect the three fixed-policy roll arms from a common live prefix."""
from contextlib import ExitStack
from types import SimpleNamespace
import time

import numpy as np
import torch

from s1_reference_inputs import ROOT, sha
from action_video import ActionVideo
from collect_roll_confirmation import frozen_state, model_input, td
from getup_anchor_runtime import RecordingSnapshot
from getup_release_fork_state import ReleaseSnapshot, clone, digest, signature, tensor_fields, warp_slots
from getup_release_scene_state import classify_scene
from record_io import write
from roll_metrics import summarize
from roll_neck_support import NeckSupport


def cpu(value):
    return value.detach().cpu().numpy().copy()


class RollSnapshot(ReleaseSnapshot):
    def __init__(self, env, limit):
        import warp as wp

        self.env, self.device = env, env.device
        self.cuda = torch.device(self.device).type == 'cuda'
        self.slots = list(warp_slots(env.sim.wp_data, 'warp'))
        self.generators = {}

        def bind(label, owner, keys):
            self.slots.extend((f'{label}.{key}', owner, key) for key in sorted(set(keys)) if hasattr(owner, key))

        def ring(label, value):
            bind(label, value, ('_pointer', '_buffer', '_num_pushes'))

        def delay(label, value):
            bind(label, value, ('_current_lags', '_step_count', '_phase_offsets'))
            ring(label + '.ring', value._buffer)
            if value.generator is not None:
                self.generators[label] = value.generator

        # Roll has no actor observation delays. Retain any actual history and
        # critic delays without inheriting getup's actor-delay requirement.
        obs = env.observation_manager
        assert not obs._group_obs_term_delay_buffer.get('actor') and obs._obs_buffer is env.obs_buf
        assert not any(obs._group_obs_class_term_cfgs.values())
        assert set(env.event_manager.available_modes) <= {'reset', 'startup'}
        assert not env.cfg.recorders and not env.cfg.curriculum and not env.cfg.sim.nan_guard.enabled
        assert not env.reward_manager._class_term_cfgs and not env.metrics_manager._class_term_cfgs
        for group, terms in obs._group_obs_term_delay_buffer.items():
            for key, buffer in terms.items():
                delay(f'obs-delay.{group}.{key}', buffer)
        for group, terms in obs._group_obs_term_history_buffer.items():
            for key, buffer in terms.items():
                ring(f'obs-history.{group}.{key}', buffer)

        assert {'dof_frictionloss', 'dof_damping'} <= env.sim.expanded_fields
        bind('model', env.sim.wp_model, env.sim.expanded_fields)
        # Include the roll reward's scalar cache tick and tensor accumulators.
        self.roll_fields = {key for key in vars(env) if key.startswith('_roulade_')}
        assert '_roulade_last_update_step' in self.roll_fields
        bind('env', env, self.roll_fields | set(tensor_fields(env)) |
             {'common_step_counter', '_sim_step_counter', 'obs_buf', 'extras'})
        bind('manager.action', env.action_manager, ('_action', '_prev_action', '_prev_prev_action'))
        term = env.action_manager.get_term('joint_pos')
        bind('processor', term, tensor_fields(term))
        for name, command in env.command_manager._commands.items():
            bind(f'command.{name}', command, (*tensor_fields(command), 'metrics'))
        bind('manager.reward', env.reward_manager, ('_episode_sums', '_reward_buf', '_step_reward'))
        bind('manager.done', env.termination_manager, ('_term_dones', '_terminated_buf', '_truncated_buf'))
        bind('manager.metrics', env.metrics_manager, ('_episode_sums', '_substep_accum', '_substep_episode_sums',
             '_substep_count', '_step_count', '_step_values'))
        terrain, regular, builtin = classify_scene(env.scene)
        bind('terrain', terrain, tensor_fields(terrain))
        bind('terrain.data', terrain.data, tensor_fields(terrain.data))
        robot = env.scene['robot']
        bind('robot', robot.data, tensor_fields(robot.data))
        for i, actuator in enumerate(robot.actuators):
            assert type(actuator).__name__ == 'FrictionDRBamActuator'
            label = f'actuator.{i}'
            bind(label, actuator, (*tensor_fields(actuator), '_friction_fields_checked'))
            delay(label + '.delay', actuator._delay_buffer)
            inner = actuator._bam_model.actuator
            bind(label + '.bam', inner, (*tensor_fields(inner), 'q_target_smooth', 'vin', 'kp'))
        for name, sensor in {**regular, **builtin}.items():
            bind(f'sensor.{name}', sensor, (*tensor_fields(sensor), '_cache_valid', '_cached_data',
                 '_air_time_state', '_history_state'))
        assert len({label for label, _, _ in self.slots}) == len(self.slots)
        self.synchronize()
        values = [getattr(owner, key) for _, owner, key in self.slots]
        self.warp_bytes = sum(value.capacity for value in values if isinstance(value, wp.array))
        assert self.warp_bytes <= limit, self.warp_bytes
        self.values = [(wp.clone(value) if value.size else value) if isinstance(value, wp.array) else clone(value)
                       for value in values]
        self.synchronize()
        self.rng = self.rng_state()
        self.expected = self.signatures(self.values, self.rng)

    def verify(self):
        assert self.roll_fields == {key for key in vars(self.env) if key.startswith('_roulade_')}
        return super().verify()


def joint_contract(env, names):
    model = env.sim.mj_model
    ids, order = env.scene['robot'].find_joints(names, preserve_order=True)
    assert order == names
    joints = [model.joint('robot/' + name).id for name in names]
    assert all(model.jnt_limited[joints])
    return ids, model.jnt_range[joints].copy(), model.jnt_dofadr[joints].copy()


class MotorReadback:
    """Record motor values immediately before the original physics call."""
    def __init__(self, env):
        self.env = env
        (self.actuator,) = env.scene['robot'].actuators
        self.inner = self.actuator._bam_model.actuator
        self.column = self.actuator._target_names.index('neck_pitch')
        self.ctrl = int(self.actuator._global_ctrl_ids[self.column])
        self.original = self.actuator.apply_delay
        self.actuator.apply_delay = self.capture_delay

    def capture_delay(self, command):
        delayed = self.original(command)
        self.delayed = delayed.position_target[:, self.column]
        return delayed

    def capture(self):
        return torch.stack((self.delayed, self.inner.q_target_smooth[:, self.column],
                            self.inner.vin[:, 0], self.actuator._prev_motor_torque[:, self.column],
                            self.env.sim.data.ctrl[:, self.ctrl]), dim=-1)

    def close(self):
        del self.actuator.apply_delay


def probe_motor(env, names, limits, maximum_warp_bytes):
    """Exercise the real processor, delay and BAM drive without integrating."""
    neck = names.index('neck_pitch')
    snap = RollSnapshot(env, maximum_warp_bytes)
    zero = snap.verify()
    assert zero['counters'] == [0, 0]
    manager = env.action_manager
    term = manager.get_term('joint_pos')
    base = torch.as_tensor(term._offset, device=env.device).expand(env.num_envs, -1).clone()
    higher = base.clone()
    higher[0, neck] += .08
    assert limits[neck, 0] < float(base[0, neck]) < float(higher[0, neck]) < limits[neck, 1]
    readback = MotorReadback(env)

    def probe(target):
        rows = []
        for request in (base, target, target):
            manager.process_action(request - base)
            manager.apply_action()
            env.scene.write_data_to_sim()
            rows.append(readback.capture())
        result = cpu(torch.stack(rows))
        # Reset-time rewards create the same caches on all probe paths.
        assert snap.restore() == zero
        return result

    try:
        reference, candidate = probe(base), probe(higher)
    finally:
        readback.close()
    np.testing.assert_array_equal(reference[:2], candidate[:2])
    np.testing.assert_array_equal(reference[:, 1:], candidate[:, 1:])
    np.testing.assert_allclose(candidate[2, 0, 0] - reference[2, 0, 0], .08, atol=1e-7, rtol=0)
    assert candidate[2, 0, 2] == reference[2, 0, 2]
    assert candidate[2, 0, 3] > reference[2, 0, 3]
    return dict(status='passed', physical_substeps=0, motor_writes=6, snapshot=zero,
                first_modified_write_after_request=2,
                neck_torque_delta_nm=float(candidate[2, 0, 3] - reference[2, 0, 3]))


def make_video(env, record, role):
    snapshot = RecordingSnapshot(env, record)
    snapshot.cam.distance, snapshot.cam.azimuth, snapshot.cam.elevation = .8, 90, -18
    video = ActionVideo(snapshot, 'roll', record)
    assert video.horizon == 450 and video.stride == 2
    original = video.media.VideoWriter

    def fixed_writer(*args, **kwargs):
        from PIL import Image, ImageDraw

        writer = original(*args, **kwargs)
        append = writer.add_image

        def frame(pixels):
            image = Image.fromarray(pixels)
            draw = ImageDraw.Draw(image)
            t = env.common_step_counter * env.step_dt
            draw.rectangle((0, 0, 640, 46), fill=(17, 30, 41))
            draw.text((8, 5), f'Fixed roll evaluation | env 0 | t={t:.2f}s', fill='white')
            draw.text((8, 26), role, fill='white')
            append(np.asarray(image))

        writer.add_image = frame
        return writer

    # Override this recorder's writer only. Archived recorders stay pinned.
    video.media = SimpleNamespace(VideoWriter=fixed_writer)
    return snapshot, video


def collect(env, actor, weights, home, sim_home, names, spec, record, state, publish, available):
    p, roles = spec['environment'], spec['roles']
    robot = env.scene['robot']
    n, h, fork = env.num_envs, round(env.cfg.episode_length_s / env.step_dt), spec['warmup_controls']
    ids, limits, dofs = joint_contract(env, names)
    neck = names.index('neck_pitch')
    model = env.sim.mj_model
    head_local = robot.find_bodies('jaw_soft')[0][0]
    head_geom = model.geom('robot/head_collision').id
    head_geoms = np.flatnonzero(model.geom_bodyid == model.geom_bodyid[head_geom])
    import mujoco

    floor = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    tolerance = np.deg2rad(p['actual_joint_limit_tolerance_deg'])
    term = env.action_manager.get_term('joint_pos')
    controller = NeckSupport(n, spec['intervention'], neck, limits[neck])
    previous = torch.zeros_like(sim_home)
    active = np.ones(n, bool)
    traces = {name: [] for name in ('shared-prefix', *roles)}
    motors, cameras, movies, receipts = {}, {}, {}, {}
    readback = MotorReadback(env)
    old_step = env.sim.step
    substeps = []
    actual = state['actual']
    begin = time.monotonic()
    role = 'shared-prefix'
    state.update(joint_names=names, joint_limits=limits.tolist(), neck_dof=int(dofs[neck]),
                 home=cpu(home).tolist(), simulation_home=cpu(sim_home[0]).tolist(),
                 motor_columns=['delayed_target', 'smoothed_target', 'voltage', 'motor_torque', 'sim_ctrl'])

    def step_physics():
        assert actual['substeps'] < spec['maximum_physics_substeps']
        substeps.append(readback.capture())
        old_step()
        actual['substeps'] += 1

    def sample(k, **extra):
        contact = env.sim.wp_data.contact
        count = int(env.sim.wp_data.nacon.numpy()[0])
        assert 0 <= count < contact.dist.shape[0]
        geoms = contact.geom.numpy()[:count]
        hits = ((np.isin(geoms[:, 0], head_geoms) & np.isin(geoms[:, 1], floor))
                | (np.isin(geoms[:, 1], head_geoms) & np.isin(geoms[:, 0], floor)))
        depth = np.zeros(n, np.float32)
        np.maximum.at(depth, contact.worldid.numpy()[:count][hits],
                      np.maximum(-contact.dist.numpy()[:count][hits], 0))
        pos = cpu(robot.data.joint_pos[:, ids])

        def sensor(name, width):
            return cpu(env.scene[name].data.found).reshape(n, width, -1).any(axis=-1)

        row = dict(time=k * .02, active=active.copy(), height=cpu(robot.data.root_link_pos_w[:, 2]),
            quaternion=cpu(robot.data.root_link_quat_w), head_quaternion=cpu(robot.data.body_link_quat_w[:, head_local]),
            angular_velocity=cpu(robot.data.root_link_ang_vel_b), joint_positions=pos,
            qpos=cpu(env.sim.data.qpos), qvel=cpu(env.sim.data.qvel),
            foot_contact=sensor('feet_ground_contact', 2), head_contact=sensor('head_ground_contact', 1)[:, 0],
            ground_contact=sensor('robot_ground_contact', 1)[:, 0], head_penetration_m=depth,
            joint_envelope_ok=((pos >= limits[:, 0]-tolerance) & (pos <= limits[:, 1]+tolerance)).all(axis=1),
            **extra)
        assert all(np.isfinite(value).all() for value in row.values())
        traces[role].append(row)
        return row

    initial_obs = env.obs_buf['actor']
    raw_zero, initial_home = cpu(previous), cpu(sim_home)
    initial = sample(0, raw_observations=cpu(initial_obs), observations=cpu(model_input(initial_obs, sim_home, home, previous)),
        raw_policy_actions=raw_zero, requested_targets=initial_home, applied_targets=initial_home,
        neck_cue=np.zeros(n, bool), neck_phase_deg=np.zeros(n), terminated=np.zeros(n, bool),
        truncated=np.zeros(n, bool), motor=np.zeros((5, n, 5), np.float32))

    def advance(k, before, filmed):
        assert k == env.common_step_counter + 1 and actual['controls'] < spec['maximum_batched_controls']
        raw_observed = env.obs_buf['actor']
        observed = model_input(raw_observed, sim_home, home, previous)
        raw = actor(td('actor', observed))
        raw_cpu, raw_observed_cpu, observations = cpu(raw), cpu(raw_observed), cpu(observed)
        requested = cpu(home + raw)
        targets, cue, phase = controller.choose(before['quaternion'], before['foot_contact'], before['head_contact'],
            before['angular_velocity'][:, 1], before['qvel'][:, dofs[neck]], active, requested, role == roles[-1])
        assert not cue.any() or k > fork, 'A causal cue became eligible during the shared prefix'
        substeps.clear()
        sent = torch.from_numpy(targets).to(env.device) - sim_home
        _, reward, terminated, truncated, _ = env.step(sent)
        actual['controls'] += 1
        assert len(substeps) == env.cfg.decimation and torch.isfinite(reward).all()
        term_done, time_done = cpu(terminated), cpu(truncated)
        active[:] &= ~term_done & (~time_done | (k == h))
        applied = cpu(term._processed_actions)
        np.testing.assert_allclose(applied, targets, atol=1e-6, rtol=0)
        row = sample(k, raw_observations=raw_observed_cpu, observations=observations, raw_policy_actions=raw_cpu,
            requested_targets=requested, applied_targets=applied, neck_cue=cue, neck_phase_deg=phase,
            terminated=term_done, truncated=time_done, motor=cpu(torch.stack(substeps)))
        previous.copy_(raw)
        for name in filmed:
            if term_done[0] or time_done[0]:
                movies[name]._sample_terminal()
            else:
                movies[name].sample()
        assert not term_done.any() and not (time_done.any() and k < h), 'Premature termination saved for review'
        if k % 25 == 0:
            available()
            assert time.monotonic() - begin <= spec['maximum_sampling_wall_seconds'], 'Sampling time limit'
        if k % 50 == 0 or k == fork:
            state.update(branch=role, control=k, first_episodes=int(active.sum()))
            publish()
            print('ROLL_NECK', role, k, '/', h, 'actual controls', actual['controls'], flush=True)
        return row

    env.sim.step = step_physics
    try:
        with ExitStack() as stack, torch.no_grad():
            for name in roles:
                directory = record / f'evaluation-{name}'
                directory.mkdir()
                camera, video = make_video(env, directory, name)
                cameras[name], movies[name] = camera, video
                stack.callback(camera.close)
                stack.callback(video.close)
                video.sample()
            before = initial
            for k in range(1, fork+1):
                before = advance(k, before, roles)
            at_fork = before
            physical = RollSnapshot(env, spec['maximum_warp_snapshot_bytes'])
            saved_previous, saved_cue = previous.clone(), controller.snapshot()
            cue_digest = digest(signature((saved_previous, saved_cue)))
            receipt = physical.verify()
            assert receipt['counters'] == [fork, fork*5] and active.all()
            write(record / 'fork-fields.json', physical.expected)
            for role in roles:
                restored = physical.verify() if role == roles[0] else physical.restore()
                previous.copy_(saved_previous)
                controller.restore(saved_cue)
                active[:] = True
                assert restored == receipt and digest(signature((previous, controller.snapshot()))) == cue_digest
                receipts[role] = dict(physical=restored, history_and_cue_sha256=cue_digest)
                write(record / 'fork-readback.json', receipts)
                print('ROLL_NECK_FORK', role, 'control', fork, 'fields', restored['fields'], flush=True)
                before = at_fork
                for k in range(fork+1, h+1):
                    before = advance(k, before, (role,))
            assert all(torch.equal(value, weights[key]) for key, value in frozen_state(actor).items())
            state['actor_unchanged'] = True
    finally:
        env.sim.step = old_step
        readback.close()
        for name, rows in traces.items():
            if rows:
                np.savez_compressed(record / f'trace-{name}.npz',
                                    **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
        for name, movie in movies.items():
            if not movie.clips:
                continue
            clip = movie.clips[0]
            path = movie.record / clip['file']
            state['conditions'].append(dict(condition='forward-roll', role=name, video=dict(
                path=path.relative_to(record).as_posix(), sha256=sha(path), environment_index=0,
                fps=clip['fps'], complete_clip=clip['complete_clip'], frame_times_seconds=clip['frame_times_seconds'],
                qpos_sha256_by_frame=[row['qpos_sha256'] for row in cameras[name].recorded_states],
                extra_physics_steps=0, source='actual fixed-evaluation states')))

    for name in roles:
        rows = traces['shared-prefix'] + traces[name]
        arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        write(record / f'original-score-{name}.json', summarize(arrays, p['criteria']))
