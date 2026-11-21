"""Record solved contact force at the original 4 ms physics cadence."""
from contextlib import ExitStack
from dataclasses import asdict
import time

import mujoco
import numpy as np
import torch
import yaml

import collect_roll_confirmation as official
from common_scene_contract import install_guard
from display_preferences import FILE as DISPLAY
from getup_value_bootstrap import td
from record_io import write
from roll_metrics import summarize
from roll_neck_support_collect import cpu, joint_contract, make_video
from s1_reference_inputs import ROOT, read, sha


LABELS = ('ground', 'head', 'left_foot', 'right_foot')


def contact_layout(model):
    floor = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).item()
    head_body = model.body('robot/jaw_soft').id
    trunk = model.body('robot/trunk_base').id
    assert model.body_rootid[trunk] == trunk
    kind = np.where(model.body_rootid[model.geom_bodyid] == trunk, 0, -1)
    # The whole-body sensor includes other robot geoms, while the head and
    # foot channels mirror their individual original geometry groups.
    kind[model.geom_bodyid == head_body] = 1
    for label, name in enumerate(('left_foot_collision', 'right_foot_collision'), 2):
        kind[model.geom(f'robot/{name}').id] = label
    assert kind[floor] == -1 and (kind == 1).any()
    weight = float(model.body_subtreemass[trunk]*-model.opt.gravity[2])
    assert weight > 0. and not np.asarray(model.opt.gravity[:2]).any()
    return floor, kind, weight


def reduce_contacts(pairs, worlds, normal, floor, kind, size):
    floor_first, floor_second = pairs[:, 0] == floor, pairs[:, 1] == floor
    on_floor = (floor_first ^ floor_second)
    other = np.where(floor_first[on_floor], pairs[on_floor, 1], pairs[on_floor, 0])
    assert (other >= 0).all() and (other < len(kind)).all()
    labels, world, force = kind[other], worlds[on_floor], normal[on_floor]
    valid, specialized = labels >= 0, labels > 0
    assert (world >= 0).all() and (world < size).all() and np.isfinite(force).all()
    assert (force >= -1e-5).all()
    force = np.maximum(force, 0.)
    counts, loads = np.zeros((size, len(LABELS)), np.int32), np.zeros((size, len(LABELS)))
    for result, weights in ((counts, np.ones(len(force), dtype=np.int32)), (loads, force)):
        np.add.at(result[:, 0], world[valid], weights[valid])
        np.add.at(result, (world[specialized], labels[specialized]), weights[specialized])
    return counts, loads


class ContactClock:
    def __init__(self, env, plan):
        import warp as wp

        self.env = env
        self.first, self.last = (round(second/env.physics_dt) for second in plan['capture_seconds'])
        self.maximum = plan['maximum_batched_substeps']
        self.floor, self.kind, self.weight = contact_layout(env.sim.mj_model)
        self.threshold = self.weight*plan['load_threshold_nominal_weight_fraction']
        self.slots = int(env.sim.wp_data.contact.dist.shape[0])
        self.ids = wp.array(np.arange(self.slots, dtype=np.int32), dtype=int, device=env.sim.wp_device)
        self.forces = wp.empty(self.slots, dtype=wp.spatial_vector, device=env.sim.wp_device)
        self.rows = []
        self.completed = 0

    def read_contacts(self):
        import mujoco_warp as mjwarp
        import warp as wp

        sim = self.env.sim
        raw = sim.wp_data
        count = int(raw.nacon.numpy()[0])
        assert 0 <= count < self.slots, 'Substep contact capacity reached'
        with wp.ScopedDevice(sim.wp_device):
            mjwarp.contact_force(sim.wp_model, raw, self.ids, False, self.forces)
        normal = self.forces.numpy()[:count, 0]
        if count:
            pairs = raw.contact.geom.numpy()[:count]
            worlds = raw.contact.worldid.numpy()[:count]
            counts, loads = reduce_contacts(pairs, worlds, normal, self.floor, self.kind, self.env.num_envs)
        else:
            counts = np.zeros((self.env.num_envs, len(LABELS)), np.int32)
            loads = np.zeros_like(counts, dtype=float)
        return counts, loads

    def record(self, original_step):
        n = self.env._sim_step_counter
        assert n == self.completed+1 and n <= self.maximum
        capture = self.first < n <= self.last
        if capture:
            q_before = cpu(self.env.sim.data.qpos[:, 3:7])
        original_step()
        self.completed = n
        if not capture:
            return
        # Solver forces belong to the collision configuration before
        # integration. mjlab's later control-boundary forward overwrites them.
        points, loads = self.read_contacts()
        q_after = cpu(self.env.sim.data.qpos[:, 3:7])
        self.rows.append(dict(substep=n, before=q_before, after=q_after, points=points, normal_force_n=loads))

    def save(self, folder):
        if not self.rows:
            return
        archive = folder/'contact-clock.npz'
        np.savez_compressed(archive, **{key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]})
        write(folder/'contact-clock.json', dict(first_substep=self.rows[0]['substep'],
            last_substep=self.rows[-1]['substep'], recorded_substeps=len(self.rows), labels=LABELS,
            force_threshold_n=self.threshold, nominal_weight_n=self.weight,
            complete=len(self.rows) == self.last-self.first, sha256=sha(archive)))


def collect(plan, folder, state, save, startup=False):
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends(allow_tf32=False)
    cfg = official.prepare(plan)
    display = read(DISPLAY)
    assert display['visible_terminal_required'] and not display['native_live_window']
    (folder/'env.yaml').write_text(yaml.dump(asdict(cfg)))
    write(folder/'display.json', dict(native_live_window=False, record_offscreen=not startup,
                                    preference_sha256=sha(DISPLAY)))
    base = ROOT/'records/reports/common-plant-screen/startup'
    cls = install_guard(plan, folder, base, 'rollout')
    env = cls(cfg=cfg, device='cuda:0', render_mode=None)
    clock, trace = None, []
    try:
        env.seed(plan['seed'])
        observed, _ = env.reset()
        actor, weights, home, sim_home, names = official.parity(env, observed, plan, folder)
        assert env.num_envs == plan['episodes'] and not cfg.auto_reset
        assert (env.physics_dt, env.step_dt, cfg.decimation) == (.004, .02, 5)
        assert env.sim.mj_model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        clock = ContactClock(env, plan)
        qpos, qvel = cpu(env.sim.data.qpos), cpu(env.sim.data.qvel)
        cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(env.device).clone()
        counts, forces = clock.read_contacts()
        np.testing.assert_array_equal(qpos, cpu(env.sim.data.qpos))
        np.testing.assert_array_equal(qvel, cpu(env.sim.data.qvel))
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state(env.device))
        assert env.common_step_counter == env._sim_step_counter == 0
        initial = dict(qpos=qpos, qvel=qvel, actor_obs=cpu(observed['actor']),
                       cpu_rng=cpu_rng.numpy(), gpu_rng=gpu_rng.cpu().numpy(), counts=counts, forces=forces)
        initial_path = folder/'contact-clock-initial.npz'
        np.savez_compressed(initial_path, **initial)
        if not startup:
            reference = ROOT/plan['startup_archive']
            with np.load(reference) as archive:
                assert sorted(initial) == sorted(archive.files)
                for name, values in initial.items():
                    np.testing.assert_array_equal(values, archive[name], err_msg=name)
        write(folder/'contact-clock-zero.json', dict(status='passed', physical_substeps=0,
              force_threshold_n=clock.threshold, raw_initial_points=int(counts[:, 0].sum()),
              maximum_initial_force_n=float(forces[:, 0].max()), rng_and_state_exact=True,
              archive_sha256=sha(initial_path), matched_standalone_startup=not startup))
        if startup:
            state.update(status='completed-zero-step', physical_steps=0)
            save()
            return

        robot = env.scene['robot']
        ids, limits, _ = joint_contract(env, names)
        tolerance = np.deg2rad(plan['actual_joint_limit_tolerance_deg'])
        head_local = robot.find_bodies('jaw_soft')[0][0]
        state.update(step_dt=.02, physics_dt=.004, joint_names=names, joint_limits=limits.tolist(),
              policy_home=cpu(home).tolist(), simulation_home=cpu(sim_home[0]).tolist(),
              first_episode_only=True, physical_resets_during_collection=0)
        save()
        h = round(cfg.episode_length_s/env.step_dt)
        assert h == 450
        step_physics = env.sim.step
        env.sim.step = lambda: clock.record(step_physics)

        def sensor(name, width):
            return cpu(env.scene[name].data.found).reshape(env.num_envs, width, -1).any(axis=-1)

        def sample(step, current, raw_observed, action, requested, applied, term, timeout):
            raw = env.sim.wp_data.contact
            count = int(env.sim.wp_data.nacon.numpy()[0])
            assert 0 <= count < clock.slots
            geoms = raw.geom.numpy()[:count]
            floor_first, floor_second = geoms[:, 0] == clock.floor, geoms[:, 1] == clock.floor
            safe = np.where(floor_first, geoms[:, 1], geoms[:, 0])
            head = (floor_first ^ floor_second) & np.isin(safe, np.flatnonzero(clock.kind == 1))
            depth = np.zeros(env.num_envs, np.float32)
            np.maximum.at(depth, raw.worldid.numpy()[:count][head], np.maximum(-raw.dist.numpy()[:count][head], 0.))
            pos = cpu(robot.data.joint_pos[:, ids])
            valid = ~term & (~timeout | (step == h))
            values = dict(time=step*.02, active=valid, height=cpu(robot.data.root_link_pos_w[:, 2]),
                quaternion=cpu(robot.data.root_link_quat_w), head_quaternion=cpu(robot.data.body_link_quat_w[:, head_local]),
                angular_velocity=cpu(robot.data.root_link_ang_vel_b), foot_contact=sensor('feet_ground_contact', 2),
                head_contact=sensor('head_ground_contact', 1)[:, 0], ground_contact=sensor('robot_ground_contact', 1)[:, 0],
                head_penetration_m=depth, joint_positions=pos,
                joint_envelope_ok=((pos >= limits[:, 0]-tolerance)&(pos <= limits[:, 1]+tolerance)).all(1),
                raw_observations=cpu(raw_observed), observations=cpu(current), raw_policy_actions=cpu(action),
                requested_targets=cpu(requested), applied_targets=cpu(applied), terminated=term, truncated=timeout,
                qpos=cpu(env.sim.data.qpos), qvel=cpu(env.sim.data.qvel))
            assert all(np.isfinite(value).all() for value in values.values())
            trace.append(values)

        previous = torch.zeros_like(sim_home)
        zero = np.zeros(env.num_envs, bool)
        raw_observed = observed['actor']
        sample(0, official.model_input(raw_observed, sim_home, home, previous), raw_observed,
               previous, sim_home, sim_home, zero, zero)
        started = time.monotonic()
        with ExitStack() as cleanup, torch.no_grad():
            camera, movie = make_video(env, folder, f'Official fixed roll | seed {plan["seed"]} | contact clock')
            cleanup.callback(camera.close)
            cleanup.callback(movie.close)
            movie.sample()
            for k in range(1, h+1):
                raw_observed = observed['actor'].clone()
                current = official.model_input(raw_observed, sim_home, home, previous)
                torch.testing.assert_close(current[:, 48:], torch.zeros_like(current[:, 48:]), atol=1e-6, rtol=0.)
                action = actor(td('actor', current))
                requested = home+action
                observed, reward, term, timeout, _ = env.step(requested-sim_home)
                assert torch.isfinite(reward).all()
                applied = env.action_manager.get_term('joint_pos')._processed_actions
                torch.testing.assert_close(applied, requested, atol=1e-6, rtol=0.)
                term, timeout = cpu(term), cpu(timeout)
                sample(k, current, raw_observed, action, requested, applied, term, timeout)
                if term[0] or timeout[0]:
                    movie._sample_terminal()
                else:
                    movie.sample()
                previous = action.clone()
                assert not term.any() and not (k < h and timeout.any()), 'Premature termination saved for review'
                if k % 25 == 0:
                    elapsed = time.monotonic()-started
                    state.update(last_control=k, wall_seconds=elapsed, instrumented_substeps=len(clock.rows))
                    save()
                    print('ROLL_CONTACT_CLOCK', k, '/', h, '| samples', len(clock.rows), '| wall', round(elapsed, 1), flush=True)
                    assert elapsed < plan['maximum_sampling_wall_seconds'], 'Sampling time limit'
        assert env.common_step_counter == h and env._sim_step_counter == plan['maximum_batched_substeps']
        assert len(clock.rows) == clock.last-clock.first
        arrays = {key: np.asarray([row[key] for row in trace]) for key in trace[0]}
        arrays.update(initial_qpos=qpos, initial_qvel=qvel)
        path = folder/'forward-roll-trace.npz'
        np.savez_compressed(path, **arrays)
        assert len(movie.clips) == 1 and movie.clips[0]['complete_clip']
        clip = movie.clips[0]
        frames = read(folder/'recording-states.json')
        assert [row['common_step'] for row in frames] == clip['common_steps']
        for key, value in weights.items():
            torch.testing.assert_close(value, actor.state_dict()[key].cpu(), atol=0., rtol=0.)
        state.update(conditions=[summarize(arrays, plan['criteria'])], trace_sha256=sha(path),
            weights_and_normalization_unchanged=True, actual_control_transitions=env.num_envs*h,
            video=dict(path=clip['file'], sha256=sha(folder/clip['file']), fps=25, environment_index=0,
                       frame_steps=clip['common_steps'], complete_horizon=True,
                       qpos_sha256_by_frame=[row['qpos_sha256'] for row in frames], original_camera=True))
        save()
    except BaseException:
        if trace:
            partial = folder/'stopped-partial-trace.npz'
            np.savez_compressed(partial, **{key: np.asarray([row[key] for row in trace]) for key in trace[0]})
            state.update(partial_trace_sha256=sha(partial), last_saved_control=len(trace)-1, partial_recording=True)
            save()
        raise
    finally:
        try:
            if clock is not None:
                state.update(completed_physical_substeps=clock.completed,
                             attempted_physical_substeps=env._sim_step_counter)
                save()
                clock.save(folder)
        finally:
            env.close()
