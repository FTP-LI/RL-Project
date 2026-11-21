"""Collect two complete forward routes from the same live control boundary."""

from time import monotonic
from unittest.mock import patch

import numpy as np

import getup_release_scene_state as scene_snapshot
from collect_forward_hip_yaw_pulse import arrays
from forward_right_support_continuation import CONTINUATION_FIELDS, RightSupportContinuation
from right_hip_pitch_duration_control import PHASE_FIELDS, SCHEDULE_FIELDS
from roll_entry_snapshot import roll_slots
from roll_stand_walk_sequence import FORWARD
from run_roll_stand_forward_diagnostic import StandForward, require
from s1_reference_inputs import sha


def take_snapshot(env, policy, controller, phase, maximum_bytes):
    def slots(view, active_policy):
        fields, generators = roll_slots(view, active_policy)
        for prefix, owner, names in (('onset', controller.onset, SCHEDULE_FIELDS),
                                     ('continuation', controller, CONTINUATION_FIELDS),
                                     ('phase', phase, PHASE_FIELDS)):
            fields.extend((f'{prefix}.{name}', owner, name) for name in names)
        return fields, generators

    with patch.object(scene_snapshot, 'state_slots', slots):
        return scene_snapshot.SceneReleaseSnapshot(env, policy, maximum_bytes)


def save_trace(folder, states, commands):
    hashes = {}
    for name, rows in (('trace.npz', states), ('commands.npz', commands)):
        if rows:
            path = folder/name
            np.savez_compressed(path, **arrays(rows))
            hashes[name] = sha(path)
    return hashes


def make_collector(base, may_continue):
    def collect(env, obs, plan, folder, endpoint, networks, homes, sim_home, limits):
        design, worlds = plan['forward_pair'], plan['worlds']
        shared, horizon = design['shared_prefix_controls'], design['standing_controls']+design['forward_controls']
        maximum, cap = plan['maximum_controls'], plan['maximum_warp_snapshot_bytes']
        robot = env.scene['robot']
        joints, names = robot.find_joints(plan['walking']['joint_names'], preserve_order=True)
        feet, sites = robot.find_sites(('left_foot', 'right_foot'), preserve_order=True)
        require(names == plan['walking']['joint_names'] and list(sites) == ['left_foot', 'right_foot'],
                'Joint or site order changed')
        head = robot.find_bodies('jaw_soft')[0][0]
        tolerance = np.deg2rad(plan['roll']['actual_joint_limit_tolerance_deg'])
        bounds, alive = base.cpu(limits), np.ones(worlds, dtype=bool)
        term = env.action_manager.get_term('joint_pos')
        phase = StandForward(plan)
        policy = base.PolicyControl(plan, networks, homes, sim_home, limits)
        controller = RightSupportContinuation(plan, plan['duration_probe'])
        started = sample_time = monotonic()
        sample_control = 0
        role = 'shared'
        cameras, results = {}, {}
        view = dict(status='running', run_id=plan['run_id'], controls=0, maximum_controls=maximum,
                    transitions=0, learning_updates=0, updates=[])

        def guard():
            require(may_continue() and monotonic()-started < plan['maximum_sampling_wall_seconds'],
                    'Observation, memory, pause or time guard stopped the forward pair')

        def capture(k):
            state = base.capture(env, endpoint, alive, joints, head, feet, bounds, tolerance)
            require(all(np.isfinite(value).all() for value in state.values()) and state['joint_envelope_ok'].all()
                    and (state['head_penetration_m'] <= plan['roll']['criteria']['head_penetration_max_m']).all(),
                    'Nonfinite state, joint excursion or head penetration')
            phase.update(k, state)
            return dict(state, phase_next=phase.phase.copy())

        def advance(k, states, commands):
            guard()
            require(env.guard_steps < maximum and env.common_step_counter == k, 'Unexpected control boundary')
            observed = env.obs_buf['actor'].clone()
            motor, row = policy.next(k+1, observed, phase)
            actor_observation = base.cpu(observed)
            expected, detail = controller.next(k, row, actor_observation, role == 'right-support-continuation')
            if detail['hip_pitch_pulse_side'].any() or detail['right_extension_applied'].any():
                motor = base.torch.as_tensor(expected, device=env.device)
            _, _, done, timeout, _ = env.step(motor-sim_home)
            base.torch.testing.assert_close(term._processed_actions, motor, rtol=0, atol=1e-6)
            alive[:] &= ~base.cpu(done | timeout)
            commands.append(dict(row, **detail, observed=actor_observation, terminated=base.cpu(done),
                                 truncated=base.cpu(timeout), actual_motor_target=base.cpu(term._processed_actions)))
            endpoint.capture()
            states.append(capture(k+1))
            require(not endpoint.first_mismatches and alive.all(), 'Contact readback differed or a world terminated')

        guard()
        initial = take_snapshot(env, policy, controller, phase, cap)
        zero = initial.verify()
        fields = sorted(initial.expected)
        require(zero['counters'] == [0, 0] and initial.restore() == zero
                and 'env.obs_buf' in fields and any('.delay.' in name for name in fields),
                'Actual CUDA zero readback lacks observation or actuator delay state')
        np.testing.assert_array_equal(base.cpu(env.obs_buf['actor']), base.cpu(obs['actor']))
        base.write(folder/'zero-readback.json', dict(initial=zero, fields=fields))
        del initial

        camera = base.Snapshot(env, folder)
        render = camera.render
        last_render, pixels = None, None

        def render_boundary():
            # Encode the shared prefix twice from the same rendered frame.
            nonlocal last_render, pixels
            key = (role, env.common_step_counter)
            if key != last_render:
                pixels, last_render = render(), key
            return pixels

        camera.render = render_boundary

        def publish(label, frame=False):
            nonlocal sample_control, sample_time
            count, now = int(env.guard_steps), monotonic()
            if count > sample_control:
                rate = (count-sample_control)*worlds/max(now-sample_time, .001)
                view['updates'].append(dict(control=count, transitions_per_second=rate))
                sample_control, sample_time = count, now
            if frame:
                original = camera.capture()['file']
                name = f'frame-{role}-{env.common_step_counter}.jpg'
                (folder/original).replace(folder/name)
                view['frame'] = name
            view.update(purpose=label, controls=count, transitions=count*worlds,
                        updated_at=base.datetime.now(base.TZ).isoformat())
            base.live.publish(folder, view)
            print('FORWARD_SUPPORT_PAIR', plan['run_id'], role, count, maximum, env.common_step_counter, flush=True)

        states, commands = [], []
        prefix_path = folder/'shared'
        prefix_path.mkdir()
        try:
            for arm in design['arm_order']:
                branch = folder/'clips'/arm
                branch.mkdir(parents=True)
                cameras[arm] = base.ActionVideo(camera, 'switch', branch, horizon*.02)
                cameras[arm].sample()
            states.append(capture(0))
            publish('共同站立与前进', frame=True)
            try:
                for k in range(shared):
                    advance(k, states, commands)
                    for recorder in cameras.values():
                        recorder.sample()
                    if (k+1) % 25 == 0:
                        publish('共同站立与前进', frame=(k+1 == shared))
            finally:
                prefix_hashes = save_trace(prefix_path, states, commands)
            require((phase.phase == FORWARD).all() and (phase.forward_at == design['standing_controls']).all(),
                    'A world did not finish the original shared standing gate')
            prefix_contacts = list(endpoint.rows)
            fork = take_snapshot(env, policy, controller, phase, cap)
            captured = fork.verify()
            require(captured['counters'] == [shared, 5*shared], 'Wrong physical fork boundary')
            base.write(folder/'fork.json', dict(physical=captured, fields=sorted(fork.expected)))
            labels = dict(zip(design['arm_order'], ('仅首段脉冲', '追加右脚支撑中段修正'), strict=True))
            for role in design['arm_order']:
                guard()
                restored = fork.restore()
                require(restored == captured, 'Could not restore the exact live state')
                branch = folder/'clips'/role
                base.write(branch/'restored-fork.json', restored)
                endpoint.rows, endpoint.folder, endpoint.first_mismatches = list(prefix_contacts), branch, {}
                current = base.capture(env, endpoint, alive, joints, head, feet, bounds, tolerance)
                for key, value in current.items():
                    np.testing.assert_array_equal(value, states[-1][key], err_msg=key)
                suffix, controls = [dict(current, phase_next=phase.phase.copy())], []
                publish(labels[role], frame=True)
                try:
                    for k in range(shared, horizon):
                        advance(k, suffix, controls)
                        cameras[role].sample()
                        if (k+1) % 25 == 0:
                            publish(labels[role], frame=((k+1-shared) % 100 == 0))
                finally:
                    branch_hashes = save_trace(branch, suffix, controls)
                    endpoint.save(horizon+1)
                clip, = cameras[role].clips
                require(clip['complete_clip'] and clip['full_action'], 'Incomplete forward route video')
                relative = f'clips/{role}/{clip["file"]}'
                results[role] = dict(controls=len(controls), trace_sha256=branch_hashes,
                                     contact_sha256=base.sha(branch/'endpoint-contact.npz'),
                                     video=dict(clip, path=relative, sha256=base.sha(folder/relative)))
                view['latest_clip'] = relative
                publish(labels[role])
            require(env.guard_steps == maximum and env.common_step_counter == horizon and len(env.guard_resets) == 1,
                    'Frozen physical budget or reset count did not match')
            base.write(folder/'guarded-evaluation.json', dict(status='collected-review-required',
                protocol_sha256=base.sha(base.PLAN), actual_control_steps=maximum, actual_physics_substeps=5*maximum,
                recorded_control_transitions=maximum*worlds, physical_resets=1, shared_controls=shared,
                prefix_sha256=prefix_hashes, branches=results, learning_updates=0, heldout_used=False, hardware_io=False))
            publish('同快照前进对照采样完成')
        finally:
            for recorder in reversed(list(cameras.values())):
                recorder.close()
            camera.close()
    return collect
