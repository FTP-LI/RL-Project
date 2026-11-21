"""Branch a live right-support interval without changing the shared route."""

from time import monotonic
from unittest.mock import patch

import numpy as np

import getup_release_scene_state as scene_snapshot
from collect_forward_hip_yaw_pulse import arrays
from forward_phase_hip_pitch import HipPitchSchedule
from getup_release_fork_state import clone
from right_hip_pitch_duration_control import PHASE_FIELDS, SCHEDULE_FIELDS, schedule_state, targets
from roll_entry_snapshot import roll_slots
from roll_stand_walk_sequence import FORWARD
from run_roll_stand_forward_diagnostic import StandForward, require


def take_snapshot(env, policy, schedule, phase, maximum_bytes):
    def slots(view, controller):
        values, generators = roll_slots(view, controller)
        values.extend((f'pitch_schedule.{name}', schedule, name) for name in SCHEDULE_FIELDS)
        values.extend((f'stand_forward.{name}', phase, name) for name in PHASE_FIELDS)
        return values, generators

    with patch.object(scene_snapshot, 'state_slots', slots):
        return scene_snapshot.SceneReleaseSnapshot(env, policy, maximum_bytes)


def make_collector(base, may_continue):
    def collect(env, obs, plan, folder, endpoint, networks, homes, sim_home, limits):
        cpu, torch = base.cpu, base.torch
        design, controls, worlds = plan['duration_probe'], plan['pulse']['response_controls'], plan['worlds']
        joint_names = plan['walking']['joint_names']
        robot = env.scene['robot']
        joints, names = robot.find_joints(joint_names, preserve_order=True)
        feet, sites = robot.find_sites(('left_foot', 'right_foot'), preserve_order=True)
        require(names == joint_names and list(sites) == ['left_foot', 'right_foot'], 'Joint or site order changed')
        head = robot.find_bodies('jaw_soft')[0][0]
        tolerance = np.deg2rad(plan['roll']['actual_joint_limit_tolerance_deg'])
        bounds, alive = cpu(limits), np.ones(worlds, dtype=bool)
        term = env.action_manager.get_term('joint_pos')
        phase = StandForward(plan)
        policy = base.PolicyControl(plan, networks, homes, sim_home, limits)
        schedule = HipPitchSchedule(plan, plan['phase_hip_pitch'])
        snapshot_cap = plan['pulse']['maximum_warp_snapshot_bytes']
        start = monotonic()
        sample_time, sample_step = start, 0
        view = dict(status='running', run_id=plan['run_id'], controls=0, maximum_controls=plan['maximum_controls'],
                    transitions=0, learning_updates=0, updates=[])

        def guard():
            require(may_continue() and monotonic()-start < plan['maximum_sampling_wall_seconds'],
                    'Observation, memory, pause or time guard stopped the duration comparison')

        def capture():
            state = base.capture(env, endpoint, alive, joints, head, feet, bounds, tolerance)
            require(all(np.isfinite(item).all() for item in state.values()) and state['joint_envelope_ok'].all()
                    and (state['head_penetration_m'] <= plan['roll']['criteria']['head_penetration_max_m']).all(),
                    'The original physical guard detected nonfinite data, joint excursion or head penetration')
            return state

        def prepare(extending=None):
            observation = env.obs_buf['actor'].clone()
            k = env.common_step_counter
            motor, command = policy.next(k+1, observation, phase)
            observed = cpu(observation)
            expected, detail = targets(schedule, design, k, command, observed, extending)
            if detail['hip_pitch_pulse_side'].any() or detail['right_extension_applied'].any():
                motor = torch.as_tensor(expected, device=env.device)
            return motor, dict(command, **detail, observed=observed)

        def advance(prepared, substeps=None):
            guard()
            require(env.guard_steps < plan['maximum_controls'], 'The frozen control budget was exhausted')
            motor, command = prepared
            if substeps is None:
                _, _, done, timeout, _ = env.step(motor-sim_home)
            else:
                existing_step = env.sim.step

                def record_motor(*args, **kwargs):
                    substeps.append(cpu(env.sim.data.ctrl))
                    return existing_step(*args, **kwargs)

                with patch.object(env.sim, 'step', record_motor):
                    _, _, done, timeout, _ = env.step(motor-sim_home)
            require(not bool((done | timeout).any()), 'A world terminated inside the duration comparison')
            torch.testing.assert_close(term._processed_actions, motor, rtol=0, atol=1e-6)
            endpoint.capture()
            require(not endpoint.first_mismatches, 'Contact data differs from its physical constraint array')
            return dict(command, actual_motor_target=cpu(term._processed_actions))

        guard()
        initial = take_snapshot(env, policy, schedule, phase, snapshot_cap)
        zero = initial.verify()
        require(zero['counters'] == [0, 0] and initial.restore() == zero, 'Actual CUDA zero readback differs')
        fields = sorted(initial.expected)
        delays = [name for name in fields if '.delay.' in name]
        require(delays and 'env.obs_buf' in fields, 'The CUDA snapshot lacks observation or actuator delays')
        np.testing.assert_array_equal(cpu(env.obs_buf['actor']), cpu(obs['actor']))
        base.write(folder/'pulse-zero.json', dict(initial=zero, delay_fields=delays, observation_shape=list(obs['actor'].shape),
                   schedule_fields=[f'pitch_schedule.{name}' for name in SCHEDULE_FIELDS if f'pitch_schedule.{name}' in fields],
                   phase_fields=[f'stand_forward.{name}' for name in PHASE_FIELDS if f'stand_forward.{name}' in fields]))
        del initial

        camera = base.Snapshot(env, folder)
        video = base.ActionVideo(camera, 'switch', folder, plan['pulse']['shared_controls']*.02)

        def publish(label, refresh=False):
            nonlocal sample_time, sample_step
            k, now = int(env.guard_steps), monotonic()
            if k > sample_step:
                rate = (k-sample_step)*worlds/max(now-sample_time, .001)
                view['updates'].append(dict(control=k, transitions_per_second=rate))
                sample_step, sample_time = k, now
            if refresh:
                view['frame'] = camera.capture()['file']
            if video.clips and video.clips[0]['complete_clip']:
                view['latest_clip'] = video.clips[0]['file']
            view.update(controls=k, transitions=k*worlds, purpose=label, updated_at=base.datetime.now(base.TZ).isoformat())
            base.live.publish(folder, view)
            print('RIGHT_PITCH_DURATION', k, plan['maximum_controls'], label, flush=True)

        def compare(k, selected, current):
            snap = take_snapshot(env, policy, schedule, phase, snapshot_cap)
            captured = snap.verify()
            require(captured['counters'] == [k, 5*k], 'The branch counters differ')
            reference_contacts, restored, hashes = list(endpoint.rows), {}, {}
            branch_dir = folder/'right'
            branch_dir.mkdir()
            for role in design['roles']:
                guard()
                restored[role] = snap.restore()
                require(restored[role] == captured, 'The branch reverted to a different state')
                initial_schedule = schedule_state(schedule)
                endpoint.rows = list(reference_contacts)
                state = capture()
                for name, value in current.items():
                    np.testing.assert_array_equal(state[name], value, err_msg=name)
                states, commands, substeps = [state], [], []
                initial_motor = cpu(env.sim.data.ctrl)
                extending = selected if role in ('one-extra-control', 'continue-through-eight') else None
                for offset in range(controls):
                    row = advance(prepare(extending), substeps)
                    if offset == 0 and extending is not None:
                        np.testing.assert_array_equal(row['right_extension_applied'], selected)
                    # Once support or headroom is lost for a world, the longer
                    # arm stays off for the remainder of this test interval.
                    extending = row['right_extension_applied'] if role == 'continue-through-eight' else None
                    commands.append(row)
                    states.append(capture())
                require(len(substeps) == 5*controls, 'The motor trace is incomplete')
                path = branch_dir/f'{role}.npz'
                np.savez_compressed(path, initial_motor=initial_motor, motor_substeps=np.stack(substeps),
                    **{f'initial_schedule_{name}': value for name, value in initial_schedule.items()},
                    **{f'state_{name}': value for name, value in arrays(states).items()},
                    **{f'command_{name}': value for name, value in arrays(commands).items()})
                hashes[role] = base.sha(path)
                publish(f'右脚支撑分支 {role}')
            require(snap.restore() == captured, 'The common simulation could not be restored')
            endpoint.rows = reference_contacts
            return dict(control=k, selected_worlds=np.flatnonzero(selected).tolist(), captured=captured,
                        restored=restored, branches=hashes)

        try:
            current = capture()
            phase.update(0, current)
            prefix, commands, fork = [current], [], None
            right_age = np.zeros(worlds, dtype=np.int32)
            last_right_pulse = np.full(worlds, -1, dtype=np.int32)
            first_age, last_age = design['recognized_right_support_age_controls']
            video.sample()
            publish('右髋俯仰 共同站立与前进', refresh=True)
            for k in range(plan['pulse']['shared_controls']):
                left, right = current['foot_contact'].T
                right_age = np.where(right & ~left, right_age+1, 0)
                if fork is None and k >= design['first_zero_based_control']:
                    require((phase.phase == FORWARD).all(), 'The original standing transition did not complete')
                    policy_memory = {name: clone(getattr(policy, name)) for name in ('previous', 'history', 'previous_features')}
                    timing = schedule_state(schedule)
                    prepared = prepare()
                    selected = (prepared[1]['right_extension_eligible'] & (right_age >= first_age) & (right_age <= last_age)
                                & (last_right_pulse >= k-right_age+1) & (last_right_pulse < k))
                    if selected.sum() >= design['minimum_common_worlds']:
                        for name, value in policy_memory.items():
                            setattr(policy, name, value)
                        for name, value in timing.items():
                            setattr(schedule, name, value)
                        fork = compare(k, selected, current)
                        base.write(folder/'pulse-forks.json', {'right': fork})
                        prepared = prepare()
                else:
                    prepared = prepare()
                row = advance(prepared)
                last_right_pulse[row['hip_pitch_pulse_side'] == 2] = k
                commands.append(row)
                current = capture()
                phase.update(k+1, current)
                prefix.append(current)
                video.sample()
                if (k+1) % 25 == 0:
                    publish('右髋俯仰 共同站立与前进', refresh=True)
            require(fork is not None and env.guard_steps == plan['maximum_controls']
                    and env.common_step_counter == plan['pulse']['shared_controls'] and len(env.guard_resets) == 1,
                    'The matched branch or frozen route budget was incomplete')
            video.close()
            clip, = video.clips
            require(clip['complete_clip'] and clip['full_action'], 'The shared route is partially recorded')
            endpoint.save(plan['pulse']['shared_controls']+1)
            trace, motor = folder/'shared-trace.npz', folder/'shared-commands.npz'
            np.savez_compressed(trace, **arrays(prefix))
            np.savez_compressed(motor, **arrays(commands))
            base.write(folder/'guarded-evaluation.json', dict(status='collected-review-required',
                actual_control_steps=env.guard_steps, actual_physics_substeps=env.guard_steps*5, shared_controls=len(commands),
                recorded_control_transitions=env.guard_steps*worlds, physical_resets=1, learning_updates=0, hardware_io=False,
                shared_trace_sha256=base.sha(trace), shared_commands_sha256=base.sha(motor),
                forks_sha256=base.sha(folder/'pulse-forks.json'), video=[dict(clip, sha256=base.sha(folder/clip['file']))]))
            publish('右脚支撑时长对照 采样完成')
        finally:
            video.close()
            camera.close()
    return collect
