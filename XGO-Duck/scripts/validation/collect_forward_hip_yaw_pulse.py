"""Apply a single matched hip-target change after an exact simulator restore."""

from time import monotonic
from unittest.mock import patch

import numpy as np

from forward_hip_yaw_pulse_targets import ROLES, SIDES, apply_pulse, eligible
from roll_entry_snapshot import RollEntrySnapshot
from roll_stand_walk_sequence import FORWARD
from run_roll_stand_forward_diagnostic import StandForward, require


def arrays(rows):
    return {name: np.stack([row[name] for row in rows]) for name in rows[0]}


def make_pulse_collector(base, may_continue):
    def collect(env, obs, plan, folder, endpoint, networks, homes, sim_home, limits):
        cpu, torch = base.cpu, base.torch
        design = plan['pulse']
        worlds = plan['worlds']
        robot = env.scene['robot']
        joints, order = robot.find_joints(plan['walking']['joint_names'], preserve_order=True)
        feet, sites = robot.find_sites(('left_foot', 'right_foot'), preserve_order=True)
        require(order == plan['walking']['joint_names'] and list(sites) == ['left_foot', 'right_foot'], 'Site order changed')
        head = robot.find_bodies('jaw_soft')[0][0]
        tolerance = np.deg2rad(plan['roll']['actual_joint_limit_tolerance_deg'])
        bounds = cpu(limits)
        alive = np.ones(worlds, dtype=bool)
        term = env.action_manager.get_term('joint_pos')
        phase = StandForward(plan)
        policy = base.PolicyControl(plan, networks, homes, sim_home, limits)
        snapshot_cap = design['maximum_warp_snapshot_bytes']
        start = monotonic()
        sample_time, sample_control = start, 0
        reports, prefix, prefix_commands = {}, [], []
        view = dict(status='running', run_id=plan['run_id'], controls=0, maximum_controls=plan['maximum_controls'],
                    transitions=0, learning_updates=0, updates=[])

        def guard():
            require(may_continue() and monotonic()-start < plan['maximum_sampling_wall_seconds'],
                    'Observation, memory, pause or time guard stopped the pulse trial')

        def capture():
            state = base.capture(env, endpoint, alive, joints, head, feet, bounds, tolerance)
            require(all(np.isfinite(value).all() for value in state.values()) and state['joint_envelope_ok'].all()
                    and (state['head_penetration_m'] <= plan['roll']['criteria']['head_penetration_max_m']).all(),
                    'The original physical guard detected a nonfinite value, joint or head bound')
            return state

        def advance(state, pulse=None, substeps=None):
            guard()
            require(env.guard_steps < plan['maximum_controls'], 'Fixed pulse budget reached')
            observed = env.obs_buf['actor'].clone()
            target, commands = policy.next(env.common_step_counter+1, observed, phase)
            if pulse is not None:
                side, expected_worlds, polarity = pulse
                candidates, joint = eligible(plan, commands['requested_targets'], commands['applied_targets'],
                                              state['foot_contact'], side)
                np.testing.assert_array_equal(np.flatnonzero(candidates), expected_worlds)
                target = torch.as_tensor(apply_pulse(commands['applied_targets'], expected_worlds, joint,
                                                     polarity*design['delta_rad']), device=env.device)
            if substeps is None:
                _, _, done, timeout, _ = env.step(target-sim_home)
            else:
                original_step = env.sim.step

                def sampled_step(*args, **kwargs):
                    substeps.append(cpu(env.sim.data.ctrl))
                    return original_step(*args, **kwargs)

                with patch.object(env.sim, 'step', sampled_step):
                    _, _, done, timeout, _ = env.step(target-sim_home)
            require(not bool((done | timeout).any()), 'A world terminated during the bounded pulse trial')
            torch.testing.assert_close(term._processed_actions, target, rtol=0, atol=1e-6)
            endpoint.capture()
            require(not endpoint.first_mismatches, 'Contact witness differs from the physical constraint array')
            return dict(commands, observed=cpu(observed), actual_motor_target=cpu(term._processed_actions))

        # The branch experiment starts only after the real CUDA inventory is
        # captured and restored with zero physics and the observer is healthy.
        guard()
        beginning = RollEntrySnapshot(env, policy, snapshot_cap)
        zero = beginning.verify()
        require(zero['counters'] == [0, 0] and beginning.restore() == zero, 'Actual CUDA zero readback differs')
        fields = sorted(beginning.expected)
        delay = [name for name in fields if '.delay.' in name]
        require(delay and 'env.obs_buf' in fields, 'The snapshot did not include observations and motor lag')
        np.testing.assert_array_equal(cpu(env.obs_buf['actor']), cpu(obs['actor']))
        base.write(folder/'pulse-zero.json', dict(initial=zero, delay_fields=delay, observation_shape=list(obs['actor'].shape)))
        del beginning

        camera = base.Snapshot(env, folder)
        video = base.ActionVideo(camera, 'switch', folder, design['shared_controls']*.02)

        def publish(label, refresh=False):
            nonlocal sample_time, sample_control
            now, k = monotonic(), int(env.guard_steps)
            if k > sample_control:
                view['updates'].append(dict(control=k, transitions_per_second=(k-sample_control)*worlds/max(now-sample_time, .001)))
                sample_control, sample_time = k, now
            if refresh:
                view['frame'] = camera.capture()['file']
            if video.clips and video.clips[0]['complete_clip']:
                view['latest_clip'] = video.clips[0]['file']
            view.update(controls=k, transitions=k*worlds, purpose=label, updated_at=base.datetime.now(base.TZ).isoformat())
            base.live.publish(folder, view)
            print('HIP_YAW_PULSE', k, plan['maximum_controls'], label, flush=True)

        def compare_side(side, selected, current):
            name = SIDES[side]
            fork = RollEntrySnapshot(env, policy, snapshot_cap)
            captured = fork.verify()
            k = int(env.common_step_counter)
            require(captured['counters'] == [k, 5*k], 'Fork counters differ')
            saved_contacts = list(endpoint.rows)
            restores, case = {}, folder/name
            case.mkdir()
            for role, polarity in ROLES.items():
                guard()
                restores[role] = fork.restore()
                require(restores[role] == captured, 'Branch start differs from the matched snapshot')
                endpoint.rows = list(saved_contacts)
                state = capture()
                for field, expected in current.items():
                    np.testing.assert_array_equal(state[field], expected, err_msg=field)
                states, commands, motor = [state], [], []
                initial_motor = cpu(env.sim.data.ctrl)
                for offset in range(design['response_controls']):
                    commands.append(advance(state, (side, selected, polarity) if offset == 0 else None, motor))
                    state = capture()
                    states.append(state)
                require(len(motor) == 5*design['response_controls'], 'Motor substep count differs')
                np.savez_compressed(case/f'{role}.npz', initial_motor=initial_motor, motor_substeps=np.stack(motor),
                    **{f'state_{key}': value for key, value in arrays(states).items()},
                    **{f'command_{key}': value for key, value in arrays(commands).items()})
                publish(f'髋偏航脉冲 {"左脚" if side == 0 else "右脚"} {role}')
            require(fork.restore() == captured, 'Shared continuation changed after the probe')
            endpoint.rows = saved_contacts
            return dict(control=k, selected_worlds=selected.tolist(), captured=captured, restored=restores,
                        branches={role: base.sha(case/f'{role}.npz') for role in ROLES})

        try:
            current = capture()
            phase.update(0, current)
            prefix.append(current)
            video.sample()
            publish('髋偏航脉冲 共同站立与前进', refresh=True)
            for k in range(design['shared_controls']):
                if k >= design['earliest_fork_control'] and len(reports) < len(SIDES):
                    require((phase.phase == FORWARD).all(), 'Expected all worlds to have entered forward')
                    memory = (policy.previous, policy.history, policy.previous_features)
                    _, preview = policy.next(k+1, env.obs_buf['actor'].clone(), phase)
                    policy.previous, policy.history, policy.previous_features = memory
                    for side, name in enumerate(SIDES):
                        if name in reports:
                            continue
                        available, _ = eligible(plan, preview['requested_targets'], preview['applied_targets'],
                                                current['foot_contact'], side)
                        selected = np.flatnonzero(available)
                        if len(selected) >= design['minimum_worlds_per_fork']:
                            reports[name] = compare_side(side, selected, current)
                            base.write(folder/'pulse-forks.json', reports)
                prefix_commands.append(advance(current))
                current = capture()
                phase.update(k+1, current)
                prefix.append(current)
                video.sample()
                if (k+1) % 25 == 0:
                    publish('髋偏航脉冲 共同站立与前进', refresh=True)
            require(set(reports) == set(SIDES) and env.guard_steps == plan['maximum_controls']
                    and env.common_step_counter == design['shared_controls'] and len(env.guard_resets) == 1,
                    'A support phase had insufficient live coverage or the sampling budget differs')
            video.close()
            clip, = video.clips
            require(clip['complete_clip'] and clip['full_action'], 'The shared route clip is incomplete')
            endpoint.save(design['shared_controls']+1)
            prefix_path, commands_path = folder/'shared-trace.npz', folder/'shared-commands.npz'
            np.savez_compressed(prefix_path, **arrays(prefix))
            np.savez_compressed(commands_path, **arrays(prefix_commands))
            base.write(folder/'guarded-evaluation.json', dict(status='collected-review-required',
                actual_control_steps=env.guard_steps, actual_physics_substeps=env.guard_steps*5, shared_controls=len(prefix_commands),
                recorded_control_transitions=env.guard_steps*worlds, physical_resets=1, learning_updates=0, hardware_io=False,
                shared_trace_sha256=base.sha(prefix_path), shared_commands_sha256=base.sha(commands_path),
                forks_sha256=base.sha(folder/'pulse-forks.json'), video=[dict(clip, sha256=base.sha(folder/clip['file']))]))
            publish('髋偏航脉冲 采样完成')
        finally:
            video.close()
            camera.close()
    return collect
