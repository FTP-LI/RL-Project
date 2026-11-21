"""Replay both support joints from one captured state with two zero branches."""

from time import monotonic
from unittest.mock import patch

import numpy as np

from collect_forward_hip_yaw_pulse import arrays
from forward_support_joint_pulse import ROLES, SIDES, branch_target, common_mask, observer_for
from roll_entry_snapshot import RollEntrySnapshot
from roll_stand_walk_sequence import FORWARD
from run_roll_stand_forward_diagnostic import StandForward, require


def make_collector(base, may_continue):
    def collect(env, obs, plan, folder, endpoint, networks, homes, sim_home, limits):
        cpu, torch = base.cpu, base.torch
        design, worlds = plan['pulse'], plan['worlds']
        robot = env.scene['robot']
        joints, order = robot.find_joints(plan['walking']['joint_names'], preserve_order=True)
        feet, sites = robot.find_sites(('left_foot', 'right_foot'), preserve_order=True)
        require(order == plan['walking']['joint_names'] and list(sites) == ['left_foot', 'right_foot'], 'Joint or site order changed')
        head = robot.find_bodies('jaw_soft')[0][0]
        tolerance = np.deg2rad(plan['roll']['actual_joint_limit_tolerance_deg'])
        bounds, alive = cpu(limits), np.ones(worlds, dtype=bool)
        term = env.action_manager.get_term('joint_pos')
        phase = StandForward(plan)
        observer = observer_for(plan)
        policy = base.PolicyControl(plan, networks, homes, sim_home, limits)
        snapshot_cap = design['maximum_warp_snapshot_bytes']
        start = monotonic()
        sample_time, sample_control = start, 0
        forks, prefix, prefix_commands = {}, [], []
        view = dict(status='running', run_id=plan['run_id'], controls=0, maximum_controls=plan['maximum_controls'],
                    transitions=0, learning_updates=0, updates=[])

        def guard():
            require(may_continue() and monotonic()-start < plan['maximum_sampling_wall_seconds'],
                    'Observation, memory, pause or time guard stopped the joint comparison')

        def capture():
            state = base.capture(env, endpoint, alive, joints, head, feet, bounds, tolerance)
            require(all(np.isfinite(value).all() for value in state.values()) and state['joint_envelope_ok'].all()
                    and (state['head_penetration_m'] <= plan['roll']['criteria']['head_penetration_max_m']).all(),
                    'The original physical guard detected a nonfinite value, joint or head bound')
            return state

        def selected_worlds(state, command, observed, side):
            mask = common_mask(plan, command['requested_targets'], command['applied_targets'],
                               state['foot_contact'], observer.predict(observed), side)
            return np.flatnonzero(mask)

        def advance(state, intervention=None, substeps=None):
            guard()
            require(env.guard_steps < plan['maximum_controls'], 'The frozen control budget was exhausted')
            observed = env.obs_buf['actor'].clone()
            target, commands = policy.next(env.common_step_counter+1, observed, phase)
            sensor_data = cpu(observed)
            if intervention is not None:
                side, selected, role = intervention
                np.testing.assert_array_equal(selected_worlds(state, commands, sensor_data, side), selected)
                changed = branch_target(plan, commands['applied_targets'], selected, side, role)
                target = torch.as_tensor(changed, device=env.device)

            if substeps is None:
                _, _, done, truncated, _ = env.step(target-sim_home)
            else:
                original_step = env.sim.step

                def record_motor(*args, **kwargs):
                    substeps.append(cpu(env.sim.data.ctrl))
                    return original_step(*args, **kwargs)

                with patch.object(env.sim, 'step', record_motor):
                    _, _, done, truncated, _ = env.step(target-sim_home)
            require(not bool((done | truncated).any()), 'A world terminated during the comparison')
            torch.testing.assert_close(term._processed_actions, target, rtol=0, atol=1e-6)
            endpoint.capture()
            require(not endpoint.first_mismatches, 'Contact data differs from the actual constraint array')
            return dict(commands, observed=sensor_data, actual_motor_target=cpu(term._processed_actions))

        guard()
        initial = RollEntrySnapshot(env, policy, snapshot_cap)
        zero = initial.verify()
        require(zero['counters'] == [0, 0] and initial.restore() == zero, 'Actual CUDA zero readback differs')
        fields = sorted(initial.expected)
        delays = [name for name in fields if '.delay.' in name]
        require(delays and 'env.obs_buf' in fields, 'The CUDA snapshot lacks observation or actuator delays')
        np.testing.assert_array_equal(cpu(env.obs_buf['actor']), cpu(obs['actor']))
        base.write(folder/'pulse-zero.json', dict(initial=zero, delay_fields=delays, observation_shape=list(obs['actor'].shape)))
        del initial

        camera = base.Snapshot(env, folder)
        video = base.ActionVideo(camera, 'switch', folder, design['shared_controls']*.02)

        def publish(label, refresh=False):
            nonlocal sample_time, sample_control
            now, k = monotonic(), int(env.guard_steps)
            if k > sample_control:
                rate = (k-sample_control)*worlds/max(now-sample_time, .001)
                view['updates'].append(dict(control=k, transitions_per_second=rate))
                sample_control, sample_time = k, now
            if refresh:
                view['frame'] = camera.capture()['file']
            if video.clips and video.clips[0]['complete_clip']:
                view['latest_clip'] = video.clips[0]['file']
            view.update(controls=k, transitions=k*worlds, purpose=label, updated_at=base.datetime.now(base.TZ).isoformat())
            base.live.publish(folder, view)
            print('SUPPORT_JOINT_PULSE', k, plan['maximum_controls'], label, flush=True)

        def compare(side, selected, current):
            fork = RollEntrySnapshot(env, policy, snapshot_cap)
            captured, k = fork.verify(), int(env.common_step_counter)
            require(captured['counters'] == [k, 5*k], 'Matched start counters differ')
            save_contacts = list(endpoint.rows)
            restores, hashes = {}, {}
            branch_dir = folder/SIDES[side]
            branch_dir.mkdir()
            for role in ROLES:
                guard()
                restores[role] = fork.restore()
                require(restores[role] == captured, 'A branch did not start from the same physics and policy state')
                endpoint.rows = list(save_contacts)
                state = capture()
                for name, array in current.items():
                    np.testing.assert_array_equal(state[name], array, err_msg=name)
                states, commands, motor = [state], [], []
                initial_motor = cpu(env.sim.data.ctrl)
                for j in range(design['response_controls']):
                    perturbation = (side, selected, role) if j == 0 else None
                    commands.append(advance(state, perturbation, motor))
                    state = capture()
                    states.append(state)
                require(len(motor) == 5*design['response_controls'], 'The motor substep recording is incomplete')
                path = branch_dir/f'{role}.npz'
                np.savez_compressed(path, initial_motor=initial_motor, motor_substeps=np.stack(motor),
                    **{f'state_{key}': value for key, value in arrays(states).items()},
                    **{f'command_{key}': value for key, value in arrays(commands).items()})
                hashes[role] = base.sha(path)
                publish(f'支撑关节脉冲 {"左脚" if side == 0 else "右脚"} {role}')
            require(fork.restore() == captured, 'The shared route could not be restored')
            endpoint.rows = save_contacts
            return dict(control=k, selected_worlds=selected.tolist(), captured=captured, restored=restores, branches=hashes)

        route_label = '支撑关节对照 共同站立与前进'
        try:
            current = capture()
            phase.update(0, current)
            prefix.append(current)
            video.sample()
            publish(route_label, refresh=True)
            for k in range(design['shared_controls']):
                if k >= design['earliest_fork_control'] and len(forks) < len(SIDES):
                    require((phase.phase == FORWARD).all(), 'Expected to find support phases while walking forward')
                    observed = env.obs_buf['actor'].clone()
                    memory = policy.previous, policy.history, policy.previous_features
                    _, preview = policy.next(k+1, observed, phase)
                    policy.previous, policy.history, policy.previous_features = memory
                    for side, label in enumerate(SIDES):
                        if label in forks:
                            continue
                        selected = selected_worlds(current, preview, cpu(observed), side)
                        if len(selected) >= design['minimum_worlds_per_fork']:
                            forks[label] = compare(side, selected, current)
                            base.write(folder/'pulse-forks.json', forks)
                prefix_commands.append(advance(current))
                current = capture()
                phase.update(k+1, current)
                prefix.append(current)
                video.sample()
                if (k+1) % 25 == 0:
                    publish(route_label, refresh=True)
            require(set(forks) == set(SIDES) and env.guard_steps == plan['maximum_controls']
                    and env.common_step_counter == design['shared_controls'] and len(env.guard_resets) == 1,
                    'A live support mask or the shared route was incomplete')
            video.close()
            clip, = video.clips
            require(clip['complete_clip'] and clip['full_action'], 'The shared route video is incomplete')
            endpoint.save(design['shared_controls']+1)
            trace, commands = folder/'shared-trace.npz', folder/'shared-commands.npz'
            np.savez_compressed(trace, **arrays(prefix))
            np.savez_compressed(commands, **arrays(prefix_commands))
            base.write(folder/'guarded-evaluation.json', dict(status='collected-review-required',
                actual_control_steps=env.guard_steps, actual_physics_substeps=env.guard_steps*5, shared_controls=len(prefix_commands),
                recorded_control_transitions=env.guard_steps*worlds, physical_resets=1, learning_updates=0, hardware_io=False,
                shared_trace_sha256=base.sha(trace), shared_commands_sha256=base.sha(commands),
                forks_sha256=base.sha(folder/'pulse-forks.json'), video=[dict(clip, sha256=base.sha(folder/clip['file']))]))
            publish('支撑关节对照 采样完成')
        finally:
            video.close()
            camera.close()
    return collect
