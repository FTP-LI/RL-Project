"""Compare two continuous rolls from the same simulator and landing-latch state."""

from time import monotonic
from unittest.mock import patch

import numpy as np

from check_continuous_forward_fork_state import controller_snapshot, restore_controller
from collect_forward_right_support_continuation import save_trace
from forward_input_timing import post_entry_forward_input
from getup_release_fork_state import digest, signature
from roll_entry_snapshot import RollEntrySnapshot
from roll_screen_stop_guard import check_online, check_state
from roll_stand_walk_sequence import FAILED, ROLL, Sequence
from run_roll_stand_forward_diagnostic import require


ROLES = ('original-forward-input', 'admitted-forward-input')
LABELS = {ROLES[0]: '原前进输入', ROLES[1]: '调整后前进输入', 'shared': '共同站立、滚翻与落地'}


def remap(command, control, first_forward, design):
    return post_entry_forward_input(command, control, first_forward,
        design['delay_from_each_world_forward_entry_controls'], design['ramp_controls'],
        design['external_forward_command_m_s'], design['admitted_input_m_s'])


def make_collector(base, may_continue):
    def collect(env, obs, plan, folder, endpoint, networks, homes, sim_home, limits):
        design = plan['integration']
        shared, horizon = design['shared_control_boundary'], plan['maximum_controls']
        worlds, budget = plan['worlds'], design['maximum_unique_controls']
        robot = env.scene['robot']
        joints, names = robot.find_joints(plan['walking']['joint_names'], preserve_order=True)
        feet, sites = robot.find_sites(('left_foot', 'right_foot'), preserve_order=True)
        require(names == plan['walking']['joint_names'] and list(sites) == ['left_foot', 'right_foot'],
                'Joint or foot site order changed')
        head = robot.find_bodies('jaw_soft')[0][0]
        tolerance = np.deg2rad(plan['roll']['actual_joint_limit_tolerance_deg'])
        bounds = base.cpu(limits)
        sequence = Sequence(plan)
        policy = base.PolicyControl(plan, networks, homes, sim_home, limits)
        term = env.action_manager.get_term('joint_pos')
        alive = np.ones(worlds, dtype=bool)
        began = sample_time = monotonic()
        sample_control = 0
        role = 'shared'
        view = dict(status='running', run_id=plan['run_id'], controls=0, maximum_controls=budget,
                    transitions=0, learning_updates=0, updates=[])
        prefix, prefix_commands, branches, cameras = [], [], {}, {}
        camera = None
        stop = None

        def guard():
            require(may_continue() and monotonic()-began < plan['maximum_sampling_wall_seconds'],
                    'Observation, memory, pause or time guard stopped the continuous pair')

        def current():
            return base.capture(env, endpoint, alive, joints, head, feet, bounds, tolerance)

        def record_state(k, states):
            state = current()
            decision = check_state(state, k, plan)
            valid = not decision['stop']
            if valid:
                sequence.update(k, state)
                decision = check_online(sequence, state, k, plan)
            states.append(dict(state, phase_next=sequence.phase.copy(), guard_phase_valid=np.asarray(valid)))
            return decision if decision['stop'] else None

        def advance(k, commands, motor=None):
            guard()
            require(env.guard_steps < budget and env.common_step_counter == k, 'The physical control boundary changed')
            observed = env.obs_buf['actor'].clone()
            if role == ROLES[1]:
                walk_worlds = (sequence.phase != ROLL) & (sequence.phase != FAILED)
                with patch.object(base, 'transform', lambda values: remap(values, k, sequence.forward_at[walk_worlds], design)):
                    target, row = policy.next(k+1, observed, sequence)
            else:
                target, row = policy.next(k+1, observed, sequence)
            if motor is None:
                _, _, ended, timed_out, _ = env.step(target-sim_home)
            else:
                original_step = env.sim.step

                def observe_motor(*args, **kwargs):
                    motor.append(base.cpu(env.sim.data.ctrl))
                    return original_step(*args, **kwargs)

                with patch.object(env.sim, 'step', observe_motor):
                    _, _, ended, timed_out, _ = env.step(target-sim_home)
            base.torch.testing.assert_close(term._processed_actions, target, atol=1e-6, rtol=0)
            alive[:] &= ~base.cpu(ended | timed_out)
            commands.append(dict(row, observed=base.cpu(observed), actual_motor_target=base.cpu(term._processed_actions),
                                 terminated=base.cpu(ended), truncated=base.cpu(timed_out)))
            endpoint.capture()
            require(not endpoint.first_mismatches, 'Physical endpoint contact differs')

        def publish(frame=False, status='running'):
            nonlocal sample_time, sample_control
            total, now = int(env.guard_steps), monotonic()
            if total > sample_control:
                view['updates'].append(dict(control=total, transitions_per_second=(total-sample_control)*worlds/
                                            max(now-sample_time, .001)))
                sample_time, sample_control = now, total
            if frame:
                old = camera.capture()['file']
                name = f'frame-{role}-{env.common_step_counter}.jpg'
                (folder/old).replace(folder/name)
                view['frame'] = name
            view.update(status=status, controls=total, transitions=total*worlds, purpose=LABELS[role],
                        updated_at=base.datetime.now(base.TZ).isoformat())
            base.live.publish(folder, view)
            print('ROLL_FORWARD_INTEGRATION', plan['run_id'], role, total, budget, env.common_step_counter, flush=True)

        def sample(recorder):
            if recorder.writer is not None:
                if alive[0]:
                    recorder.sample()
                else:
                    recorder._sample_terminal()

        def clip_receipt(recorder, arm):
            recorder.finish(complete=False)
            clip, = recorder.clips
            relative = f'clips/{arm}/{clip["file"]}'
            return dict(clip, path=relative, sha256=base.sha(folder/relative))

        guard()
        zero = RollEntrySnapshot(env, policy, plan['maximum_warp_snapshot_bytes'])
        initial = zero.verify()
        require(initial['counters'] == [0, 0] and zero.restore() == initial, 'Actual CUDA zero state changed')
        np.testing.assert_array_equal(base.cpu(env.obs_buf['actor']), base.cpu(obs['actor']))
        base.write(folder/'zero-readback.json', dict(initial=initial, fields=sorted(zero.expected)))
        del zero
        camera = base.Snapshot(env, folder)
        render = camera.render
        last, pixels = None, None

        def render_once():
            nonlocal last, pixels
            key = role, env.common_step_counter
            if key != last:
                last, pixels = key, render()
            return pixels

        camera.render = render_once
        shared_folder = folder/'shared'
        shared_folder.mkdir()
        try:
            for arm in ROLES:
                branch = folder/'clips'/arm
                branch.mkdir(parents=True)
                cameras[arm] = base.ActionVideo(camera, 'switch', branch, horizon*.02)
                cameras[arm].sample()
            stop = record_state(0, prefix)
            publish(frame=True)
            try:
                for k in range(shared):
                    if stop:
                        break
                    advance(k, prefix_commands)
                    stop = record_state(k+1, prefix)
                    for recorder in cameras.values():
                        sample(recorder)
                    if (k+1) % 50 == 0 or stop:
                        publish(frame=True)
            finally:
                prefix_hashes = save_trace(shared_folder, prefix, prefix_commands)
            require(not stop and len(prefix_commands) == shared, f'Shared route stopped at {stop}')
            contacts = list(endpoint.rows)
            memory = controller_snapshot(sequence)
            controller_hash = digest(signature(memory))
            np.savez_compressed(folder/'fork-controller.npz', **memory)
            fork = RollEntrySnapshot(env, policy, plan['maximum_warp_snapshot_bytes'])
            physical = fork.verify()
            require(physical['counters'] == [shared, shared*5], 'The exact CUDA fork boundary differs')
            base.write(folder/'fork.json', dict(physical=physical, controller_sha256=controller_hash,
                controller_file_sha256=base.sha(folder/'fork-controller.npz'), fields=sorted(fork.expected)))
            motor_reference = None
            for role in ROLES:
                guard()
                restored = fork.restore()
                restore_controller(sequence, memory)
                require(restored == physical and digest(signature(controller_snapshot(sequence))) == controller_hash,
                        'Physical state or landing latch did not restore')
                branch = folder/'clips'/role
                base.write(branch/'restored-fork.json', dict(physical=restored, controller_sha256=controller_hash))
                endpoint.rows, endpoint.folder, endpoint.first_mismatches = list(contacts), branch, {}
                alive[:] = prefix[-1]['active']
                boundary = dict(current(), phase_next=sequence.phase.copy(), guard_phase_valid=np.asarray(True))
                for key, value in boundary.items():
                    np.testing.assert_array_equal(value, prefix[-1][key], err_msg=key)
                states, commands, first_motor = [boundary], [], []
                publish(frame=True)
                try:
                    for k in range(shared, horizon):
                        advance(k, commands, first_motor if k == shared else None)
                        stop = record_state(k+1, states)
                        sample(cameras[role])
                        if (k+1-shared) % 50 == 0 or stop:
                            publish(frame=True)
                        if stop:
                            break
                    require(len(first_motor) == 5, 'First physical motor control is incomplete')
                    if motor_reference is None:
                        motor_reference = np.stack(first_motor)
                    np.testing.assert_array_equal(np.stack(first_motor), motor_reference)
                    np.savez_compressed(branch/'first-motor.npz', substeps=np.stack(first_motor))
                finally:
                    hashes = save_trace(branch, states, commands)
                    endpoint.save(horizon+1)
                clip = clip_receipt(cameras[role], role)
                branches[role] = dict(controls=len(commands), stop=stop, trace_sha256=hashes,
                    contact_sha256=base.sha(branch/'endpoint-contact.npz'), first_motor_sha256=base.sha(branch/'first-motor.npz'),
                    video=clip)
                view['latest_clip'] = clip['path']
                publish()
                require(not stop and len(commands) == horizon-shared, f'Original action guard stopped {role}: {stop}')
            require(env.guard_steps == budget and env.common_step_counter == horizon and len(env.guard_resets) == 1,
                    'Actual dynamics do not match the two declared branches')
            base.write(folder/'guarded-evaluation.json', dict(status='collected-review-required',
                protocol_sha256=base.sha(base.PLAN), actual_control_steps=budget, actual_physics_substeps=budget*5,
                recorded_control_transitions=budget*worlds, saved_episode_controls=horizon, physical_resets=1,
                prefix_sha256=prefix_hashes, branches=branches, learning_updates=0, heldout_used=False, hardware_io=False))
        except BaseException as error:
            # Retain the authoritative physical counter across restored clocks.
            base.write(folder/'paired-stop.json', dict(stopped=True, detail=stop, role=role, reason=str(error),
                physical_controls=int(env.guard_steps), episode_control=int(env.common_step_counter), branches=branches))
            raise
        finally:
            for recorder in reversed(list(cameras.values())):
                recorder.close()
            camera.close()
    return collect
