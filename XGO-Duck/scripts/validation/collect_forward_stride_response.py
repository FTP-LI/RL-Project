"""Replay a common CUDA state with repeat controls and three prospective interventions."""

from time import monotonic
from unittest.mock import patch

import numpy as np

from collect_forward_right_support_continuation import save_trace, take_snapshot
from command_input_mapping import transform
from forward_right_support_continuation import RightSupportContinuation
from roll_stand_walk_sequence import FORWARD
from run_roll_stand_forward_diagnostic import StandForward, require


LABELS = {
    'reference-a': '首段脉冲参照一',
    'right-hip-reverse': '反向右髋俯仰',
    'forward-input-0130': '前进模型输入0.130m/s',
    'forward-input-0115': '前进模型输入0.115m/s',
    'reference-b': '首段脉冲参照二',
}


class StrideControl(RightSupportContinuation):
    def next(self, k, command, observed, enabled):
        target, row = super().next(k, command, observed, enabled)
        applied = row['right_extension_applied']
        # Keep the measured support, gyro and headroom conditions. Reverse
        # only the added right hip target, at controls where no onset stacks.
        axis = self.onset.pitch[1]
        target[applied, axis] = command['applied_targets'][applied, axis] - np.float32(self.design['single_control_delta_rad'])
        return target, row


def mapped_command(command, design, role, k):
    candidate = design['internal_forward_input_m_s'].get(role)
    if candidate is None:
        return transform(command)
    internal = command.copy()
    scale = np.clip((np.asarray(k, dtype=np.float32)-design['shared_controls'])/design['input_ramp_controls'], 0, 1)
    delta = np.float32(candidate)-np.float32(design['external_forward_command_m_s'])
    internal[..., 0] += np.where(internal[..., 0] > 0, scale*delta, 0)
    return transform(internal)


def make_collector(base, may_continue):
    def collect(env, obs, plan, folder, endpoint, networks, homes, sim_home, limits):
        design, worlds = plan['stride_response'], plan['worlds']
        shared, length = design['shared_controls'], design['branch_controls']
        horizon, maximum = shared+length, plan['maximum_controls']
        robot = env.scene['robot']
        joints, names = robot.find_joints(plan['walking']['joint_names'], preserve_order=True)
        feet, sites = robot.find_sites(('left_foot', 'right_foot'), preserve_order=True)
        require(names == plan['walking']['joint_names'] and list(sites) == ['left_foot', 'right_foot'],
                'Joint or foot site order changed')
        head = robot.find_bodies('jaw_soft')[0][0]
        tolerance = np.deg2rad(plan['roll']['actual_joint_limit_tolerance_deg'])
        bounds = base.cpu(limits)
        phase, alive = StandForward(plan), np.ones(worlds, dtype=bool)
        policy = base.PolicyControl(plan, networks, homes, sim_home, limits)
        controller = StrideControl(plan, plan['duration_probe'])
        motor_term = env.action_manager.get_term('joint_pos')
        start = sample_time = monotonic()
        sample_step = 0
        role = 'shared'
        view = dict(status='running', run_id=plan['run_id'], controls=0, maximum_controls=maximum,
                    transitions=0, learning_updates=0, updates=[])
        cameras, branches = {}, {}

        def guard():
            require(may_continue() and monotonic()-start < plan['maximum_sampling_wall_seconds'],
                    'Observation, memory, pause or time guard stopped the stride comparison')

        def capture(k, update=True):
            state = base.capture(env, endpoint, alive, joints, head, feet, bounds, tolerance)
            require(all(np.isfinite(value).all() for value in state.values()) and state['joint_envelope_ok'].all()
                    and (state['head_penetration_m'] <= plan['roll']['criteria']['head_penetration_max_m']).all(),
                    'The original damaged-state guard stopped the stride comparison')
            if update:
                phase.update(k, state)
            return dict(state, phase_next=phase.phase.copy())

        def advance(k, states, commands):
            guard()
            require(env.guard_steps < maximum and env.common_step_counter == k, 'Physical control count exceeds protocol')
            observed = env.obs_buf['actor'].clone()
            with patch.object(base, 'transform', lambda command: mapped_command(command, design, role, k)):
                motor, row = policy.next(k+1, observed, phase)
            sensor = base.cpu(observed)
            target, detail = controller.next(k, row, sensor, role == 'right-hip-reverse')
            if detail['hip_pitch_pulse_side'].any() or detail['right_extension_applied'].any():
                motor = base.torch.as_tensor(target, device=env.device)
            _, _, done, timeout, _ = env.step(motor-sim_home)
            base.torch.testing.assert_close(motor_term._processed_actions, motor, atol=1e-6, rtol=0)
            alive[:] &= ~base.cpu(done | timeout)
            commands.append(dict(row, **detail, observed=sensor, actual_motor_target=base.cpu(motor_term._processed_actions),
                                 terminated=base.cpu(done), truncated=base.cpu(timeout)))
            endpoint.capture()
            states.append(capture(k+1))
            require(not endpoint.first_mismatches and alive.all(), 'Physical contact mismatch or terminated world')

        guard()
        snapshot_cap = plan['maximum_warp_snapshot_bytes']
        initial = take_snapshot(env, policy, controller, phase, snapshot_cap)
        initial_state = initial.verify()
        require(initial_state['counters'] == [0, 0] and initial.restore() == initial_state, 'Actual CUDA zero state changed')
        np.testing.assert_array_equal(base.cpu(env.obs_buf['actor']), base.cpu(obs['actor']))
        base.write(folder/'zero-readback.json', dict(initial=initial_state, fields=sorted(initial.expected)))
        del initial

        camera = base.Snapshot(env, folder)
        render = camera.render
        last, pixels = None, None

        def render_once():
            # All five encoders reuse the same physical frame in the shared route.
            nonlocal last, pixels
            key = role, env.common_step_counter
            if key != last:
                last, pixels = key, render()
            return pixels

        camera.render = render_once

        def publish(frame=False):
            nonlocal sample_time, sample_step
            count, now = int(env.guard_steps), monotonic()
            if count > sample_step:
                view['updates'].append(dict(control=count, transitions_per_second=(count-sample_step)*worlds/max(now-sample_time, .001)))
                sample_time, sample_step = now, count
            if frame:
                old = camera.capture()['file']
                name = f'frame-{role}-{env.common_step_counter}.jpg'
                (folder/old).replace(folder/name)
                view['frame'] = name
            view.update(controls=count, transitions=count*worlds, purpose=LABELS.get(role, '共同站立与前进'),
                        updated_at=base.datetime.now(base.TZ).isoformat())
            base.live.publish(folder, view)
            print('STRIDE_RESPONSE', plan['run_id'], role, count, maximum, env.common_step_counter, flush=True)

        prefix, prefix_commands = [], []
        shared_path = folder/'shared'
        shared_path.mkdir()
        try:
            for arm in design['role_order']:
                branch = folder/'clips'/arm
                branch.mkdir(parents=True)
                cameras[arm] = base.ActionVideo(camera, 'switch', branch, horizon*.02)
                cameras[arm].sample()
            prefix.append(capture(0))
            publish(frame=True)
            try:
                for k in range(shared):
                    advance(k, prefix, prefix_commands)
                    for recorder in cameras.values():
                        recorder.sample()
                    if (k+1) % 25 == 0:
                        publish(frame=(k+1) % 100 == 0 or k+1 == shared)
            finally:
                prefix_hashes = save_trace(shared_path, prefix, prefix_commands)
            require((phase.phase == FORWARD).all() and (phase.forward_at == design['standing_controls']).all(),
                    'A world did not pass the original standing gate')
            prefix_contacts = list(endpoint.rows)
            fork = take_snapshot(env, policy, controller, phase, snapshot_cap)
            captured = fork.verify()
            require(captured['counters'] == [shared, shared*5], 'The chosen CUDA state was not at its frozen control boundary')
            base.write(folder/'fork.json', dict(physical=captured, fields=sorted(fork.expected)))
            for role in design['role_order']:
                guard()
                restored = fork.restore()
                require(restored == captured, 'The CUDA state or controller memory did not restore')
                branch = folder/'clips'/role
                base.write(branch/'restored-fork.json', restored)
                endpoint.rows, endpoint.folder, endpoint.first_mismatches = list(prefix_contacts), branch, {}
                boundary = capture(shared, update=False)
                for key, value in boundary.items():
                    np.testing.assert_array_equal(value, prefix[-1][key], err_msg=key)
                states, commands = [boundary], []
                publish(frame=True)
                try:
                    for k in range(shared, horizon):
                        advance(k, states, commands)
                        cameras[role].sample()
                        if (k+1-shared) % 25 == 0:
                            publish(frame=(k+1-shared) % 50 == 0)
                finally:
                    hashes = save_trace(branch, states, commands)
                    endpoint.save(horizon+1)
                clip, = cameras[role].clips
                require(clip['complete_clip'] and clip['full_action'], 'The comparison route was only partially recorded')
                relative = f'clips/{role}/{clip["file"]}'
                branches[role] = dict(controls=len(commands), trace_sha256=hashes,
                    contact_sha256=base.sha(branch/'endpoint-contact.npz'),
                    video=dict(clip, path=relative, sha256=base.sha(folder/relative)))
                view['latest_clip'] = relative
                publish()
            require(env.guard_steps == maximum and env.common_step_counter == horizon and len(env.guard_resets) == 1,
                    'Physical rollover, reset or control counts differ from the protocol')
            base.write(folder/'guarded-evaluation.json', dict(status='collected-review-required',
                protocol_sha256=base.sha(base.PLAN), actual_control_steps=maximum, actual_physics_substeps=5*maximum,
                recorded_control_transitions=worlds*maximum, physical_resets=1, shared_controls=shared,
                prefix_sha256=prefix_hashes, branches=branches, learning_updates=0, hardware_io=False, heldout_used=False))
            publish()
        finally:
            for recorder in reversed(list(cameras.values())):
                recorder.close()
            camera.close()
    return collect
