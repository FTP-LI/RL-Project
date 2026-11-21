"""Bound BAM electrical drive at saved head-support samples without stepping."""
import inspect
from pathlib import Path

import numpy as np
import torch
from mjlab.scene import Scene

from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_roll_head_support_geometry import Geometry, JOINTS, OUT, prepare
from s1_reference_inputs import ROOT, read, sha


def reconstruct_internal_target(targets, max_change):
    """Replay the one-substep delay and five firmware updates per control."""
    smooth = np.zeros_like(targets)
    for step in range(1, len(targets)):
        # The buffer backfills with the first command on its first write.
        prior = targets[max(1, step - 1)]
        state = np.clip(prior, smooth[step-1] - max_change, smooth[step-1] + max_change)
        for _ in range(4):
            state = np.clip(targets[step], state - max_change, state + max_change)
        smooth[step] = state
    return smooth


def check_target_replay(targets, expected, motor, dt):
    """Compare the compact recurrence with the real delay and firmware methods."""
    from mjlab.utils.buffers import DelayBuffer

    subset = torch.from_numpy(np.ascontiguousarray(targets))
    delay = DelayBuffer(min_lag=1, max_lag=1, batch_size=targets.shape[1], device='cpu')
    zero = np.zeros_like(targets[0])
    motor.q_target_smooth = zero
    for step, command in enumerate(subset[1:], start=1):
        for _ in range(5):
            delay.append(command)
            motor.compute_control(delay.compute().numpy(), zero, zero, dt)
        np.testing.assert_array_equal(motor.q_target_smooth, expected[step])


def motor_at_voltage(motor, dt, q, dq, target, smooth, limits, voltage):
    motor.vin = voltage
    motor.q_target_smooth = smooth.copy()
    command = motor.compute_control(target, q, dq, dt)
    filtered = motor.q_target_smooth
    direction = np.sign(filtered - q)
    endpoint = np.where(direction > 0, limits[:, 1], limits[:, 0])
    motor.q_target_smooth = smooth.copy()
    max_command = motor.compute_control(endpoint, q, dq, dt)
    torque = motor.compute_torque(command, True, q, dq)
    max_torque = motor.compute_torque(max_command, True, q, dq)

    kt, resistance = motor.model.kt.value, motor.model.R.value
    emf = kt * dq
    max_change = motor.model.max_velocity.value * dt
    np.testing.assert_array_equal(filtered, np.clip(target, smooth - max_change, smooth + max_change))
    raw = voltage * (filtered - q) * motor.kp * motor.error_gain * motor.model.error_gain_ratio.value
    cap = voltage * motor.max_pwm
    np.testing.assert_allclose(command, np.clip(raw, -cap, cap), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(torque, kt * (command - emf) / resistance, rtol=1e-12, atol=1e-12)
    headroom = np.maximum(0., direction * (max_torque - torque))
    return dict(pwm_limited=np.isclose(abs(command), cap, rtol=0, atol=1e-9),
        drive_torque_nm=direction * torque, legal_target_headroom_nm=headroom,
        demand_deg=np.rad2deg(filtered-q), speed_deg_s=np.rad2deg(dq),
        smooth_limited=np.abs(filtered-target) > 1e-7,
        request_outside_limits=(target < limits[:, 0]) | (target > limits[:, 1]))


def review(trace, rows, geometry, order, motor, dt, voltages):
    columns = [order.index(name) for name in JOINTS]
    worlds = np.asarray([row['world'] for row in rows])
    succeeded = np.asarray([row['success'] for row in rows], bool)
    saved_steps = np.asarray([round(row['before_air']['time_s'] / .02) for row in rows])
    assert len(np.unique(worlds)) == len(rows)
    targets = trace['applied_targets'][:, :, columns]
    smooth = reconstruct_internal_target(targets, motor.model.max_velocity.value * dt)
    check_target_replay(targets, smooth, motor, dt)
    groups = {}
    for lead in (2, 1, 0):
        steps = saved_steps - lead
        assert (steps > 0).all()
        supported = trace['active'][steps, worlds] & trace['head_contact'][steps, worlds]
        q = trace['qpos'][steps, worlds][:, geometry.addresses].astype(np.float64)
        dq = trace['qvel'][steps, worlds][:, geometry.dofs].astype(np.float64)
        np.testing.assert_array_equal(q, trace['joint_positions'][steps, worlds][:, columns])
        window = groups[str(lead)] = {}
        for name, offset in (('delayed_first_substep', 0), ('undelayed_counterfactual', 1)):
            # Both requests start at the same archived state and firmware goal.
            # The new command normally arrives one physical substep later.
            target = targets[steps + offset, worlds].astype(np.float64)
            internal = smooth[steps, worlds].astype(np.float64)
            requests = window[name] = {}
            for voltage in voltages:
                metrics = motor_at_voltage(motor, dt, q, dq, target, internal, geometry.ranges, voltage)
                result = requests[str(voltage)] = {}
                for success, role in ((False, 'failure'), (True, 'success')):
                    mask = supported & (succeeded == success)
                    assert mask.any()
                    result[role] = {}
                    for i, joint in enumerate(JOINTS):
                        data = {key: value[mask, i] for key, value in metrics.items()}
                        result[role][joint] = dict(samples=int(mask.sum()),
                            pwm_limited=int(data['pwm_limited'].sum()),
                            smooth_limited=int(data['smooth_limited'].sum()),
                            request_outside_limits=int(data['request_outside_limits'].sum()),
                            motor_brakes_command_direction=int((data['drive_torque_nm'] < 0).sum()),
                            median_drive_torque_nm=float(np.median(data['drive_torque_nm'])),
                            median_legal_target_headroom_nm=float(np.median(data['legal_target_headroom_nm'])),
                            median_demand_deg=float(np.median(data['demand_deg'])),
                            median_speed_deg_s=float(np.median(data['speed_deg_s'])))
    return groups


def main():
    from bam.mjlab import BamActuator
    from bam.feetech.actuator import STS3215Actuator
    from mjlab.utils.buffers import DelayBuffer
    from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator

    geometry_path = OUT / 'development-head-support-geometry.json'
    geometry_proof = read(geometry_path)
    assert geometry_proof['status'] == 'passed-saved-data-only' and not geometry_proof['heldout_used']
    sources = dict(geometry_proof['source_sha256'])
    sources[geometry_path.relative_to(ROOT).as_posix()] = sha(geometry_path)
    for name, digest in sources.items():
        assert sha(ROOT / name) == digest, name
    cohorts, parameters = {}, {}
    control_boundaries_checked = {}
    for role in ('development-a', 'development-b'):
        proof = read(OUT / f'{role}-analysis.json')
        folder = ROOT / proof['evaluator']
        plan = read(folder / 'protocol.json')
        run = read(folder / 'evaluation.json')
        cfg = prepare(plan)
        (servo,) = cfg.scene.entities['robot'].articulation.actuators
        assert cfg.sim.mujoco.timestep == .004 and cfg.decimation == 5
        assert servo.vin is None and servo.vin_range == (7.7, 7.7)
        assert servo.vin_min == 7.0 and servo.vin_drop_gain_range == (.1, .1)
        assert servo.delay_min_lag == servo.delay_max_lag == 1
        assert set(cfg.events) <= {
            'reset_base', 'reset_robot_joints', 'reset_action_history', 'set_ground_state',
            'set_roulade_state', 'ground_roulade_reset', 'foot_friction', 'expand_bam_friction_fields',
        }
        assert run['seed'] == plan['seed'] and run['first_episode_only']
        params_path = Path(servo._resolved_json_path)
        scene = Scene(cfg.scene, device='cpu')
        (wrapper,) = scene['robot'].actuators
        assert type(wrapper) is FrictionDRBamActuator
        motor = wrapper._bam_model.actuator
        assert type(motor) is STS3215Actuator and motor.kp == servo.kp_fw
        parameters[role] = dict(actuator_type=type(motor).__name__, physics_dt=.004, control_dt=.02, kp_fw=motor.kp,
            kp_scale=1., kd_scale=1., kt=motor.model.kt.value, resistance=motor.model.R.value,
            max_pwm=motor.max_pwm, error_gain=motor.error_gain * motor.model.error_gain_ratio.value,
            max_internal_speed_rad_s=motor.model.max_velocity.value, initial_smoothed_goal_rad=0.,
            voltage_floor=servo.vin_min, nominal_voltage=servo.vin_range[0], load_drop_gain=servo.vin_drop_gain_range[0])
        geometry = Geometry(scene.compile())
        with np.load(folder / 'scene-static-fields.npz') as stored:
            compare_arrays(model_arrays(geometry.model), {key: stored[key] for key in stored.files})
        names = run['joint_names']
        assert len(names) == 14 and all(name in names for name in JOINTS)
        with np.load(folder / 'forward-roll-trace.npz') as data:
            trace = {key: data[key] for key in ('qpos', 'qvel', 'joint_positions', 'active',
                                               'head_contact', 'applied_targets')}
        control_boundaries_checked[role] = len(trace['applied_targets']) - 1
        cohorts[role] = review(trace, geometry_proof['cohorts'][role]['rows'], geometry, names, motor,
                               .004, (servo.vin_min, servo.vin_range[0]))
        print('ROLL_HEAD_MOTOR_ENVELOPE', role, cohorts[role]['0']['delayed_first_substep'], flush=True)
        sources[(folder / 'evaluation.json').relative_to(ROOT).as_posix()] = sha(folder / 'evaluation.json')
        source_types = (STS3215Actuator, BamActuator, FrictionDRBamActuator, DelayBuffer)
        source_files = (params_path, *(Path(inspect.getfile(cls)) for cls in source_types))
        for path in source_files:
            digest = sha(path)
            matching = [pinned for name, pinned in plan['training_source_sha256'].items()
                        if path.as_posix().endswith('/' + name)]
            assert all(digest == pinned for pinned in matching), path
            sources[str(path)] = digest
    assert parameters['development-a'] == parameters['development-b'] and not torch.cuda.is_initialized()
    write(OUT / 'development-head-motor-envelope.json', dict(status='passed-saved-data-only',
        source_sha256=sources, script_sha256=sha(Path(__file__)), motor_parameters=parameters['development-a'],
        cohorts=cohorts, step_offsets_are_controls_before_departure=True, analytic_parity=True,
        delay_and_target_replay_exact=True, control_boundaries_checked=control_boundaries_checked,
        scope=('Motor-side calculation at archived control boundaries, with the STS3215 target reconstructed from '
               'the saved command stream and both configured voltage endpoints. The undelayed request is a fixed-state '
               'counterfactual. No substep voltage, realized joint torque, friction or contact force was measured.'),
        heldout_used=False, new_physical_steps=0, learning_updates=0))


if __name__ == '__main__':
    main()
