"""Measure whether paired leg motion can load the head during roll transfer."""

from pathlib import Path

import mujoco
from mjlab.scene import Scene
import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha
from review_roll_head_support_geometry import Geometry
from review_roll_lateral_pitch_authority import torque
from review_roll_support_target_authority import generalized
from run_nominal_skill_composition import config


KEY = 'nominal_roll_sagittal_coupling_replay'
COHORTS = ('supine_supported_angle_failure', 'supine_roll_pass')


def read_sources(study):
    def source(key):
        record = study['sources'][key]
        return read(ROOT / record['file'], record['sha256'])

    plan = source('protocol')
    book = source('collection')
    assert plan['environment']['seed'] == study['seed']
    archive, earlier = source('support_transfer_comparison'), source('actuator_geometry_review')
    observed = archive['groups'][str(study['seed'])]
    result = []
    for label, count in zip(COHORTS, study['expected_events'], strict=True):
        before, after = study['verified_support_transition'].split('-to-')
        events = [row for row in observed['rows'][label]['unloads'] if row['before'] == before and row['after'] == after]
        assert len(events) == count and all(row['gap_ms'] > 0 for row in events)
        result.append(events)
    folder = (ROOT / study['sources']['collection']['file']).parent
    marker = read(folder / 'contact-clock.json', book['contact_load']['meta_sha256'])
    raw = folder / 'contact-clock.npz'
    assert sha(raw) == marker['sha256'] == observed['contact_clock_sha256'] == earlier['raw_contact_clock_sha256']
    with np.load(raw, allow_pickle=False) as data:
        clock = dict(data)
    return plan, result, marker, clock


def voltage_for(previous, supply):
    source, gain, minimum = supply
    return np.maximum(source - gain * np.abs(previous).sum(axis=1, keepdims=True), minimum)


def motor_pulse(motor, original, indices, world, coordinates, supply, dt, target_shift, limits):
    q_indices, v_indices = coordinates
    previous_torque = original['motor_ctrl'][indices - 5, world]
    smooth = original['filtered_target'][indices - 5, world]
    scheduled = original['delayed_target'][indices, world]
    target = np.clip(scheduled + target_shift, limits[:, 0], limits[:, 1])
    values, battery, filtered, realized = [], [], [], []
    # With one physical substep of actuator delay, the request issued in
    # control k first reaches the motor on the second substep of control k.
    for offset in range(-4, 1):
        t = indices + offset
        np.testing.assert_array_equal(original['delayed_target'][t, world], scheduled)
        supply_voltage = voltage_for(previous_torque, supply)
        q = original['qpos_before'][t, world][:, q_indices]
        velocity = original['qvel_before'][t, world][:, v_indices]
        previous_torque, smooth = torque(motor, supply_voltage, smooth, target, q, velocity, dt)
        values.append(previous_torque)
        battery.append(supply_voltage)
        filtered.append(smooth)
        realized.append(target - scheduled)
    return dict(torque=np.stack(values), voltage=np.stack(battery), filtered=np.stack(filtered), target=np.stack(realized))


def normal_response(plant, clock, indices, world, dofs):
    geometry = Geometry(plant)
    mass = np.empty((plant.nv, plant.nv))
    responses, clearances, velocities = [], [], []
    for q, velocity in zip(clock['qpos_before'][indices, world], clock['qvel_before'][indices, world], strict=True):
        state = geometry.measure(q.astype(float), velocity.astype(float))
        normal = geometry.data.geom_xmat[geometry.floor].reshape(3, 3)[:, 2]
        jacobian = normal @ geometry.jac
        mujoco.mj_fullM(plant, geometry.data, mass)
        responses.append(np.linalg.solve(mass, jacobian)[dofs])
        clearances.append(state['clearance_mm'])
        velocities.append(state['normal_velocity_mm_s'])
    return np.asarray(responses), np.asarray(clearances), np.asarray(velocities)


def summary(values, gap, select, dt):
    acceleration, clearance = values[select], gap[select]
    displacement = -0.5 * acceleration * dt**2 * 1000.
    positive = clearance > 0
    characteristic = float(np.median(clearance[positive]))
    return dict(events=len(displacement), positive_geometry_clearance_events=int(positive.sum()),
        downwards=int((acceleration < 0).sum()), median_head_acceleration_delta_m_s2=float(np.median(acceleration)),
        median_free_one_substep_downwards_displacement_mm=float(np.median(displacement)),
        median_positive_head_clearance_mm=characteristic,
        ratio_of_median_downward_displacement_to_median_positive_clearance=float(np.median(displacement) / characteristic),
        estimates_exceed_positive_clearance=int((positive & (displacement >= clearance)).sum()))


def main():
    project_file = ROOT / 'project.json'
    project = read(project_file)
    state, task = project['goal_execution'][KEY], project['staged_training_request']
    study = read(ROOT / state['protocol'], state['protocol_sha256'])
    assert state['status'] == 'frozen' and task['current_stage'] == study['stage']
    assert not (task['learning_running'] or task['simulation_running'])
    prior = study['sources']['sampling_audit']
    read(ROOT / prior['file'], prior['sha256'])
    official_symmetry = study['sources']['official_symmetry']
    assert sha(ROOT / official_symmetry['file']) == official_symmetry['sha256']
    from mjlab_microduck.tasks.symmetry import _JOINT_PERM, _JOINT_SIGN

    plan, events, marker, clock = read_sources(study)
    first_set, second_set = events
    event = first_set + second_set
    world = np.array([row['world'] for row in event])
    indices = np.array([row['start'] for row in event]) - clock['substep'][0]
    np.testing.assert_array_equal(clock['substep'][indices], [row['start'] for row in event])
    assert (indices >= 5).all() and (indices < len(clock['substep'])).all()
    cutoff = marker['force_threshold_n']
    force = clock['normal_force_n']
    assert (force[indices - 1, world, 1] > cutoff).all() and (force[indices, world, 0] <= cutoff).all()
    names = marker['actuator_layout']['joint_names']
    assert names == plan['walking']['joint_names']
    cfg = config(plan)
    scene = Scene(cfg.scene, device='cpu')
    plant = scene.compile()
    (drive,) = scene['robot'].actuators
    motor = drive._bam_model.actuator
    nominal, = set(drive.cfg.vin_range)
    drop, = set(drive.cfg.vin_drop_gain_range)
    supply = nominal, drop, drive.cfg.vin_min
    dt = study['physics_period_s']
    assert dt == cfg.sim.mujoco.timestep and drive.cfg.delay_min_lag == drive.cfg.delay_max_lag == 1
    joints = [plant.joint('robot/' + name).id for name in names]
    actuator = marker['actuator_layout']['motor_ids']
    q, dof = plant.jnt_qposadr[joints], plant.jnt_dofadr[joints]
    assert plant.nu == len(joints) == 14
    np.testing.assert_array_equal(plant.actuator_trnid[actuator, 0], joints)
    np.testing.assert_array_equal(q, marker['actuator_layout']['qpos_ids'])
    np.testing.assert_array_equal(dof, marker['actuator_layout']['dof_ids'])
    limits = np.asarray(plan['environment']['joint_limits'])
    coordinates = (q, dof)
    baseline = motor_pulse(motor, clock, indices, world, coordinates, supply, dt, np.zeros(len(names)), limits)
    span = indices[None, :] + np.arange(-4, 1)[:, None]
    observed_torque = clock['motor_ctrl'][span, world[None, :]]
    observed_voltage = clock['firmware_vin'][span, world[None, :]]
    observed_filtered = clock['filtered_target'][span, world[None, :]]
    torque_error = float(np.max(np.abs(baseline['torque'] - observed_torque)))
    voltage_error = float(np.max(np.abs(baseline['voltage'] - observed_voltage)))
    assert torque_error <= study['max_control_replay_error_nm'] and voltage_error <= study['max_voltage_error_v'], (torque_error, voltage_error)
    np.testing.assert_allclose(baseline['filtered'], observed_filtered, atol=2e-6, rtol=0)
    np.testing.assert_allclose(generalized(plant, actuator, observed_torque),
        clock['joint_actuator_torque_nm'][span, world[None, :]], atol=1e-6, rtol=0)

    relation, gaps, velocities = normal_response(plant, clock, indices, world, dof)
    failed = np.arange(len(event)) < len(first_set)
    options, admitted = {}, []
    criterion = study['admission']
    for joint in study['joint_pairs']:
        left, right = (names.index(side + '_' + joint) for side in ('left', 'right'))
        assert (_JOINT_PERM[left], _JOINT_PERM[right]) == (right, left) and _JOINT_SIGN[left] == _JOINT_SIGN[right] == -1
        for offset in study['signed_offsets_rad']:
            shift = np.zeros(len(names))
            shift[[left, right]] = [offset, -offset]
            changed = motor_pulse(motor, clock, indices, world, coordinates, supply, dt, shift, limits)
            final = generalized(plant, actuator, changed['torque'][-1])
            reference = generalized(plant, actuator, baseline['torque'][-1])
            normal_delta = np.einsum('ij,ij->i', relation, final - reference)
            outcome = {key: summary(normal_delta, gaps, pick, dt) for key, pick in zip(COHORTS, (failed, ~failed), strict=True)}
            f = outcome[COHORTS[0]]
            accepted = (
                f['downwards'] / f['events'] >= criterion['minimum_downward_response_in_failure_fraction'] and
                f['ratio_of_median_downward_displacement_to_median_positive_clearance']
                >= criterion['minimum_median_displacement_fraction_of_positive_head_clearance']
            )
            name = f'{joint}:{offset:+.2f}'
            options[name] = dict(left_target_delta_rad=offset, right_target_delta_rad=-offset, results=outcome,
                passes_mechanical_screen=bool(accepted))
            if accepted:
                admitted.append(name)
    primary_clearance_median = float(np.median(gaps[failed]))
    evidence = dict(protocol_sha256=state['protocol_sha256'], source_contact_clock_sha256=marker['sha256'],
        script_sha256=sha(Path(__file__)), cohorts=dict(zip(COHORTS, map(len, events), strict=True)),
        baseline_max_motor_torque_error_nm=torque_error, baseline_max_voltage_error_v=voltage_error,
        actual_failed_group_head_clearance_mm_median=primary_clearance_median,
        actual_failed_group_head_normal_velocity_mm_s_median=float(np.median(velocities[failed])),
        candidates=options, admitted=admitted, nominal_supply_v=nominal, drop_gain_v_per_nm=drop,
        estimated_displacement_is_fixed_state_free_response=True,
        new_physics_steps=0, learning_updates=0, reserved_seed_used=False, hardware_io=False)
    out = ROOT / study['evidence']
    assert not out.exists()
    write(out, evidence)
    state.update(status='reviewed', evidence=study['evidence'], evidence_sha256=sha(out), admitted=admitted)
    task.update(current_protocol_execution='sagittal-coupling-replay-reviewed', current_evidence=state['evidence'])
    project.update(current_protocol_execution=task['current_protocol_execution'], current_evidence=state['evidence'])
    write(project_file, project)
    print('NOMINAL_ROLL_SAGITTAL_COUPLING', sha(out), 'admitted', admitted, 'reference errors', torque_error, voltage_error, flush=True)


if __name__ == '__main__':
    main()
