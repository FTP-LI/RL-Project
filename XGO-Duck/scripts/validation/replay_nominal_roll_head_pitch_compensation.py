"""Compare a neck pitch change with a head pitch compensation on saved states."""

from pathlib import Path

import numpy as np
from mjlab.scene import Scene

from record_io import write
from replay_nominal_roll_sagittal_coupling import COHORTS, motor_pulse, normal_response, read_sources, summary
from review_nominal_skill_composition_handoff import ROOT, read, sha
from review_roll_head_support_geometry import Geometry
from review_roll_support_target_authority import generalized
from run_nominal_skill_composition import config


KEY = 'nominal_roll_head_pitch_compensation_replay'


def axes_and_attitude(plant, states, velocities, neck, head, shifts, limits):
    geom = Geometry(plant)
    joints = np.asarray([neck, head])
    qpos = plant.jnt_qposadr[joints]
    axes, angle, clearance = [], [], []
    for row, (pose, velocity) in enumerate(zip(states, velocities, strict=True)):
        initial = geom.measure(pose, velocity)
        neck_axis, head_axis = geom.data.xaxis[joints]
        axes.append(neck_axis @ head_axis)
        rotation = geom.data.geom_xmat[geom.head].reshape(3, 3).copy()
        angle_row, clearance_row = [], []
        for shift in shifts[:, row]:
            changed = pose.copy()
            changed[qpos] = np.clip(pose[qpos] + shift, limits[:, 0], limits[:, 1])
            after = geom.measure(changed, velocity)
            local = geom.data.geom_xmat[geom.head].reshape(3, 3)
            angle_row.append(float(np.rad2deg(np.arccos(np.clip((np.sum(rotation * local) - 1) / 2, -1., 1.)))))
            clearance_row.append(after['clearance_mm'] - initial['clearance_mm'])
        angle.append(angle_row)
        clearance.append(clearance_row)
    return np.asarray(axes), np.asarray(angle), np.asarray(clearance)


def main():
    file = ROOT / 'project.json'
    project = read(file)
    state, task = project['goal_execution'][KEY], project['staged_training_request']
    plan = read(ROOT / state['protocol'], state['protocol_sha256'])
    assert task['current_stage'] == plan['stage'] and state['status'] == 'frozen'
    source = plan['torque_replay_source']
    assert sha(ROOT / source['file']) == source['sha256']
    prior = plan['prior']
    study = read(ROOT / prior['protocol'], prior['protocol_sha256'])
    leg = read(ROOT / prior['evidence'], prior['evidence_sha256'])
    plant_spec, cohorts, marker, clock = read_sources(study)
    assert marker['sha256'] == leg['source_contact_clock_sha256']
    event = cohorts[0] + cohorts[1]
    world = np.array([row['world'] for row in event])
    index = np.array([row['start'] for row in event]) - clock['substep'][0]
    assert (index >= 5).all() and (index < len(clock['substep'])).all()
    names = marker['actuator_layout']['joint_names']
    assert names == plant_spec['walking']['joint_names']
    cols = [names.index(name) for name in ('neck_pitch', 'head_pitch')]
    cfg = config(plant_spec)
    scene = Scene(cfg.scene, device='cpu')
    plant = scene.compile()
    (drive,) = scene['robot'].actuators
    ids = np.array([plant.joint('robot/' + name).id for name in names])
    motor_ids = marker['actuator_layout']['motor_ids']
    q, dof = plant.jnt_qposadr[ids], plant.jnt_dofadr[ids]
    np.testing.assert_array_equal(plant.actuator_trnid[motor_ids, 0], ids)
    np.testing.assert_array_equal(q, marker['actuator_layout']['qpos_ids'])
    np.testing.assert_array_equal(dof, marker['actuator_layout']['dof_ids'])
    pos = clock['qpos_before'][index, world].astype(float)
    velocity = clock['qvel_before'][index, world].astype(float)
    limits = np.asarray(plant_spec['environment']['joint_limits'])
    empty = np.empty((0, len(event), len(cols)))
    axes, _, _ = axes_and_attitude(plant, pos, velocity, *ids[cols], empty, limits[cols])
    assert (np.abs(axes) >= .50).all() and (np.sign(axes) == np.sign(axes[0])).all()
    alignment = int(np.sign(axes[0]))
    supply = leg['nominal_supply_v'], leg['drop_gain_v_per_nm'], drive.cfg.vin_min
    dt = study['physics_period_s']
    motor = drive._bam_model.actuator
    baseline = motor_pulse(motor, clock, index, world, (q, dof), supply, dt, np.zeros(len(names)), limits)
    margin = index[None, :] + np.arange(-4, 1)[:, None]
    np.testing.assert_allclose(baseline['torque'], clock['motor_ctrl'][margin, world[None, :]],
        atol=study['max_control_replay_error_nm'], rtol=0)
    np.testing.assert_allclose(baseline['voltage'], clock['firmware_vin'][margin, world[None, :]],
        atol=study['max_voltage_error_v'], rtol=0)
    original = generalized(plant, motor_ids, baseline['torque'][-1])
    relation, gaps, _ = normal_response(plant, clock, index, world, dof)
    failed = np.arange(len(event)) < len(cohorts[0])
    comparison = {}
    admitted = []
    for neck_offset in plan['neck_target_magnitudes_rad']:
        head_offset = -alignment * np.sign(neck_offset) * plan['head_compensation_magnitude_rad']
        choices = ([neck_offset, 0.], [neck_offset, head_offset])
        shifts, response = [], []
        for neck_head in choices:
            shift = np.zeros(len(names))
            shift[cols] = neck_head
            actual = motor_pulse(motor, clock, index, world, (q, dof), supply, dt, shift, limits)
            shifts.append(actual['target'][-1][:, cols])
            delta = generalized(plant, motor_ids, actual['torque'][-1]) - original
            response.append(np.einsum('ij,ij->i', delta, relation))
        _, angle, clearance = axes_and_attitude(plant, pos, velocity, *ids[cols], np.stack(shifts), limits[cols])
        arms = {}
        for i, label in enumerate(('neck-only', 'neck-with-head-compensation')):
            rows = {}
            for group, mask in zip(COHORTS, (failed, ~failed), strict=True):
                row = summary(response[i], gaps, mask, dt)
                row.update(median_static_attitude_change_deg=float(np.median(angle[mask, i])),
                    median_static_head_clearance_delta_mm=float(np.median(clearance[mask, i])))
                rows[group] = row
            arms[label] = dict(target_radians=choices[i], rows=rows)
        single, paired = (arm['rows'][COHORTS[0]] for arm in arms.values())
        reduction = 1 - paired['median_static_attitude_change_deg'] / single['median_static_attitude_change_deg']
        gate = study['admission']
        passes = bool(reduction >= plan['orientation_reduction_min_fraction'] and
            paired['downwards'] / paired['events'] >= gate['minimum_downward_response_in_failure_fraction'] and
            paired['ratio_of_median_downward_displacement_to_median_positive_clearance'] >=
            gate['minimum_median_displacement_fraction_of_positive_head_clearance'])
        label = f'{neck_offset:+.2f}'
        comparison[label] = dict(arms=arms, median_static_orientation_reduction_in_shortfall_group_fraction=float(reduction),
            passes_mechanical_screen=passes)
        if passes:
            admitted.append(label)
    output = ROOT / plan['evidence']
    assert not output.exists()
    write(output, dict(protocol_sha256=state['protocol_sha256'], script_sha256=sha(Path(__file__)),
        fixed_event_clock_sha256=marker['sha256'], axis_dot_product_min=float(axes.min()), axis_dot_product_max=float(axes.max()),
        cohorts=dict(zip(COHORTS, map(len, cohorts), strict=True)), comparison=comparison, admitted_neck_signs=admitted,
        orientation_metric='Static change in world head geometry attitude for the recorded realizable joint target change.',
        force_metric='Fixed-state free normal acceleration; no new contacts were solved dynamically.',
        new_physical_steps=0, new_learning_updates=0, hardware_io=False))
    state.update(status='reviewed', evidence=plan['evidence'], evidence_sha256=sha(output), admitted=admitted)
    task.update(current_protocol_execution='head-pitch-compensation-replay-reviewed', current_evidence=state['evidence'])
    project.update(current_protocol_execution=task['current_protocol_execution'], current_evidence=state['evidence'])
    write(file, project)
    print('NOMINAL_ROLL_HEAD_PITCH_COMPENSATION', sha(output), admitted, flush=True)


if __name__ == '__main__':
    main()
