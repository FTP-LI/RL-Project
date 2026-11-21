"""Measure yaw-invariant lean, support contacts and sagittal joint authority."""

from pathlib import Path

import mujoco
from mjlab.scene import Scene
import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import read
from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL
from routed_getup_sequence import GETUP
from run_nominal_skill_composition import PROJECT, ROOT, allowed, config, rel, sha
from verify_nominal_skill_composition import read_trace


KEY = 'nominal_supine_terminal_stance'
REPORT = ROOT / 'records/reports/nominal-skill-composition/supine-terminal-stance'
JOINT_PAIRS = (('left_hip_pitch', 'right_hip_pitch'), ('left_knee', 'right_knee'), ('left_ankle', 'right_ankle'))


def attitude(quaternion):
    r = rotation(quaternion)
    pitch = np.arctan2(-r[..., 2, 0], np.hypot(r[..., 2, 1], r[..., 2, 2]))
    roll = np.arctan2(r[..., 2, 1], r[..., 2, 2])
    tilt = np.arccos(np.clip(r[..., 2, 2], -1., 1.))
    return np.rad2deg(np.stack((pitch, roll, tilt), axis=-1))


def contacts(plant, data):
    floor, = np.flatnonzero(plant.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    left, right = (plant.geom(f'robot/{side}_foot_collision').id for side in ('left', 'right'))
    feet, internal = np.zeros(2, dtype=bool), {}
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.efc_address < 0:
            continue
        a, b = map(int, contact.geom)
        pair = {a, b}
        feet |= [pair == {floor, foot} for foot in (left, right)]
        if floor not in pair:
            key = ' + '.join(sorted(plant.geom(i).name.removeprefix('robot/') for i in (a, b)))
            internal[key] = min(internal.get(key, 0.), float(contact.dist))
    return feet, internal


def fixed_foot_response(plant, data, joints, sites, length, shift):
    jacobians = []
    for site in sites:
        linear, angular = (np.zeros((3, plant.nv)) for _ in range(2))
        mujoco.mj_jacSite(plant, data, linear, angular, site)
        jacobians.extend([linear / length, angular])
    matrix = np.vstack(jacobians)
    lateral = rotation(data.qpos[3:7])[:, 1]
    base = attitude(data.qpos[3:7])[0]
    arms = {}
    for names in JOINT_PAIRS:
        ids = [joints[name] for name in names]
        columns = plant.jnt_dofadr[ids]
        along = data.xaxis[ids] @ lateral
        assert np.all(np.abs(along) > .8), along
        directions = np.sign(along)
        pulse = np.zeros(plant.nv)
        pulse[columns] = directions
        pulse[:6] = np.linalg.lstsq(matrix[:, :6], -(matrix @ pulse), rcond=None)[0]
        trial = data.qpos.copy()
        mujoco.mj_integratePos(plant, trial, pulse, shift)
        residue = (matrix @ pulse).reshape(2, 2, 3) * shift
        arms[' / '.join(names)] = dict(target_direction_signs=directions.tolist(),
            root_pitch_delta_deg=float(attitude(trial[3:7])[0] - base),
            foot_displacement_residual_mm=float(np.linalg.norm(residue[:, 0], axis=-1).max() * length * 1000),
            foot_orientation_residual_deg=float(np.rad2deg(np.linalg.norm(residue[:, 1], axis=-1).max())))
    return arms


def trigger(trace, angles, limits):
    pitch, roll, tilt = np.moveaxis(angles, -1, 0)
    lo, hi = limits['root_tilt_deg']
    kinematic = (trace['active'] & trace['foot_contact'].all(axis=-1) & trace['joint_envelope_ok'] &
        (trace['mode'] == GETUP) & (tilt >= lo) & (tilt <= hi) &
        (abs(roll) <= limits['maximum_absolute_roll_deg']) & np.isfinite(pitch) &
        (np.linalg.norm(trace['angular_velocity'], axis=-1) <= limits['maximum_angular_speed_rad_s']))
    dwell = np.zeros(kinematic.shape[1], dtype=np.int32)
    profile = np.empty_like(kinematic)
    for k, sample in enumerate(kinematic):
        dwell = np.where(sample, dwell + 1, 0)
        profile[k] = dwell >= limits['minimum_dual_foot_dwell_controls']
    return profile


def main():
    project = read(PROJECT)
    state = project['goal_execution'][KEY]
    design = ROOT / state['design']
    audit = read(design)
    assert state['status'] == 'archive-audit-ready' and allowed(project, audit['stage'])
    assert not project['staged_training_request']['simulation_running'] and not REPORT.exists()
    prior = project['goal_execution']['nominal_roll_timestep_closed_loop']
    proof = read(ROOT / audit['target_cohort_source'], audit['target_cohort_sha256'])
    assert proof['collection_sha256'] == prior['collection_sha256']
    plan = read(ROOT / prior['protocol'], prior['protocol_sha256'])
    assert plan['environment']['seed'] == proof['seed'] == audit['seed'] not in audit['reserved_seeds']
    d = plan['composition']
    collection = ROOT / prior['collection']
    raw = read(collection, prior['collection_sha256'])
    folder = collection.parent
    release_path = folder / 'release-poses.npz'
    assert sha(release_path) == raw['release_sha256']
    with np.load(release_path, allow_pickle=False) as saved:
        poses = saved['poses']
    worlds = [x['world'] for x in proof['evaluated_forward_assignments']]
    end = d['roll_start_control']
    trace = read_trace(folder / 'shared', raw['prefix']['chunks'], 1, end, (
        'qpos', 'qvel', 'quaternion', 'mode', 'height', 'angular_velocity', 'active', 'foot_contact',
        'joint_envelope_ok', 'roll_phase'), sha)
    entry = trace['roll_phase'][-1] == ROLL
    assert entry.sum() == proof['total_roll_entry'] and not entry[worlds].any()
    assert tuple(poses[worlds]) == ('supine',) * 5
    references = np.flatnonzero(entry & (poses == 'supine'))[:audit['comparison_successful_supine_worlds']].tolist()
    angles = attitude(trace['quaternion'])
    flags = trigger(trace, angles, audit['candidate_trigger'])
    start, stop = audit['candidate_start_search_controls']
    first = {w: (int(found[0]) + start if len(found := np.flatnonzero(flags[start-1:stop, w])) else None) for w in worlds}
    c = audit['actuator_boundary_control']
    boundary, = (mark for mark in raw['prefix']['boundaries'] if mark['control'] == c)
    p = folder / 'shared' / boundary['file']
    assert sha(p) == boundary['sha256']
    with np.load(p, allow_pickle=False) as saved:
        prior_target, motor = saved['previous_target'], saved['motor'][0]

    scene = Scene(config(plan).scene, device='cpu')
    plant = scene.compile()
    (drive,) = scene['robot'].actuators
    firmware = drive._bam_model.actuator
    nominal, = set(drive.cfg.vin_range)
    kt, resistance = firmware.model.kt.value, firmware.model.R.value
    actuator = [plan['walking']['joint_names'].index(name) for pair in JOINT_PAIRS for name in pair]
    names = plan['walking']['joint_names']
    joints = {name: plant.joint('robot/' + name).id for name in names}
    dof, q = (np.asarray([values[joints[name]] for name in names]) for values in (plant.jnt_dofadr, plant.jnt_qposadr))
    slots = [int(np.flatnonzero(plant.actuator_trnid[:, 0] == joints[name]).item()) for name in names]
    assert plant.nu == len(names) and np.all(plant.actuator_gear[slots, 0] == 1.)
    sites = [plant.site(f'robot/{side}_foot').id for side in ('left', 'right')]
    bounds = np.asarray(plan['environment']['joint_limits'])
    speed = trace['qvel'][c-1][:, dof]
    max_ctrl = kt * (nominal * firmware.max_pwm - kt * speed) / resistance
    min_ctrl = kt * (-nominal * firmware.max_pwm - kt * speed) / resistance
    data, samples, responses = mujoco.MjData(plant), [], {}
    for k in audit['static_controls']:
        for world in (*worlds, *references):
            mujoco.mj_resetData(plant, data)
            data.qpos[:] = trace['qpos'][k-1, world]
            data.qvel[:] = trace['qvel'][k-1, world]
            mujoco.mj_forward(plant, data)
            feet, internal = contacts(plant, data)
            samples.append(dict(control=k, world=world, target_cohort=world in worlds,
                pitch_roll_tilt_deg=angles[k-1, world].tolist(),
                angular_speed_rad_s=float(np.linalg.norm(trace['angular_velocity'][k-1, world])),
                joints_rad=data.qpos[q].tolist(), archived_feet_contact=trace['foot_contact'][k-1, world].tolist(),
                reconstructed_feet_contact=feet.tolist(), reconstructed_self_contacts_distance_m=internal))
            if k == c and world in worlds:
                responses[world] = fixed_foot_response(plant, data, joints, sites, audit['fixed_foot_jacobian_scale_m'],
                    audit['geometry_motion_rad'])
    motors = []
    for world in worlds:
        values, targets = motor[world, slots], prior_target[world]
        assert np.isfinite(values).all()
        motors.append(dict(world=world, names=[names[j] for j in actuator],
            applied_prior_target_rad=targets[actuator].tolist(), motor_torque_nm=values[actuator].tolist(),
            nominal_firmware_upper_torque_margin_nm=(max_ctrl[world] - values)[actuator].tolist(),
            nominal_firmware_lower_torque_margin_nm=(values - min_ctrl[world])[actuator].tolist(),
            target_upper_range_margin_rad=(bounds[:, 1] - targets)[actuator].tolist(),
            target_lower_range_margin_rad=(targets - bounds[:, 0])[actuator].tolist()))
    windows = {k: dict(target_count=int(flags[k-1, worlds].sum()),
        total_count=int(flags[k-1].sum()), original_roll_entries=int((flags[k-1] & entry).sum()))
        for k in range(start, stop + 1, 25)}
    REPORT.mkdir()
    destination = REPORT / 'archive-audit.json'
    write(destination, dict(protocol_sha256=sha(design), source_protocol_sha256=prior['protocol_sha256'],
        collection_sha256=prior['collection_sha256'], source_script_sha256=sha(Path(__file__)),
        archived_motor_source_sha256=boundary['sha256'], target_worlds=worlds, matched_success_reference_worlds=references,
        samples=samples, first_candidate_detection_control_by_world=first, trigger_window_counts=windows,
        sagittal_kinematic_response_at_control=c, sagittal_kinematic_response=responses, actuator_control=c,
        motor_margin_scope='Measured first BAM torque and nominal supply PWM envelope. Supply drop and torque during a correction require closed-loop measurement.',
        motor_readback=motors, physical_steps=0, learning_updates=0, reserved_seed_used=False, hardware_io=False))
    state.update(status='archive-audit-reviewed', archive_audit=rel(destination), archive_audit_sha256=sha(destination),
        design_sha256=sha(design))
    project['staged_training_request']['current_evidence'] = rel(destination)
    project['current_evidence'] = rel(destination)
    write(PROJECT, project)
    print('SUPINE_TERMINAL_ARCHIVE_AUDIT', sha(destination), first, windows, flush=True)


if __name__ == '__main__':
    main()
