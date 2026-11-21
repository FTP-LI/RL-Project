"""Read back firmware torque and screen small target offsets on saved geometry."""

from pathlib import Path

import mujoco
from mjlab.scene import Scene
import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha
from review_roll_foot_transfer import SoleGeometry
from review_roll_head_support_geometry import Geometry
from review_roll_lateral_pitch_authority import torque
from review_roll_support_target_authority import generalized, position_response
from run_nominal_skill_composition import config


KEY = 'nominal_roll_support_transfer'
OFFSETS = (-.02, .02)
HEAD = ('neck_pitch', 'head_pitch')
FEET = ('left_hip_pitch', 'left_knee', 'left_ankle', 'right_hip_pitch', 'right_knee', 'right_ankle')


def statistics(values):
    return dict(median=float(np.median(values)), p90=float(np.quantile(values, .9)), maximum=float(np.max(values)))


def field(clock, name, time, worlds, columns=None):
    sampled = clock[name][time, worlds]
    return sampled if columns is None else sampled[:, columns]


def prefix_repetition(earlier, recent, recent_folder, boundary):
    ledger = ROOT / earlier['collection']
    prior = read(ledger, earlier['collection_sha256'])
    first_change, changed = None, 0
    for left, right in zip(prior['chunks'], recent['chunks']):
        assert (left['first'], left['last']) == (right['first'], right['last'])
        if left['first'] > boundary:
            break
        a, b = ledger.parent / left['file'], recent_folder / right['file']
        assert sha(a) == left['sha256'] and sha(b) == right['sha256']
        with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
            span = x['control'] <= boundary
            differs = (x['qpos'][span] != y['qpos'][span]).any(axis=-1)
            any_change = np.flatnonzero(differs.any(axis=1))
            if len(any_change) and first_change is None:
                first_change = int(x['control'][int(any_change[0])])
            changed += int(differs.sum())
    return dict(first_qpos_different_control=first_change, differing_scene_controls=changed,
                all_controls_at_or_before=boundary, earlier_collection_sha256=sha(ledger))


def main():
    registry = ROOT / 'project.json'
    project = read(registry)
    state = project['goal_execution'][KEY]
    comparison = ROOT / state['support_transfer_comparison']
    transfer = read(comparison, state['support_transfer_comparison_sha256'])
    ledger = ROOT / state['collection']
    record = read(ledger, state['collection_sha256'])
    raw = ledger.parent
    plan = read(raw / 'protocol.json', state['protocol_sha256'])
    meta = read(raw / 'contact-clock.json', record['contact_load']['meta_sha256'])
    archive = raw / 'contact-clock.npz'
    assert sha(archive) == meta['sha256'] == transfer['groups']['1257']['contact_clock_sha256']
    group = transfer['groups']['1257']['rows']['supine_supported_angle_failure']
    events = [row for row in group['unloads'] if row['before'] == row['after'] == 'head']
    assert events and len(events) == group['transition_counts']['head-to-head']
    with np.load(archive, allow_pickle=False) as stored:
        clock = dict(stored)
    worlds = plan['composition']['worlds']
    names = plan['walking']['joint_names']
    channels = meta['actuator_layout']
    assert channels['joint_names'] == names and worlds == len(clock['before'][0])
    assert clock['firmware_vin'].shape == (len(clock['substep']), worlds, 1)
    np.testing.assert_array_equal(clock['qpos_before'][..., 3:7], clock['before'])
    np.testing.assert_allclose(clock['delayed_target'][1:], clock['position_target'][:-1], atol=1e-7, rtol=0)
    scene = Scene(config(plan).scene, device='cpu')
    model = scene.compile()
    (drive,) = scene['robot'].actuators
    motor = drive._bam_model.actuator
    joints = [model.joint('robot/' + name).id for name in names]
    act = channels['motor_ids']
    np.testing.assert_array_equal(model.actuator_trnid[act, 0], joints)
    np.testing.assert_array_equal(model.jnt_dofadr[joints], channels['dof_ids'])
    np.testing.assert_array_equal(model.jnt_qposadr[joints], channels['qpos_ids'])
    dt = plan['contact_load']['physics_period_s']
    step_limit = motor.model.max_velocity.value * dt
    raw_limits = np.asarray(plan['environment']['joint_limits'])
    np.testing.assert_allclose(clock['filtered_target'][1:],
        np.clip(clock['delayed_target'][1:], clock['filtered_target'][:-1] - step_limit,
                clock['filtered_target'][:-1] + step_limit), atol=2e-6, rtol=0)
    measured = clock['joint_actuator_torque_nm']
    compiled = generalized(model, act, clock['motor_ctrl'])
    np.testing.assert_allclose(compiled, measured, atol=1e-6, rtol=0)

    off = np.array([row['start'] - int(clock['substep'][0]) for row in events])
    world = np.array([row['world'] for row in events])
    assert (off >= 5).all() and (off < len(clock['before'])).all()
    np.testing.assert_array_equal(clock['substep'][off], [row['start'] for row in events])
    poses = field(clock, 'qpos_before', off, world).astype(float)
    velocities = field(clock, 'qvel_before', off, world).astype(float)
    head_geom, inverse, clear_mm = position_response(Geometry(model), poses, velocities, HEAD, OFFSETS)
    sole = SoleGeometry(model)
    old_gaps = np.array([[g['clearance_mm'] for g in sole.measure(p, v).values()] for p, v in zip(poses, velocities, strict=True)])
    raw_original = field(clock, 'motor_ctrl', off, world)
    errors = []
    for shift in range(-4, 1):
        time = off + shift
        replay, _ = torque(motor, field(clock, 'firmware_vin', time, world),
            field(clock, 'filtered_target', time - 1, world), field(clock, 'delayed_target', time, world),
            field(clock, 'qpos_before', time, world, channels['qpos_ids']),
            field(clock, 'qvel_before', time, world, channels['dof_ids']), dt)
        errors.append(float(np.abs(replay - field(clock, 'motor_ctrl', time, world)).max()))
    assert max(errors) <= 3e-6, errors

    head = {}
    for j, name in enumerate(HEAD):
        col = names.index(name)
        for k, offset in enumerate(OFFSETS):
            smooth = field(clock, 'filtered_target', off - 5, world)
            for shift in range(-4, 1):
                time = off + shift
                target = field(clock, 'delayed_target', time, world)
                target[:, col] = np.clip(target[:, col] + offset, *raw_limits[col])
                raw_delta, smooth = torque(motor, field(clock, 'firmware_vin', time, world), smooth, target,
                    field(clock, 'qpos_before', time, world, channels['qpos_ids']),
                    field(clock, 'qvel_before', time, world, channels['dof_ids']), dt)
            torque_change = generalized(model, act, raw_delta)[:, col] - generalized(model, act, raw_original)[:, col]
            gap_change = clear_mm[:, j, k]
            accel = inverse[:, j] * torque_change
            promising = (gap_change <= -.1) & (accel < 0.) & (np.abs(torque_change) >= 1e-3)
            head[f'{name}:{offset:+.2f}'] = dict(cases_with_combined_effect=int(promising.sum()),
                static_clearance_change_mm=statistics(gap_change), torque_change_nm=statistics(torque_change),
                free_normal_acceleration_change_m_s2=statistics(accel))

    legs = {}
    for name in FEET:
        col = names.index(name)
        position, side = channels['qpos_ids'][col], int(name.startswith('right_'))
        for offset in OFFSETS:
            gaps = []
            for p, v in zip(poses, velocities, strict=True):
                moved = p.copy()
                moved[position] = np.clip(moved[position] + offset, *raw_limits[col])
                gaps.append(sole.measure(moved, v)[name.split('_', 1)[0]]['clearance_mm'])
            values = np.array(gaps)
            legs[f'{name}:{offset:+.2f}'] = dict(cases_with_sole_within_one_mm=int((values <= 1.).sum()),
                remaining_clearance_mm=statistics(values), clearance_change_mm=statistics(values - old_gaps[:, side]))

    out = comparison.with_name('actuator-geometry-review.json')
    assert not out.exists()
    previous = project['goal_execution']['nominal_skill_composition_independent']
    prefix = prefix_repetition(previous, record, raw, plan['contact_load']['capture_first_control'])
    report = dict(transfer_comparison_sha256=sha(comparison), raw_contact_clock_sha256=sha(archive),
        script_sha256=sha(Path(__file__)), events=len(events), worlds=int(np.unique(world).size),
        pre_instrumentation_repeatability=prefix,
        max_original_motor_replay_error_nm=max(errors),
        max_motor_to_generalized_torque_error_nm=float(np.abs(compiled - measured).max()),
        actual_head_clearance_mm=statistics(head_geom[:, 0]), nearest_actual_sole_clearance_mm=statistics(old_gaps.min(axis=1)),
        signed_joint_offsets_rad=list(OFFSETS), head_fixed_pose_probes=head, sole_fixed_pose_probes=legs,
        integrated_perturbed_rolls=0, new_physics_steps=0)
    write(out, report)
    state.update(actuator_geometry_review=out.relative_to(ROOT).as_posix(), actuator_geometry_review_sha256=sha(out))
    write(registry, project)
    print('NOMINAL_SUPPORT_ACTUATOR_REVIEWED', sha(out), len(events), max(errors),
        max(x['cases_with_combined_effect'] for x in head.values()),
        max(x['cases_with_sole_within_one_mm'] for x in legs.values()), flush=True)


if __name__ == '__main__':
    main()
