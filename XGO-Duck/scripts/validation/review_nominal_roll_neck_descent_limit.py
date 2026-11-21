"""Screen a bounded target-slew rule on saved control states before using it dynamically."""

import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha
from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL


SPEC = ROOT / 'configs/experiments/nominal_roll_neck_descent_limit_draft.json'


def correction(previous, current, gravity, pitch_speed, rolling, spec):
    x, y, z = gravity.T
    lateral = np.maximum(np.sqrt(np.maximum(1. - y*y, 0.)), 1e-6)
    angle = np.rad2deg(np.arctan2(x, -z / lateral)) % 360.
    a, b, c, d = spec['phase_ramp_deg']
    phase = np.clip(np.minimum((angle-a)/(b-a), (d-angle)/(d-c)), 0., 1.)
    jump = current - previous
    limited = np.clip(spec['minimum_nominal_target_delta_rad_per_control'] - jump, 0., spec['maximum_candidate_addition_rad'])
    active = rolling & (pitch_speed > 0.) & (abs(y) <= spec['maximum_lateral_gravity_magnitude'])
    return limited * phase * active


def main():
    project_file = ROOT / 'project.json'
    project = read(project_file)
    row = project['goal_execution']['nominal_roll_support_transfer']
    spec = read(SPEC)
    paths = {}
    for label in ('support_transfer_comparison', 'actuator_geometry_review'):
        path = ROOT / row[label]
        digest = spec['support_transfer_review_sha256' if label == 'support_transfer_comparison' else label + '_sha256']
        assert sha(path) == row[label + '_sha256'] == digest
        paths[label] = path
    comparison = read(paths['support_transfer_comparison'])
    episode = comparison['groups']['1257']['rows']['supine_supported_angle_failure']
    cases = [case for case in episode['unloads'] if case['before'] == case['after'] == 'head']
    ledger = ROOT / row['collection']
    raw = read(ledger, row['collection_sha256'])
    folder = ledger.parent
    plan = read(folder / 'protocol.json', row['protocol_sha256'])
    clocks = read(folder / 'contact-clock.json', raw['contact_load']['meta_sha256'])
    archive = folder / 'contact-clock.npz'
    assert sha(archive) == clocks['sha256'] and clocks['sha256'] == comparison['groups']['1257']['contact_clock_sha256']
    worlds = np.array([case['world'] for case in cases])
    controls = np.array([case['score_control'] - 1 for case in cases])
    substeps = np.array([case['start'] for case in cases]) - clocks['first_substep'] - 5
    assert (substeps >= 5).all()
    column = clocks['actuator_layout']['joint_names'].index(spec['changed_joint'])
    with np.load(archive, allow_pickle=False) as clock:
        np.testing.assert_array_equal(clock['substep'][substeps], 5*controls+1)
        current = clock['position_target'][substeps, worlds, column]
        previous = clock['position_target'][substeps-5, worlds, column]
        pose = clock['before'][substeps, worlds]
    speed, phase = np.empty(len(cases)), np.zeros(len(cases), bool)
    checked = np.zeros(len(cases), bool)
    for chunk in raw['chunks']:
        ids = (controls >= chunk['first']) & (controls <= chunk['last'])
        if ids.any():
            path = folder / chunk['file']
            assert sha(path) == chunk['sha256']
            with np.load(path, allow_pickle=False) as trace:
                local = controls[ids] - chunk['first']
                np.testing.assert_array_equal(trace['qpos'][local, worlds[ids], 3:7], pose[ids])
                speed[ids] = trace['angular_velocity'][local, worlds[ids], 1]
                phase[ids] = trace['roll_phase'][local, worlds[ids]] == ROLL
            checked |= ids
    assert checked.all() and phase.all()
    raw_change = correction(previous, current, -rotation(pose)[:, 2, :], speed, phase, spec)
    lo, hi = plan['environment']['joint_limits'][column]
    addition = np.clip(current + raw_change, lo, hi) - current
    assert (addition >= -1e-7).all() and (addition <= spec['maximum_candidate_addition_rad'] + 1e-7).all()
    np.testing.assert_array_equal(correction(previous, current, -rotation(pose)[:, 2, :], speed, ~phase, spec), 0.)
    gate = spec['archive_acceptance']
    seen = addition >= gate['minimum_pre_unload_adjustment_rad']
    qualified = sum(bool(seen[worlds == i].any()) for i in np.unique(worlds))
    covered = qualified / np.unique(worlds).size
    limits = spec['future_dynamics']
    expected_controls = limits['matched_snapshot_control'] + 3*(plan['composition']['maximum_controls']-limits['matched_snapshot_control'])
    assert expected_controls == limits['maximum_batched_controls']
    assert expected_controls * plan['composition']['worlds'] == limits['maximum_control_transitions']
    output = paths['support_transfer_comparison'].with_name('neck-descent-limit-preflight.json')
    assert not output.exists()
    write(output, dict(protocol_sha256=sha(SPEC), collection_sha256=sha(ledger), actual_clock_sha256=sha(archive),
        checked_archived_events=len(cases), checked_worlds=int(np.unique(worlds).size), events_above_effect_floor=int(seen.sum()),
        worlds_above_effect_floor=qualified, world_coverage=covered, approved_for_actual_cuda_zero=covered >= gate['minimum_episodes_covered_fraction'],
        source_phase='Archived noiseless actor geometry proxy and saved body pitch rate',
        median_archived_neck_target_descend_rad=float(np.median(current-previous)),
        maximum_addition_rad=float(addition.max()), additional_physical_steps=0,
        cases=[dict(world=int(i), control=int(c), adjustment_rad=float(d)) for i, c, d in zip(worlds, controls, addition, strict=True)]))
    row.update(neck_descent_limit_preflight=output.relative_to(ROOT).as_posix(), neck_descent_limit_preflight_sha256=sha(output))
    write(project_file, project)
    print('NOMINAL_NECK_DESCENT_LIMIT_PREVIEW', sha(output), len(cases), int(seen.sum()), qualified, covered)


if __name__ == '__main__':
    main()
