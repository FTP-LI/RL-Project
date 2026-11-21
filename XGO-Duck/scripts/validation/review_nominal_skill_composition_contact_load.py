"""Align saved solver loads with the original nominal roll scoring samples."""

import numpy as np

from record_io import write
from review_nominal_roll_supported_rotation import saved_window
from review_nominal_skill_composition_handoff import ROOT, read, replay, sha
from roll_metrics import progress
from roll_stand_walk_sequence import ROLL


KEY = 'nominal_skill_composition_contact_load'


def trace_quaternions(folder, chunks, first, last, worlds):
    values = []
    for part in chunks:
        start, end = max(first, part['first']), min(last, part['last'])
        if start > end:
            continue
        path = folder / part['file']
        assert sha(path) == part['sha256']
        span = slice(start - part['first'], end - part['first'] + 1)
        with np.load(path, allow_pickle=False) as trace:
            np.testing.assert_array_equal(trace['control'][span], np.arange(start, end + 1))
            values.append(trace['qpos'][span, :, 3:7])
    result = np.concatenate(values)
    assert result.shape == (last - first + 1, worlds, 4)
    return result


def angle(quaternion):
    active = np.ones(quaternion.shape[:2], dtype=bool)
    return progress(dict(quaternion=quaternion, ground_contact=active))[0]


def cohort(gaps, degrees, load, following, force, weight, endpoints, commands):
    rows, worlds = np.nonzero(gaps)
    on = load[rows, :, worlds]
    deg = degrees[rows, :, worlds]
    count = on[:, :, 0].sum(axis=1)
    has_following = following[rows, worlds]
    measures = []
    for i, (row, world) in enumerate(zip(rows, worlds, strict=True)):
        measures.append(dict(control=int(endpoints[row]), world=int(world), assigned_command=commands[world],
            positive_deg=float(deg[i].sum()), loaded_positive_deg=float((deg[i] * on[i, :, 0]).sum()),
            head_loaded_positive_deg=float((deg[i] * on[i, :, 1]).sum()), loaded_substeps=int(count[i]),
            final_substep_loaded=bool(on[i, -1, 0]), following_substep_loaded=bool(has_following[i])))
    total = float(deg.sum())
    under_load = float((deg * on[:, :, 0]).sum())
    return dict(intervals=len(rows), worlds_with_gaps=int(np.unique(worlds).size),
        physics_load_count_histogram=np.bincount(count, minlength=6).tolist(),
        final_substep_loaded=int(on[:, -1, 0].sum()),
        next_substep_loaded=int(has_following.sum()),
        final_loaded_following_unloaded=int((on[:, -1, 0] & ~has_following).sum()),
        head_loaded_intervals=int(on[:, :, 1].any(axis=1).sum()),
        foot_loaded_intervals=int(on[:, :, 2:].any(axis=(1, 2)).sum()),
        positive_rotation_deg=total, positive_rotation_under_load_deg=under_load,
        positive_rotation_fraction_under_load=under_load / total if total else None,
        median_load_by_physics_substep_body_weights=np.median(force[rows, :, worlds, 0] / weight, axis=0).tolist()
        if len(rows) else None, cases=measures)


def main():
    project_path = ROOT / 'project.json'
    project = read(project_path)
    state = project['goal_execution'][KEY]
    assert state['status'] == 'reviewed'
    result_file, collection_file = (ROOT / state[key] for key in ('result', 'collection'))
    score, record = (read(path, state[key + '_sha256']) for key, path in (('result', result_file), ('collection', collection_file)))
    folder = collection_file.parent
    plan = read(folder / 'protocol.json', state['protocol_sha256'])
    windows = plan['contact_load']
    n, dt = plan['composition']['worlds'], windows['physics_period_s']
    first, last = (windows[f'capture_{key}_control'] for key in ('first', 'last'))
    lo, hi = (windows[f'score_gap_{key}_endpoint_control'] for key in ('first', 'last'))
    proof = read(folder / 'zero/proof.json', record['process_local_zero_sha256'])
    guard = read(folder / 'zero/load-instrument-zero.json')
    assert proof['physical_substeps'] == guard['physical_substeps'] == 0 and guard['archived_reset_matches']
    receipt = record['contact_load']
    meta_path = ROOT / receipt['meta']
    assert meta_path == folder / 'contact-clock.json'
    meta = read(meta_path, receipt['meta_sha256'])
    assert meta['complete'] and meta['recorded_substeps'] == receipt['measured_substeps'] == (last - first) * 5
    assert receipt['substeps'] == record['physics_substeps'] == 5 * record['controls']
    weight, limit = meta['nominal_weight_n'], meta['force_threshold_n']
    assert limit == guard['threshold_n'] == weight * windows['load_threshold_nominal_weight_fraction']
    data_path = folder / 'contact-clock.npz'
    assert sha(data_path) == meta['sha256']
    with np.load(data_path, allow_pickle=False) as content:
        clock = dict(content)
    np.testing.assert_array_equal(clock['substep'], np.arange(first * 5 + 1, last * 5 + 1))
    np.testing.assert_array_equal(clock['before'][1:], clock['after'][:-1])
    qpos = trace_quaternions(folder, record['chunks'], first, last, n)
    np.testing.assert_array_equal(qpos[:-1], clock['before'][::5])
    np.testing.assert_array_equal(qpos[1:], clock['after'][4::5])
    integrated = np.concatenate((clock['before'][:1], clock['after']))
    delta = np.diff(angle(integrated), axis=0).reshape(last - first, 5, n)
    np.testing.assert_allclose(delta.sum(axis=1), np.diff(angle(qpos), axis=0), atol=1e-10, rtol=0.)
    frame = saved_window(folder, record['chunks'], first, last)
    score_delta = np.diff(angle(frame['quaternion']), axis=0)
    endpoint = folder / 'endpoint-contact.npz'
    assert sha(endpoint) == read(folder / 'endpoint-contact.json')['sha256']
    with np.load(endpoint, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['fresh'][first:last + 1, :, 0], frame['ground_contact'])

    force = clock['normal_force_n'].reshape(last - first, 5, n, 4)
    points = clock['points'].reshape(force.shape)
    assert np.isfinite(force).all() and (force >= 0).all() and (points >= 0).all()
    assert not (force[points == 0] > 1e-8).any()
    on = force > limit
    targets = np.arange(first + 1, last + 1)
    inspect = (targets >= lo) & (targets <= hi)
    assert targets[inspect][-1] < last
    gaps = inspect[:, None] & (frame['roll_phase'][:-1] == ROLL) & ~frame['ground_contact'][1:] & (score_delta > 0)
    _, _, passed, reasons, _, _ = replay(plan, folder, record)
    assert passed.sum() == score['roll_complete']
    assigned = np.empty(n, dtype=object)
    for name, worlds in plan['seven_command']['groups'].items():
        assigned[worlds] = name
    assert all(isinstance(name, str) for name in assigned)
    groups = {}
    for label, selected in (('roll_success', passed),
            ('insufficient_supported_rotation', reasons == 'insufficient-ground-supported-rotation'),
            ('other_roll_failure', ~passed & (reasons != 'insufficient-ground-supported-rotation'))):
        groups[label] = dict(roll_outcome_worlds=int(selected.sum()), **cohort(gaps & selected[None, :],
            np.rad2deg(np.maximum(delta, 0.)), on, np.concatenate((on[1:, 0, :, 0], np.zeros((1, n), bool))),
            force, weight, targets, assigned))
    assert sum(item['intervals'] for item in groups.values()) == int(gaps.sum())
    output = result_file.with_name('contact-load-review.json')
    review = dict(protocol_sha256=sha(folder / 'protocol.json'), collection_sha256=sha(collection_file),
        action_score_sha256=sha(result_file), contact_clock_sha256=sha(data_path), original_gates_unchanged=True,
        worlds=n, capture_control_boundaries=[first, last], assessed_endpoint_boundaries=[lo, hi],
        physics_period_s=dt, captured_physics_substeps=len(clock['substep']),
        measured_scene_substeps=n * len(clock['substep']), nominal_weight_n=weight, load_threshold_n=limit,
        qpos_physics_alignment_exact=True,
        max_score_vs_integrated_rotation_delta_rad=float(abs(score_delta - np.diff(angle(qpos), axis=0)).max()),
        groups=groups, additional_physical_steps=0, learning_updates=0)
    assert not output.exists()
    write(output, review)
    state.update(contact_load_review=str(output.relative_to(ROOT).as_posix()), contact_load_review_sha256=sha(output))
    write(project_path, project)
    failed = groups['insufficient_supported_rotation']
    print('NOMINAL_ROLL_CONTACT_LOAD_REVIEWED', sha(output), failed['worlds_with_gaps'],
          failed['intervals'], failed['positive_rotation_fraction_under_load'],
          failed['final_loaded_following_unloaded'], flush=True)


if __name__ == '__main__':
    main()
