"""Align control-rate roll scoring and substep loads within saved runs."""

from collections import Counter

import numpy as np

from record_io import write
from review_nominal_roll_supported_rotation import saved_window
from review_nominal_skill_composition_contact_load import angle, trace_quaternions
from review_nominal_skill_composition_handoff import ROOT, read, replay, sha
from review_nominal_roll_neck_pair_physics import release_poses
from review_nominal_roll_support_transfer import support_name
from roll_stand_walk_sequence import ROLL


KEY = 'nominal_roll_support_sampling_audit'
LABELS = ('ground', 'head', 'left_foot', 'right_foot')


def load_session(source):
    paths = {key: ROOT / source[key] for key in ('collection', 'protocol', 'result')}
    book, plan, score = (read(paths[key], source[key + '_sha256']) for key in paths)
    folder = paths['collection'].parent
    assert sha(folder / 'protocol.json') == source['protocol_sha256'] == score['protocol_sha256']
    assert score['collection_sha256'] == source['collection_sha256']
    return folder, book, plan, score


def longest_unloaded(timing):
    consecutive = np.zeros(timing.shape[1], dtype=int)
    peak = consecutive.copy()
    for off in timing:
        consecutive = np.where(off, consecutive + 1, 0)
        peak = np.maximum(peak, consecutive)
    return peak * 4


def med(values, mask):
    return float(np.median(values[mask]))


def describe(mask, phase, angle20, endpoint, amount, loads, following, force):
    scored = phase & ~endpoint & (angle20 > 0)
    control_lost = (angle20 * scored).sum(axis=0)
    ground = loads[..., 0]
    active = phase[:, None, :]
    unsupported = (amount * ~ground * active).sum(axis=(0, 1))
    off = (~ground & active).reshape(-1, len(mask))
    time = off.sum(axis=0) * 4
    duration = longest_unloaded(off)
    offsets = (angle20[:, None, :] * phase[:, None, :] * ~ground).sum(axis=0)
    interval, world = np.where(scored & mask[None, :])
    angle_at_gap = amount[interval, :, world]
    contact_at_gap = loads[interval, :, world, 0]
    previous = ground[interval, -1, world]
    next_load = following[interval, world]
    physical_at_gap = angle_at_gap * contact_at_gap
    return dict(
        worlds=int(mask.sum()), missed_intervals_20ms=len(world), worlds_with_missed_intervals=len(np.unique(world)),
        median_missed_positive_rotation_20ms_deg=med(control_lost, mask),
        median_physical_unloaded_positive_rotation_4ms_deg=med(unsupported, mask),
        median_total_unloaded_ms_4ms=med(time, mask),
        median_longest_contiguous_unload_ms_4ms=med(duration, mask),
        median_peak_head_force_n=med(force[..., 1].max(axis=(0, 1)), mask),
        positive_missed_rotation_with_load_fraction=float(physical_at_gap.sum() / angle_at_gap.sum()),
        final_loaded_following_unloaded=int((previous & ~next_load).sum()),
        fully_unloaded_intervals=int((~contact_at_gap.any(axis=1)).sum()),
        intervals_loading_both_sides_of_20ms_boundary=int((previous & next_load).sum()),
        diagnostic_median_contact_offset_loss_deg=[med(offset, mask) for offset in offsets],
    )


def reload_pattern(gaps, load_flags, start, last, mask):
    ranges = Counter()
    durations = []
    delayed = 0
    for interval, world in zip(*np.where(gaps & mask[None, :]), strict=True):
        i = start + (interval + 1) * 5
        assert i < last
        if not load_flags[i - 1, world, 0] or load_flags[i, world, 0]:
            continue
        following = np.flatnonzero(load_flags[i:last, world, 0])
        if len(following):
            r = int(following[0])
            durations.append(r * 4)
            before, after = support_name(load_flags[i - 1, world]), support_name(load_flags[i + r, world])
            ranges[f'{before}-to-{after}'] += 1
        else:
            delayed += 1
    assert durations
    return dict(recovery_counts=dict(ranges), onset_load_gap_ms_median=float(np.median(durations)),
        onset_load_gap_ms_p90=float(np.quantile(durations, .9)), remaining_unloaded_at_capture_end=delayed)


def analyze_branch(folder, branch, chunks, plan, rolled, poses, spec):
    is_branch = branch is not None
    clock_folder = folder / branch['role'] if is_branch else folder
    marker, data = (clock_folder / name for name in ('contact-clock.json', 'contact-clock.npz'))
    contact = branch['summary'] if is_branch else rolled['book']['contact_load']
    clock = read(marker, contact['contact_clock_sha256'] if is_branch else contact['meta_sha256'])
    digest = sha(data)
    assert digest == clock['sha256'] and clock['complete'] and clock['labels'] == list(LABELS)
    if is_branch:
        assert digest == rolled['score']['neck_rule']['loads']['contact_clock_sha256']
    assert clock['force_threshold_n'] == clock['nominal_weight_n'] * spec['load_threshold_nominal_weight_fraction']
    n = plan['composition']['worlds']
    first = (clock['first_substep'] - 1) // 5
    last = clock['last_substep'] // 5
    assert first == spec['window']['roll_origin_control'] + 20 and last >= first + 35
    q = trace_quaternions(folder, chunks, first, last, n)
    with np.load(data, allow_pickle=False) as physics:
        np.testing.assert_array_equal(physics['substep'], np.arange(first * 5 + 1, last * 5 + 1))
        np.testing.assert_array_equal(physics['before'][1:], physics['after'][:-1])
        np.testing.assert_array_equal(q[:-1], physics['before'][::5])
        np.testing.assert_array_equal(q[1:], physics['after'][4::5])
        fine = np.diff(angle(np.concatenate((physics['before'][:1], physics['after']))), axis=0)
        raw_load = physics['normal_force_n']
        points = physics['points']
    edges = saved_window(folder, chunks, first, last)
    integrated = fine.reshape(-1, 5, n).sum(axis=1)
    np.testing.assert_allclose(integrated, np.diff(angle(q), axis=0), atol=1e-10, rtol=0)
    # The scored root quaternion is read through forward kinematics in float32
    # while qpos is the exact integrated coordinate saved by the solver.
    broad = np.diff(angle(edges['quaternion']), axis=0)
    np.testing.assert_allclose(integrated, broad, atol=16 * np.finfo(q.dtype).eps, rtol=0)
    assert raw_load.shape == points.shape == (len(fine), n, len(LABELS))
    assert np.isfinite(raw_load).all() and (raw_load >= 0).all() and not (raw_load[points == 0] > 1e-8).any()
    endpoint_file = clock_folder / 'endpoint-contact.npz'
    contact_digest = read(clock_folder / 'endpoint-contact.json')['sha256']
    assert sha(endpoint_file) == contact_digest
    if is_branch:
        assert contact_digest == branch['summary']['contact_sha256']
    with np.load(endpoint_file, allow_pickle=False) as endpoint:
        np.testing.assert_array_equal(endpoint['fresh'][first:last + 1, :, 0], edges['ground_contact'])
    load_flags = raw_load > clock['force_threshold_n']
    lo, hi = spec['window']['end_controls']
    begin, end = lo - first - 1, hi - first
    assert 0 <= begin < end < len(fine) // 5
    current = slice(begin, end)
    sample = slice(begin * 5, end * 5)
    phase = edges['roll_phase'][begin:end] == ROLL
    contact = edges['ground_contact'][begin + 1:end + 1]
    positive20 = np.maximum(np.rad2deg(broad[current]), 0.)
    amount = np.maximum(np.rad2deg(fine[sample]), 0.).reshape(-1, 5, n)
    following = load_flags[5 * (np.arange(begin, end) + 1), :, 0]
    interval_force = raw_load[sample].reshape(-1, 5, n, len(LABELS))
    support = load_flags[sample].reshape(interval_force.shape)
    _, entered, passed, reasons, _, _ = replay(plan, folder, dict(chunks=chunks))
    assert int(passed.sum()) == rolled['score']['roll_complete']
    supine = (poses == spec['actual_release']) & entered
    groups = {}
    for label, selector in zip(spec['cohorts'], (passed, reasons == 'insufficient-ground-supported-rotation'), strict=True):
        mask = supine & selector
        values = describe(mask, phase, positive20, contact, amount, support, following, interval_force)
        recovery_cutoff = (spec['window']['roll_origin_control'] + 55 - first) * 5
        values.update(reload_pattern(phase & ~contact & (positive20 > 0), load_flags, begin * 5, recovery_cutoff, mask))
        groups[label] = values
    return dict(seed=plan['environment']['seed'], run=rolled['book']['run_id'], role=branch['role'] if branch else 'official-reference',
        first_capture_control=first, last_capture_control=last, clock_sha256=digest, supine_entrants=int(supine.sum()),
        worlds=n, bands=groups, quaternion_boundary_match=True, normal_force_threshold_n=clock['force_threshold_n'],
        maximum_derived_quaternion_step_error_rad=float(np.max(np.abs(broad - integrated))))


def main():
    file = ROOT / 'project.json'
    project = read(file)
    task = project['staged_training_request']
    state = project['goal_execution'][KEY]
    spec = read(ROOT / state['protocol'], state['protocol_sha256'])
    assert task['current_stage'] == spec['stage'] and state['status'] == 'frozen'
    all_bands = {}
    latest = None
    for key, source in spec['sources'].items():
        folder, book, plan, score = load_session(source)
        assert plan['composition']['roll_start_control'] == spec['window']['roll_origin_control']
        assert not (book['hardware_io'] or book['learning_updates'])
        if 'supine_support_window' in source:
            window = read(ROOT / source['supine_support_window'], source['supine_support_window_sha256'])
            assert window['source_collection_sha256'] == source['collection_sha256']
            assert window['control_window_ending'] == spec['window']['end_controls']
            assert window['actual_release'] == spec['actual_release'] and window['period_s'] == spec['window']['control_period_s']
            latest = dict(run=book['run_id'], seed=window['development_seed'],
                source_sha256=source['supine_support_window_sha256'], cohorts=window['metrics'])
            continue
        poses = release_poses(folder, book)
        if 'branches' not in book:
            label = f"official-recording-{plan['environment']['seed']}"
            all_bands[label] = analyze_branch(folder, None, book['chunks'], plan, dict(book=book, score=score), poses, spec)
        else:
            for role, summary in book['branches'].items():
                chunks = [dict(row, file='shared/' + row['file']) for row in book['prefix']['chunks']]
                chunks += [dict(row, file=role + '/' + row['file']) for row in summary['chunks']]
                all_bands[role] = analyze_branch(folder, dict(role=role, summary=summary), chunks, plan,
                    dict(book=book, score=score['results'][role]), poses, spec)
    assert latest is not None and len(all_bands) == 5
    out = ROOT / spec['record'] / 'analysis.json'
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, dict(protocol_sha256=state['protocol_sha256'], window=spec['window'], force_aligned_runs=all_bands,
        independent_paired_20ms_trace=latest, simulated_transitions=0, learning_updates=0, hardware_io=False,
        diagnostic_contact_offsets='Five positions within each 20 ms interval. Only the support bit changes and the same angle remains. This does not rescore the route.'))
    state.update(status='reviewed', evidence=out.relative_to(ROOT).as_posix(), evidence_sha256=sha(out))
    task.update(current_protocol_execution='support-sampling-audit-reviewed', current_evidence=state['evidence'])
    project.update(current_protocol_execution=task['current_protocol_execution'], current_evidence=state['evidence'])
    write(file, project)
    for name, run in all_bands.items():
        a, b = (run['bands'][key] for key in spec['cohorts'])
        print(name, a['worlds'], b['worlds'], [v['median_missed_positive_rotation_20ms_deg'] for v in (a,b)],
            [v['median_physical_unloaded_positive_rotation_4ms_deg'] for v in (a,b)], flush=True)
    print('NOMINAL_ROLL_SUPPORT_SAMPLING_AUDIT', sha(out), flush=True)


if __name__ == '__main__':
    main()
