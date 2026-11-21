"""Measure complete strides and attainable yaw margins from the verified paired traces."""

import numpy as np

from plot_forward_support_continuation import REPORT as SOURCE, ROOT, read, sha, write


REPORT = ROOT/'records/reports/forward-stride-yaw-response'
CASES = ('s1247-friction-low', 's1248-friction-low')
FORK, RESPONSE, CYCLES = 250, 150, 2


def stride_windows(feet, start, length, count):
    """Return windows from one single-foot onset to the same onset two strides later."""
    n = feet.shape[1]
    known = np.zeros(n, dtype=np.int8)
    entries = np.zeros((length+1, n), dtype=np.int8)
    for step, value in enumerate(feet[:start+length+1]):
        side = value[:, 0].astype(np.int8) + 2*value[:, 1].astype(np.int8)
        single = (side == 1) | (side == 2)
        if step >= start:
            entries[step-start] = np.where(single & (side != known), side, 0)
        known[single] = side[single]
    windows = np.full((n, 2, 2), -1, dtype=np.int32)
    for world in range(n):
        for side in (1, 2):
            edges = np.flatnonzero(entries[:, world] == side)
            if len(edges) >= count+1:
                windows[world, side-1] = [edges[0], edges[count]]
    return windows


def stride_mean(yaw, windows):
    n = windows.shape[0]
    means = np.full((n, 2), np.nan, dtype=np.float64)
    for world in range(n):
        for side in range(2):
            start, end = windows[world, side]
            if start >= 0:
                means[world, side] = np.abs(yaw[start+1:end+1, world]).mean()
    return means


def archived_measure(case):
    folder = ROOT/'records/evaluations'/case['run_id']
    source = folder/'guarded-evaluation.json'
    assert sha(source) == case['collection_sha256']
    receipt = read(source)
    ids = ('onset-only', 'right-support-continuation')
    paths = {'shared': (folder/'shared/trace.npz', receipt['prefix_sha256']['trace.npz'])}
    for role in ids:
        paths[role] = (folder/'clips'/role/'trace.npz', receipt['branches'][role]['trace_sha256']['trace.npz'])
    motion, proofs = {}, {}
    for role, (path, digest) in paths.items():
        assert sha(path) == digest
        proofs[path.relative_to(ROOT).as_posix()] = digest
        with np.load(path, allow_pickle=False) as data:
            motion[role] = (data['angular_velocity'][..., 2].astype(np.float64), data['foot_contact'])
    arms, measurements = {}, {}
    for role in ids:
        yaw, feet = (np.concatenate((prefix, tail[1:])) for prefix, tail in zip(motion['shared'], motion[role], strict=True))
        assert yaw.shape == feet.shape[:2] == (1051, 128)
        full = np.abs(yaw[51:]).mean(axis=0)
        np.testing.assert_allclose(np.median(full), case['arms'][role]['median_abs_yaw_rad_s'], atol=1e-6, rtol=0)
        window = stride_windows(feet, FORK, RESPONSE, CYCLES)
        in_full_cycle = stride_mean(yaw[FORK:FORK+RESPONSE+1], window)
        fixed = np.abs(yaw[FORK+1:FORK+RESPONSE+1]).mean(axis=0)
        # Given the real first four seconds, this is the remaining time-mean
        # allowed to reach 0.20 rad/s on the complete 20-second route.
        allowed = (.20*1000 - np.abs(yaw[51:FORK+1]).sum(axis=0))/(1050-FORK)
        complete = (window[..., 0] >= 0).all(axis=1)
        widths = window[..., 1]-window[..., 0]
        arms[role] = dict(complete_worlds=int(complete.sum()),
                         max_cycles_elapsed_controls=int(window[complete, :, 1].max()),
                         median_stride_duration_controls=(np.median(widths[complete], axis=0)/CYCLES).tolist(),
                         fixed_window_median_abs_yaw_rad_s=float(np.median(fixed)),
                         median_yaw_reduction_needed_rad_s=float(np.median(fixed-allowed)),
                         two_cycle_medians_rad_s=np.median(in_full_cycle[complete], axis=0).tolist(),
                         fixed_window_at_or_below_projected_route_limit=int((fixed <= allowed).sum()))
        measurements[role] = (fixed, in_full_cycle, complete)
    old, new = (measurements[role] for role in ids)
    both = old[2] & new[2]
    return dict(arms=arms, matched_complete_worlds=int(both.sum()),
                paired_fixed_window_delta_rad_s=float(np.median(new[0]-old[0])),
                paired_two_cycle_delta_by_start_side_rad_s=np.median(new[1][both]-old[1][both], axis=0).tolist(),
                source_sha256=proofs)


def main():
    source = SOURCE/'closed-loop-comparison.json'
    project = read(ROOT/'project.json')
    reference = project['goal_execution']['forward_right_support_continuation']
    assert sha(source) == reference['result_sha256'] and reference['status'] == 'reviewed'
    data = read(source)
    assert not data['candidate_passes_full_range']
    cases = {key: archived_measure(data['cases'][key]) for key in CASES}
    assert all(row['matched_complete_worlds'] == 128 for row in cases.values())
    REPORT.mkdir(exist_ok=True)
    write(REPORT/'archived-stride-coverage.json', dict(source_sha256=sha(source), cases=cases,
          fork_control=FORK, response_controls=RESPONSE, complete_cycles_per_anchor=CYCLES,
          transition_rule='A new left or right single-support run requires the preceding recognized single-support side to differ.',
          archive_only=True, physical_steps=0, learning_updates=0, heldout_used=False, hardware_io=False))
    for key, row in cases.items():
        print('ARCHIVED_STRIDE_COVERAGE', key, row['matched_complete_worlds'], row['paired_fixed_window_delta_rad_s'],
              row['paired_two_cycle_delta_by_start_side_rad_s'], row['arms']['onset-only']['median_yaw_reduction_needed_rad_s'])


if __name__ == '__main__':
    main()
