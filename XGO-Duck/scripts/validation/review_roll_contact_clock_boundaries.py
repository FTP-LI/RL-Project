"""Compare the saved coarse sensor and solved force at the same pose."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT/'records/reports/roll-contact-clock'


def summarize(sensor, native):
    return dict(total=int(sensor.size), agreement=int((sensor == native).sum()),
                sensor_off_native_on=int((~sensor & native).sum()),
                sensor_on_native_off=int((sensor & ~native).sum()))


def main():
    proof_path, coordinator = OUT/'analysis.json', read(OUT/'coordinator.json')
    result = read(proof_path)
    assert coordinator['status'] == 'completed' and coordinator['analysis_sha256'] == sha(proof_path)
    assert result['status'] == 'passed-independent-diagnostic-verification' and not result['heldout_used']
    folder = ROOT/result['record']
    trace_path, clock_path = folder/'forward-roll-trace.npz', folder/'contact-clock.npz'
    assert sha(trace_path) == result['trace_sha256'] and sha(clock_path) == result['substeps_sha256']
    with np.load(trace_path) as data:
        controls = {key: data[key] for key in ('qpos', 'ground_contact', 'head_contact', 'foot_contact', 'active')}
    with np.load(clock_path) as data:
        samples = {key: data[key] for key in data.files}
    plan = read(folder/'protocol.json')
    n = round(.02/.004)
    indices = np.flatnonzero((samples['substep']-1) % n == 0)
    k = (samples['substep'][indices]-1)//n
    assert len(indices) == 60 and np.array_equal(k, np.arange(10, 70))
    np.testing.assert_array_equal(samples['before'][indices], controls['qpos'][k, :, 3:7])
    active = controls['active'][k]
    threshold = result['force_threshold_n']
    sensor = np.stack((controls['ground_contact'][k], controls['head_contact'][k], controls['foot_contact'][k].any(-1)), -1)
    points, loads = (samples[key] for key in ('points', 'normal_force_n'))

    def any_feet(values):
        return np.stack((values[..., 0], values[..., 1], values[..., 2] | values[..., 3]), -1)

    candidate = any_feet(points[indices] > 0)
    loaded = any_feet(loads[indices] > threshold)
    order = ('ground', 'head', 'either_foot')
    channels = {key: dict(points=summarize(sensor[..., i][active], candidate[..., i][active]),
                          load=summarize(sensor[..., i][active], loaded[..., i][active]))
                for i, key in enumerate(order)}
    same_pose = dict(zip(k.tolist(), indices.tolist(), strict=True))
    gaps = []
    for row in result['rows']:
        if not row['covered']:
            continue
        world, gap = row['world'], row['gap']
        begin = round(gap['start_seconds']/.02)
        endpoints = range(begin, begin+gap['air_steps'])
        at_end = [same_pose[end] for end in endpoints if end in same_pose]
        raw_start = same_pose[begin-1]
        raw_gaps = loads[raw_start:raw_start+n*gap['air_steps'], world, 0]
        np.testing.assert_array_equal(controls['ground_contact'][list(endpoints), world], False)
        assert int((raw_gaps > threshold).sum()) == row['loaded_substeps']
        assert len(raw_gaps) == row['physics_samples']
        gaps.append(dict(world=world, success=row['success'], control_begin=begin,
            endpoint_count=len(at_end), endpoint_force_n=loads[at_end, world, 0].tolist(),
            endpoint_points=points[at_end, world, 0].tolist(),
            endpoint_load_on=int((loads[at_end, world, 0] > threshold).sum()),
            force_at_each_control_offset_n=raw_gaps.reshape(-1, n).tolist()))
    groups = {}
    for label, success in (('success', True), ('failure', False)):
        rows = [row for row in gaps if row['success'] == success]
        matrix = np.concatenate([np.asarray(row['force_at_each_control_offset_n']) for row in rows], axis=0)
        groups[label] = dict(worlds=len(rows), control_intervals=len(matrix),
            aligned_coarse_off_boundaries=sum(row['endpoint_count'] for row in rows),
            aligned_coarse_off_native_loaded_boundaries=sum(row['endpoint_load_on'] for row in rows),
            loaded_counts_by_offset=(matrix > threshold).sum(axis=0).tolist(),
            median_force_by_offset_n=np.median(matrix, axis=0).tolist())
    check = dict(status='passed-saved-data-only', clock_analysis_sha256=sha(proof_path),
        trace_sha256=result['trace_sha256'], substeps_sha256=result['substeps_sha256'],
        script_sha256=sha(Path(__file__)), seed=plan['seed'], same_pose_controls=len(k),
        contact_channels=channels, coarse_gaps=groups, rows=gaps, force_threshold_n=threshold,
        new_physical_steps=0, learning_updates=0, heldout_used=False,
        alignment='A coarse endpoint k shares qpos with the next substep solve before integration. The following command can change solver force at the same pose.')
    target = OUT/'boundary-review.json'
    assert not target.exists()
    write(target, check)
    print('ROLL_CONTACT_CLOCK_BOUNDARIES', channels, groups, flush=True)


if __name__ == '__main__':
    main()
