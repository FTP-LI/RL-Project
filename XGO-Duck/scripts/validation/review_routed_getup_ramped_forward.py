"""Read back the admitted startup window and explain the full run throughput."""

from collections import Counter

import numpy as np

from check_continuous_lateral_push_geometry import ROOT, load, sha
from standup_metrics import projected_gravity


OUT = ROOT / 'records/reports/continuous-routed-getup-ramped-forward'
PLAN = ROOT / 'configs/experiments/continuous_routed_getup_ramped_forward.json'


def main():
    import json

    spec = load(PLAN)
    record = ROOT / spec['record']
    state_file = record / 'evaluation.json'
    state = load(state_file)
    assert state['status'] == 'completed' and state['protocol_sha256'] == sha(PLAN)
    d = spec['ramped_forward']
    paired = ROOT / d['admitted_pair'][0]
    assert sha(paired) == d['admitted_pair'][1]
    gate = load(paired)['pair']['diagnostic_gate']
    k = d['boundaries'][-2]
    chunk, = [item for item in state['record_chunks'] if item['first'] == k + 1]
    path = record / chunk['file']
    assert sha(path) == chunk['sha256'] and chunk['last'] >= k + 100
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['control'][:100], np.arange(k+1, k+101))
        rows = dict(qpos=saved['qpos'][:100], angular=saved['angular_velocity'][:100],
            velocity=saved['velocity'][:100], mode=saved['mode'][:100])
    mode = rows['mode']
    entered = mode[0] == 2
    assert np.all(mode == mode[0]) and int(entered.sum()) == state['forward_worlds']
    pos = rows['qpos']
    gravity = projected_gravity(pos[..., 3:7])
    angle = np.rad2deg(np.arccos(np.clip(-gravity[..., 2], -1., 1.)))
    severe = (angle > gate['severe_tilt_deg']) | (pos[..., 2] < gate['severe_height_min_m'])
    sustained = np.lib.stride_tricks.sliding_window_view(severe, gate['sustained_failure_controls'], axis=0).all(axis=-1)
    startup = dict(worlds=int(entered.sum()), sustained_severe=int((sustained.any(axis=0)&entered).sum()),
        mean_absolute_yaw_rad_s=float(abs(rows['angular'][:, entered, 2]).mean(dtype=np.float64)),
        second_second_forward_speed_m_s=float(rows['velocity'][50:, entered, 0].mean(dtype=np.float64)))

    updates = state['updates']
    size = spec['sequence']['record_window_controls']
    assert [r['control'] for r in updates] == list(range(50, d['maximum_unique_controls']+1, 50))
    # The sampler publishes each 50-step rate just before flushing a full
    # chunk. The next window includes that preceding compression and write.
    write_windows = [r['transitions_per_second'] for r in updates if r['control'] > 50 and r['control'] % size == 50]
    other_windows = [r['transitions_per_second'] for r in updates if r['control'] % size != 50]
    telemetry = load(record / 'gpu-telemetry.json')
    analysis_path = OUT / 'analysis.json'
    analysis = load(analysis_path)
    assert analysis['protocol_sha256'] == sha(PLAN) and analysis['score']['forward_entries'] == startup['worlds']
    forward = analysis['score']['forward_worlds']
    failures = [row for row in forward if row['entered_forward'] and not row['success']]
    failure_groups = Counter(', '.join(row['failed_checks']) for row in failures)
    assert len(failures) == startup['worlds'] - analysis['score']['forward_passes']
    yaw_errors = [row['yaw_error_rad_s'] for row in failures if row['yaw_error_rad_s'] is not None]
    residual = dict(entered_failures=len(failures), failed_check_combinations=dict(failure_groups),
        failure_median_yaw_rad_s=float(np.median(yaw_errors)) if yaw_errors else None)
    output = OUT / 'startup-throughput-review.json'
    assert not output.exists()
    result = dict(status='verified-from-complete-trajectory', startup=startup, remaining_forward_failures=residual,
        net_transitions_per_second=state['actual_control_transitions']/state['sampling_wall_seconds'],
        median_rate_windows_after_chunk_write=float(np.median(write_windows)),
        median_rate_other_windows=float(np.median(other_windows)),
        peak_gpu_memory_mib=state['gpu']['peak_mib'],
        mean_gpu_utilization_percent=float(np.mean([row['gpu_percent'] for row in telemetry])),
        phase_rates_scope='Wall-clock windows, measured by the sampler. Includes rendering and processing; not a timed decomposition.',
        protocol_sha256=sha(PLAN), analysis_sha256=sha(analysis_path), evaluation_sha256=sha(state_file),
        early_trace_sha256=chunk['sha256'], physical_substeps=0)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('ROUTED_FULL_STARTUP', startup, result['net_transitions_per_second'], flush=True)


if __name__ == '__main__':
    main()
