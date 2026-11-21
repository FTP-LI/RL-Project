"""Recompute saved first-episode metrics and decode every evaluation video.

No simulation or policy inference. Reads immutable per-condition traces rather
than trusting the summary's pass flag. Early failure and timeout classification
are independently derived from the first episode's last trace sample.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import imageio_ffmpeg
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    report = json.loads((root/'evaluation.json').read_text())
    assert report['status'] == 'completed'
    assert digest(root/'protocol.json') == report['protocol_sha256']
    assert digest(root/'evaluate_fixed_policy.py') == report['evaluator_sha256']
    assert digest(root/'evaluation_metrics.py') == report['metric_code_sha256']
    if report['phase'] == 'heldout':
        from heldout_evaluation import check_output
        check_output(Path(__file__).resolve().parents[2], root)
    protocol = json.loads((root/'protocol.json').read_text())['evaluation']
    assert [r['condition'] for r in report['conditions']] == list(protocol['commands'])
    checks = []
    for item in report['conditions']:
        condition = item['condition']
        path = root/f'{condition}-trace.npz'
        data = np.load(path)
        active = data['active']
        assert active.dtype == bool and active.shape[1] == report['num_envs']
        assert active[0].all() and not np.any(np.diff(active.astype(int), axis=0) > 0)
        count = active.sum(axis=0)
        assert np.all(count > 0)
        command = np.asarray(protocol['commands'][condition])
        for key in ('xy', 'velocity', 'yaw_rate', 'actions', 'foot_speed'):
            assert np.isfinite(data[key][active]).all(), f'{condition}: {key}'
        if 'observations' in data:
            assert np.isfinite(data['observations'][active]).all()
            np.testing.assert_allclose(data['observations'][:,:,48:51][active],
                                       np.broadcast_to(command,(active.sum(),3)), atol=1e-6)
            np.testing.assert_allclose(data['observations'][:,:,51:61][active], 0, atol=1e-6)
        delta = np.diff(data['actions'], axis=0, prepend=np.zeros_like(data['actions'][:1]))
        velocity_error = data['velocity']-command[:2]
        if report.get('linear_xy_error_definition') == 'time_mean_planar_l2_then_episode_mean':
            xy_error = np.linalg.norm(velocity_error, axis=2)
        else:
            xy_error = np.abs(velocity_error).mean(axis=2)  # Early preflight evidence only.
        xy = (xy_error*active).sum(axis=0)/count
        yaw = (np.abs(data['yaw_rate']-command[2])*active).sum(axis=0)/count
        rms = np.sqrt((np.square(delta).mean(axis=2)*active).sum(axis=0)/count)
        contact = data['contact']*active[:,:,None]
        contacts = contact.sum(axis=(0,2))
        slip = np.divide((data['foot_speed']*contact).sum(axis=(0,2)), contacts,
                         out=np.full(report['num_envs'], np.nan), where=contacts != 0)
        assert [r['steps'] for r in item['episodes']] == count.tolist()
        outcome = []
        horizon = round(report['episode_seconds']/report['step_dt'])
        for i, samples in enumerate(count):
            last = int(samples)-1
            if 'nan_state' in data and data['nan_state'][last,i]: reason='nan_state'
            elif data['fell_over'][last,i]: reason='fell_over'
            elif data['out_of_bounds'][last,i]: reason='out_of_terrain_bounds'
            elif data['terminated'][last,i]: reason='other_termination'
            elif samples < horizon: reason='early_timeout'
            else: reason='horizon_complete'
            outcome.append(reason)
            assert item['episodes'][i]['outcome'] == reason
        for key, value in [('linear_xy_mae_m_s',xy),('yaw_mae_rad_s',yaw),('action_delta_rms',rms)]:
            np.testing.assert_allclose([row[key] for row in item['episodes']], value, atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(item[key], np.mean(value), atol=1e-6, rtol=1e-5)
        if (contacts > 0).any():
            np.testing.assert_allclose(item['contact_foot_slip_m_s'], np.nanmean(slip), atol=1e-6, rtol=1e-5)
        assert item['fall_count_excluding_timeouts'] == outcome.count('fell_over')
        assert item['full_episode_survival'] == outcome.count('horizon_complete')/len(outcome)
        if 'initial_xy' in data:
            drift = np.linalg.norm(data['xy'][count-1,np.arange(len(count))]-data['initial_xy'], axis=1)
            np.testing.assert_allclose([row['final_displacement_m'] for row in item['episodes']], drift, atol=1e-6)
            np.testing.assert_allclose(item['final_displacement_m_median'], np.median(drift), atol=1e-6)
        limits = protocol['criteria_per_condition']
        criteria = (item['full_episode_survival'] >= limits['full_episode_survival_min']
                    and float(np.mean(xy)) <= limits['linear_xy_mae_max_m_s']
                    and float(np.mean(yaw)) <= limits['yaw_mae_max_rad_s']
                    and (condition != 'stop' or item['final_displacement_m_median'] <= limits['stop_drift_max_m_median']))
        assert bool(criteria) == item['meets_numeric_criteria']
        video = root/item['video']['path']
        assert digest(video) == item['video']['sha256']
        reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
        metadata = next(reader)
        frames = sum(1 for _ in reader)
        assert frames == len(item['video']['frame_times_seconds'])
        assert metadata['size'] == (960,720) and abs(metadata['fps']-25) < 1e-6
        checks.append(dict(condition=condition, episodes=len(outcome), trace_sha256=digest(path),
                           video_sha256=digest(video), decoded_frames=frames,
                           duration_seconds=metadata['duration'], observations_saved='observations' in data,
                           drift_recomputed='initial_xy' in data))
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), status='passed',
                  report_sha256=digest(root/'evaluation.json'),
                  scope='Independent saved-trace metric recomputation and full video decode; no hardware claims',
                  run_id=report['run_id'], checks=checks)
    eligible = all(v.get('formal_acceptance_eligible', report['phase']=='reference') for v in report['conditions'])
    assert report['meets_all_reference_criteria'] == (eligible and all(v['meets_numeric_criteria'] for v in report['conditions']))
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
