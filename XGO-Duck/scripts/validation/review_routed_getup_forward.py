"""Localize the forward failures using only the recorded continuous development run."""

from collections import Counter
from pathlib import Path

import numpy as np

import run_routed_getup_sequence as study
from standup_metrics import projected_gravity


def first(mask):
    return np.where(mask.any(axis=0), mask.argmax(axis=0), -1)


def describe(values):
    return dict(median=float(np.median(values)), p90=float(np.quantile(values, .9)))


def main():
    out, read, sha, write = study.OUT, study.read, study.sha, study.write
    source = out / 'analysis.json'
    score, spec = read(source), read(study.PLAN)
    assert score['protocol_sha256'] == sha(study.PLAN) and not score['heldout_used']
    assert not score['score']['passed_development_gate']
    record = study.ROOT / score['record']
    run = read(record / 'evaluation.json')
    rows = score['score']['forward_worlds']
    entered = np.array([item['entered_forward'] for item in rows])
    passed = np.array([item['success'] for item in rows])
    failures = entered & ~passed
    yaw_only = np.array([item['failed_checks'] == ['yaw_error'] for item in rows])
    assert all('yaw_error' in rows[i]['failed_checks'] for i in np.flatnonzero(failures))
    forward_at = spec['sequence']['boundary_controls'][-1]
    walking = spec['walking']['criteria']
    pose = walking['posture']
    out_file = out / 'forward-failure-review.json'
    assert not out_file.exists()
    parts = {name: [] for name in ('yaw', 'tilt', 'height', 'speed', 'feedback')}
    entry = {}
    for chunk in run['record_chunks']:
        if chunk['last'] < forward_at:
            continue
        path = record / chunk['file']
        assert sha(path) == chunk['sha256']
        with np.load(path, allow_pickle=False) as saved:
            k = saved['control']
            q = saved['qpos']
            ang = saved['angular_velocity']
            if forward_at in k:
                i = int(np.flatnonzero(k == forward_at)[0])
                entry['tilt_deg'] = np.rad2deg(np.arccos(np.clip(-projected_gravity(q[i, :, 3:7])[:, 2], -1., 1.)))
                entry['angular_speed_rad_s'] = np.linalg.norm(ang[i], axis=1)
                entry['height_m'] = q[i, :, 2].copy()
                entry['previous'] = saved['applied'][i].copy()
            if forward_at + 1 in k:
                i = int(np.flatnonzero(k == forward_at + 1)[0])
                entry['unlimited_request_jump_rad'] = abs(saved['requested'][i]-entry['previous']).max(axis=1)
            forward = k > forward_at
            if not forward.any():
                continue
            parts['yaw'].append(abs(ang[forward, :, 2]))
            parts['height'].append(q[forward, :, 2])
            parts['tilt'].append(np.rad2deg(np.arccos(np.clip(-projected_gravity(q[forward, :, 3:7])[..., 2], -1., 1.))))
            parts['speed'].append(saved['velocity'][forward, :, 0])
            parts['feedback'].append(abs(saved['feedback_yaw'][forward]))
    data = {key: np.concatenate(pieces) for key, pieces in parts.items()}
    worlds = len(rows)
    assert all(value.shape == (1000, worlds) and np.isfinite(value).all() for value in data.values())
    posture = ((data['tilt'] > pose['tilt_max_deg']) |
               (abs(data['height'] - pose['nominal_height_m']) > pose['height_tolerance_m']))
    block = lambda array: array.reshape(20, 50, worlds).mean(axis=1)
    yaw, bad, speed, feedback = map(block, (data['yaw'], posture, data['speed'], data['feedback']))
    first_yaw = first(yaw > walking['yaw_error_max'])
    first_bad = first(bad > pose['maximum_bad_sample_fraction'])
    fallen = (data['tilt'] > 45.) | (data['height'] < pose['nominal_height_m'] - pose['height_tolerance_m'])
    prefix = np.concatenate((np.zeros((1, worlds), np.int16), np.cumsum(fallen, axis=0, dtype=np.int16)))
    first_fallen = first(prefix[25:] - prefix[:-25] == 25)
    groups = {}
    for name, mask in (('pass', passed), ('yaw-only', failures & yaw_only), ('yaw-and-other', failures & ~yaw_only)):
        ids = np.flatnonzero(mask)
        if not len(ids):
            continue
        groups[name] = dict(worlds=len(ids), entry={key: describe(values[ids]) for key, values in entry.items() if key != 'previous'},
            first_yaw_overlimit_block=dict(Counter(str(value+1) if value >= 0 else 'none' for value in first_yaw[ids])),
            first_bad_posture_block=dict(Counter(str(value+1) if value >= 0 else 'none' for value in first_bad[ids])),
            severe_posture_worlds=int((first_fallen[ids] >= 0).sum()),
            median_severe_posture_onset_seconds=(float(np.median(first_fallen[ids][first_fallen[ids] >= 0] / 50.))
                if (first_fallen[ids] >= 0).any() else None),
            yaw_before_bad=int(((first_yaw[ids] >= 0) & (first_bad[ids] >= 0) & (first_yaw[ids] < first_bad[ids])).sum()),
            bad_before_yaw=int(((first_yaw[ids] >= 0) & (first_bad[ids] >= 0) & (first_bad[ids] < first_yaw[ids])).sum()),
            median_by_second={key: np.median(values[:, ids], axis=1).tolist()
                              for key, values in (('yaw_rad_s', yaw), ('forward_speed_m_s', speed),
                                                  ('bad_posture_fraction', bad), ('feedback_rad_s', feedback))})
    write(out_file, dict(status='completed-forward-development-localization', analysis_sha256=sha(source),
        protocol_sha256=sha(study.PLAN), script_sha256=sha(Path(__file__)), entered=int(entered.sum()),
        entered_failed=int(failures.sum()), groups=groups,
        scope='One-second localization windows, separate from the frozen full-episode score',
        learning_updates=0, physics_substeps=0, heldout_used=False))
    print('ROUTED_FORWARD_REVIEW', {name: (group['worlds'], group['severe_posture_worlds'],
        group['median_severe_posture_onset_seconds'], group['yaw_before_bad'], group['bad_before_yaw'])
        for name, group in groups.items()}, flush=True)


if __name__ == '__main__':
    main()
