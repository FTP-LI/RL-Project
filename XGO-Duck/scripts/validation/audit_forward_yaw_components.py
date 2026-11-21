"""Describe sustained yaw and support-phase yaw in archived official forward runs."""

from pathlib import Path

import numpy as np

import run_roll_stand_forward_diagnostic as direct


OUTPUT = direct.ROOT / 'records/reports/forward-output-layer-pair/official-yaw-components.json'


def summarize(case, mark, sources):
    root = direct.ROOT
    receipt = root / mark['path']
    result = direct.read(receipt)
    run = root / 'records/evaluations' / result['run_id']
    trace, score, manifest = (run / name for name in ('trace.npz', 'raw-score.json', 'guarded-evaluation.json'))
    source_hashes = {receipt: mark['sha256'], trace: direct.read(manifest)['trace_sha256'], score: result['raw_score_sha256']}
    direct.require(result['full_route'] and not any(result['safety'].values()) and not result['holdout_used']
                   and all(direct.sha(path) == digest for path, digest in source_hashes.items()), 'Official archive changed')
    sources.update({direct.rel(path): digest for path, digest in source_hashes.items()})
    source = direct.read(score)
    direct.require(set(source['forward_at']) == {50} and len(source['forward_at']) == 128, 'Unexpected walking interval')
    with np.load(trace, allow_pickle=False) as states:
        time = slice(51, 1051)
        yaw = states['angular_velocity'][time, :, 2]
        contacts = states['foot_contact'][time]
        np.testing.assert_array_equal(states['active'][time], True)
    np.testing.assert_allclose(np.abs(yaw).mean(axis=0), [row['yaw_error_rad_s'] for row in source['forward']],
                               rtol=0, atol=1e-6)
    yaw = yaw.astype(np.float64)
    mean = yaw.mean(axis=0)
    centered = np.abs(yaw-mean).mean(axis=0)
    left, right = contacts[..., 0], contacts[..., 1]  # Same site order as the route collector.
    phase = {}
    for name, mask in (('left_only', left & ~right), ('right_only', right & ~left),
                       ('double_support', left & right), ('no_support', ~left & ~right)):
        count = mask.sum(axis=0)
        valid = count > 0
        phase[name] = dict(active_worlds=int(valid.sum()), sample_fraction=float(mask.mean()),
                           median_world_signed_yaw_rad_s=float(np.median((yaw*mask).sum(axis=0)[valid]/count[valid])) if valid.any() else None,
                           median_world_abs_yaw_rad_s=float(np.median((np.abs(yaw)*mask).sum(axis=0)[valid]/count[valid])) if valid.any() else None)
    return dict(case=case, run_id=run.name, mean_abs_yaw_world_median_rad_s=float(np.median(np.abs(yaw).mean(axis=0))),
                absolute_signed_mean_world_median_rad_s=float(np.median(np.abs(mean))),
                centered_mean_abs_yaw_world_median_rad_s=float(np.median(centered)),
                centered_below_original_yaw_threshold=int((centered <= .2).sum()), support=phase)


def main():
    project = direct.read(direct.P)
    reference = project['goal_execution']['roll_friction_stand_forward']
    direct.require(reference['status'] == 'reviewed' and len(reference['scored']) == 4, 'Expected two official paired seeds')
    sources = {direct.rel(Path(__file__).resolve()): direct.sha(Path(__file__).resolve())}
    data = {case: summarize(case, mark, sources) for case, mark in reference['scored'].items()}
    direct.put(OUTPUT, dict(samples=512000, world_windows=512, window_seconds=20, data=data, sources=sources,
                           method='Describe the observed trajectories after subtracting each world mean; no control intervention.',
                           physics_steps=0, learning_updates=0, heldout_used=False, hardware_io=False))
    print('OFFICIAL_FORWARD_YAW_COMPONENTS', {case: round(row['centered_mean_abs_yaw_world_median_rad_s'], 5)
          for case, row in data.items()}, flush=True)


if __name__ == '__main__':
    main()
