"""Separate sustained yaw from oscillation using development walks only."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT / 'records/reports/common-walk-confirmation'
DT = .02
DEVELOPMENT = (1040, 1042, 1043)


def measure(data, command):
    active = data['active']
    assert active.shape == (1000, 32) and active.all()
    error = data['yaw_rate'].astype(np.float64) - command
    after = error[50:]
    bias = after.mean(axis=0)
    centered = after - bias
    fft = abs(np.fft.rfft(centered * np.hanning(len(after))[:, None], axis=0))
    peak = 1 + fft[1:].argmax(axis=0)
    gyro = data['observations'][51:, :, 2].astype(np.float64)
    aligned_yaw = data['yaw_rate'][50:-1].astype(np.float64)
    gyro_error = gyro - aligned_yaw
    return dict(mae=np.abs(error).mean(axis=0), late_mae=np.abs(after).mean(axis=0), late_bias=bias,
                late_centered_mae=np.abs(centered).mean(axis=0), peak_hz=peak / (DT * len(after)),
                gyro_aligned_rms=np.sqrt(np.mean(gyro_error ** 2, axis=0)))


def main():
    batches, sources = {}, {}
    for seed in DEVELOPMENT:
        proof_path = (OUT if seed == 1040 else OUT / 'reset-recovery') / f's{seed}/analysis.json'
        proof = read(proof_path)
        assert proof['status'] == 'passed' and not proof['heldout_used']
        directory = ROOT / proof['evaluator']
        plan, report = (directory / name for name in ('protocol.json', 'evaluation.json'))
        config, logged = read(plan), read(report)
        assert config['seed'] == seed and not config['heldout_used'] and logged['status'] == 'completed'
        for path in (proof_path, plan, report):
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
        batches[str(seed)] = {}
        for condition in ('stand', 'forward', 'backward'):
            path = directory / f'{condition}-trace.npz'
            condition_log = next(row for row in logged['conditions'] if row['condition'] == condition)
            digest = sha(path)
            assert digest == condition_log['trace_sha256']
            sources[path.relative_to(ROOT).as_posix()] = digest
            with np.load(path) as trace:
                statistics = measure(trace, config['commands'][condition][2])
            original = next(row for row in proof['rows'] if row['condition'] == condition)
            np.testing.assert_allclose(statistics['mae'].mean(), original['yaw_error'], rtol=1e-5, atol=1e-6)
            indices = config['groups'][condition]
            summary = {k: float(np.median(v)) for k, v in statistics.items()}
            summary['absolute_bias'] = float(np.median(abs(statistics['late_bias'])))
            high = np.flatnonzero(statistics['mae'] >= .18)
            rows = [dict(world=world, **{key: float(value[i]) for key, value in statistics.items()})
                    for i, world in enumerate(indices)]
            batches[str(seed)][condition] = dict(median=summary, high_yaw_cases=[rows[i] for i in high], rows=rows)
        forward = batches[str(seed)]['forward']
        print('WALK_YAW', seed, forward['median'], 'high', forward['high_yaw_cases'], flush=True)
    write(OUT / 'development-yaw-review.json', dict(status='passed-saved-development-data-only', cohorts=batches,
        source_sha256=sources, script_sha256=sha(Path(__file__)), new_physical_steps=0, learning_updates=0,
        heldout_used=False, original_scores_unchanged=True,
        method='Analyze all original yaw errors. Remove each post-first-second mean to quantify oscillation. Demeaning is descriptive, not a predicted feedback result.'))


if __name__ == '__main__':
    main()
