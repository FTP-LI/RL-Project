"""Check whether measured joint phase leads the forward yaw oscillation."""
from pathlib import Path

import numpy as np

from review_common_walk_yaw import DEVELOPMENT, DT, OUT
from s1_reference_inputs import ROOT, read, sha
from record_io import write


LEADS = (1, 2, 3, 4, 5, 6)
PHASE_JOINTS = ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle')
MIN_MEDIAN = .4
MIN_WITH_SAME_SIGN = 24


def correlation(a, b):
    a, b = (x.astype(np.float64) - x.mean(axis=0) for x in (a, b))
    magnitude = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return np.divide((a * b).sum(axis=0), magnitude, where=magnitude > 0,
                     out=np.zeros_like(magnitude))


def signals(trace, plan):
    start = np.arange(50, 1000 - max(LEADS) + 1)
    assert trace['active'].shape == (1000, 32) and trace['active'].all()
    obs = trace['observations'][start]
    result = {'gyro_z': obs[..., 2]}
    feet = trace['contact'][start - 1].astype(np.float32)
    result['sim_feet_difference'] = feet[..., 0] - feet[..., 1]
    for joint in PHASE_JOINTS:
        left, right = (obs[..., 20 + plan['joint_names'].index(f'{side}_{joint}')] for side in ('left', 'right'))
        result.update({f'left_{joint}': left, f'right_{joint}': right,
                       f'sum_{joint}': left + right, f'difference_{joint}': left - right})
    return {(name, lead): correlation(value, trace['yaw_rate'][start + lead - 1])
            for name, value in result.items() for lead in LEADS}


def source(seed, previous):
    review = read((OUT if seed == 1040 else OUT / 'reset-recovery') / f's{seed}/analysis.json')
    assert review['status'] == 'passed' and not review['heldout_used']
    folder = ROOT / review['evaluator']
    plan_file, trace_file = folder / 'protocol.json', folder / 'forward-trace.npz'
    plan = read(plan_file)
    assert plan['seed'] == seed and not plan['heldout_used']
    for path in (plan_file, trace_file):
        assert sha(path) == previous['source_sha256'][path.relative_to(ROOT).as_posix()]
    with np.load(trace_file) as trace:
        return signals(trace, plan)


def measured(series, polarity):
    return dict(signed_median=float(np.median(series)),
                polarity_adjusted_median=float(np.median(polarity * series)),
                same_sign_worlds=int((polarity * series > 0).sum()), correlation_by_world=series.tolist())


def main():
    previous_path = OUT / 'development-yaw-review.json'
    previous = read(previous_path)
    assert previous['status'] == 'passed-saved-development-data-only' and not previous['heldout_used']
    batches = {seed: source(seed, previous) for seed in DEVELOPMENT}
    first = batches[DEVELOPMENT[0]]
    selected = max((key for key in first if key[0] not in ('gyro_z', 'sim_feet_difference')),
                   key=lambda key: abs(np.median(first[key])))
    polarity = int(np.sign(np.median(first[selected])))
    assert polarity in (-1, 1)
    rows = {}
    for seed, batch in batches.items():
        rows[str(seed)] = dict(encoder=measured(batch[selected], polarity),
            gyro_same_lead_median=float(np.median(batch['gyro_z', selected[1]])),
            simulated_stance_same_lead_median=float(np.median(batch['sim_feet_difference', selected[1]])))
    replicated = all(r['encoder']['polarity_adjusted_median'] >= MIN_MEDIAN and
                     r['encoder']['same_sign_worlds'] >= MIN_WITH_SAME_SIGN for r in rows.values())
    write(OUT / 'development-phase-review.json', dict(status='passed-saved-development-data-only',
        encoder_channel=selected[0], lead_controls=selected[1], lead_seconds=selected[1] * DT, polarity=polarity,
        discovery_seed=DEVELOPMENT[0], replication_development_seeds=list(DEVELOPMENT[1:]), cohorts=rows,
        minimum_median=MIN_MEDIAN, minimum_same_sign_worlds=MIN_WITH_SAME_SIGN, association_replicated=replicated,
        scope='Observed encoder velocity before the control interval versus subsequent physical yaw. Correlation does not measure a command-to-yaw response.',
        yaw_source_sha256=sha(previous_path), script_sha256=sha(Path(__file__)), new_physical_steps=0,
        policy_learning_updates=0, heldout_used=False))
    print('WALK_PHASE', selected, polarity, replicated, {seed: {key: value for key, value in row['encoder'].items()
          if key != 'correlation_by_world'} for seed, row in rows.items()}, flush=True)


if __name__ == '__main__':
    main()
