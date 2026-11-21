"""Check causal yaw forecasts on archived development forward walks."""
from pathlib import Path

import numpy as np

from review_common_walk_yaw import DEVELOPMENT, OUT
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def load(seed, hashes):
    proof = read((OUT if seed == DEVELOPMENT[0] else OUT/'reset-recovery')/f's{seed}/analysis.json')
    assert proof['status'] == 'passed' and not proof['heldout_used']
    record = ROOT/proof['evaluator']
    config, archive = record/'protocol.json', record/'forward-trace.npz'
    for source in (config, archive):
        assert sha(source) == hashes[source.relative_to(ROOT).as_posix()]
    plan = read(config)
    assert plan['seed'] == seed and not plan['heldout_used'] and plan['commands']['forward'][2] == 0
    indices = [20+plan['joint_names'].index(f'{side}_hip_pitch') for side in ('left', 'right')]
    with np.load(archive) as trace:
        assert trace['active'].shape == (1000, 32) and trace['active'].all()
        obs, yaw = trace['observations'].astype(np.float64), trace['yaw_rate'].astype(np.float64)
    sensors = np.stack((obs[..., 2], obs[..., indices].sum(axis=-1)), axis=-1)
    # Each input is read before that control is applied. Do not look ahead
    # in either observed sensor when forecasting the first physical response.
    change = sensors[50:998]-sensors[49:997]
    x = np.concatenate((sensors[50:998], change), axis=-1)
    return x, (yaw[50:998], yaw[52:1000])


def fit(x, y, columns):
    x = x[..., columns].reshape(-1, len(columns))
    mean, scale = x.mean(axis=0), x.std(axis=0)
    assert np.all(scale > 0)
    z = (x-mean)/scale
    target = y.flatten()
    center = float(target.mean())
    reg = .001*len(z)*np.eye(z.shape[1])
    coef = np.linalg.solve(z.T@z+reg, z.T@(target-center))
    return dict(columns=list(columns), means=mean.tolist(), scales=scale.tolist(), weights=coef.tolist(), bias=center)


def predict(x, model):
    return ((x[..., model['columns']]-model['means'])/model['scales'])@model['weights']+model['bias']


def main():
    prior = OUT/'development-yaw-review.json'
    saved = read(prior)
    assert saved['status'] == 'passed-saved-development-data-only' and not saved['heldout_used']
    series = {seed: load(seed, saved['source_sha256']) for seed in DEVELOPMENT}
    train, train_yaw = series[DEVELOPMENT[0]]
    candidates = {'gyro': (0, 2), 'hips': (1, 3), 'gyro_and_hips': (0, 1, 2, 3)}
    results = {}
    for j, horizon in enumerate((.02, .06)):
        models = {key: fit(train, train_yaw[j], columns) for key, columns in candidates.items()}
        cohorts = {}
        for seed in DEVELOPMENT[1:]:
            x, y = series[seed]
            target = y[j]
            errors = {name: np.abs(predict(x, model)-target).mean(axis=0) for name, model in models.items()}
            errors['zero'] = np.abs(target).mean(axis=0)
            gain = errors['gyro']-errors['gyro_and_hips']
            cohorts[str(seed)] = dict(mae_rad_s={k: float(v.mean()) for k, v in errors.items()},
                                      paired_mean_gain_over_gyro_rad_s=float(gain.mean()),
                                      improved_worlds=int((gain > 0).sum()))
        results[str(horizon)] = dict(models=models, cohorts=cohorts)
    short = results[str(.02)]['cohorts']
    admission = all(row['paired_mean_gain_over_gyro_rad_s'] >= .01 and row['improved_worlds'] >= 24
                    for row in short.values())
    out = OUT/'development-yaw-forecast.json'
    write(out, dict(status='passed-saved-development-data-only', source_sha256=sha(prior),
                   script_sha256=sha(Path(__file__)), fit_seed=DEVELOPMENT[0],
                   check_seeds=list(DEVELOPMENT[1:]), min_gain_rad_s=.01, min_improved_worlds=24,
                   results=results, combined_20ms_predictor_nominated=admission,
                   physical_substeps=0, learning_updates=0, heldout_used=False,
                   scope='Forecast diagnostics on saved trajectories; command efficacy requires a new paired physical trial.'))
    print('WALK_YAW_FORECAST', short, 'nominated', admission, flush=True)


if __name__ == '__main__':
    main()
