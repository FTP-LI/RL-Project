"""Screen one rebound-aware feedback rule on saved development trajectories."""
from pathlib import Path

import numpy as np

from review_common_walk_yaw import DEVELOPMENT, OUT as WALK
from review_common_walk_yaw_forecast import fit, predict
from review_walk_yaw_pulse_capacity import OUT, apply_kernel
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def fixture(seed, hashes):
    directory = WALK if seed == DEVELOPMENT[0] else WALK/'reset-recovery'
    proof = read(directory/f's{seed}/analysis.json')
    assert not proof['heldout_used'] and proof['status'] == 'passed'
    record = ROOT/proof['evaluator']
    for file in (record/'protocol.json', record/'forward-trace.npz'):
        assert sha(file) == hashes[file.relative_to(ROOT).as_posix()]
    plan = read(record/'protocol.json')
    assert plan['seed'] == seed and not plan['heldout_used'] and plan['commands']['forward'][2] == 0
    indices = [20+plan['joint_names'].index(f'{side}_hip_pitch') for side in ('left', 'right')]
    with np.load(record/'forward-trace.npz') as archive:
        assert archive['active'].shape == (1000, 32) and archive['active'].all()
        obs = archive['observations'].astype(np.float64)
        yaw = archive['yaw_rate'].astype(np.float64)
    sensors = np.stack((obs[..., 2], obs[..., indices].sum(axis=-1)), axis=-1)
    current = np.arange(50, 995)
    features = np.concatenate((sensors[current], sensors[current]-sensors[current-1]), axis=-1)
    targets = np.stack([yaw[current+lag] for lag in range(6)], axis=-1)
    return features, targets


def main():
    capacity_path = OUT/'saved-feedback-capacity.json'
    capacity = read(capacity_path)
    pulse = OUT/'analysis.json'
    assert not capacity['heldout_used'] and not capacity['current_cap_reaches_gate_in_model']
    assert capacity['pulse_sha256'] == sha(pulse)
    source = WALK/'development-yaw-review.json'
    history = read(source)
    assert not history['heldout_used'] and history['status'] == 'passed-saved-development-data-only'
    x, target = fixture(DEVELOPMENT[0], history['source_sha256'])
    models = [fit(x, np.sign(target[..., i]), (0, 1, 2, 3)) for i in range(6)]
    first = capacity['cohorts'][str(DEVELOPMENT[1])]
    kernels = {name: np.asarray(row['kernel']) for name, row in first.items()}
    mean_kernel = np.mean(list(kernels.values()), axis=0)

    def gradient(observed):
        # Weight predicted future signs by the measured response of one
        # command. A later opposite response subtracts from its first benefit.
        signs = np.clip(np.stack([predict(observed, model) for model in models], axis=-1), -1., 1.)
        return signs@mean_kernel

    scale = float(np.quantile(np.abs(gradient(x)), .9))
    assert scale > 0
    results = {}
    for seed in DEVELOPMENT[1:]:
        x, target = fixture(seed, history['source_sha256'])
        direction = -np.clip(gradient(x)/scale, -1., 1.)
        original = target[..., 0]
        rows = {}
        for cap in (.12, .18):
            rows[str(cap)] = {name: float((np.abs(original)-np.abs(
                original+apply_kernel(cap*direction, kernel))).mean()) for name, kernel in kernels.items()}
        results[str(seed)] = rows
    gate = capacity['tracking_gate_rad_s']
    passed = [cap for cap in (.12, .18) if all(
        gain >= gate for rows in results.values() for gain in rows[str(cap)].values())]
    write(OUT/'rebound-feedback-screen.json', dict(status='passed-saved-linear-response-check',
          script_sha256=sha(Path(__file__)), capacity_sha256=sha(capacity_path), development_source_sha256=sha(source),
          fit_seed=DEVELOPMENT[0], check_seeds=list(DEVELOPMENT[1:]), response_kernel=mean_kernel.tolist(),
          sign_predictors=models, gradient_scale=scale, results_rad_s=results,
          minimum_model_gain_rad_s=gate, nominated_cap_rad_s=min(passed, default=None),
          physical_substeps=0, learning_updates=0, heldout_used=False,
          scope='A causal proposed mapping replays on unperturbed traces. This frozen-input linear surrogate does not measure nonlinear closed-loop tracking.'))
    print('WALK_REBOUND_FEEDBACK', results, 'nominated', min(passed, default=None), flush=True)


if __name__ == '__main__':
    main()
