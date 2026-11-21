"""Screen a causal feedback rule under the measured actuator-only response."""
from pathlib import Path

import numpy as np

from review_common_walk_yaw import DEVELOPMENT, OUT as WALK
from review_common_walk_yaw_forecast import fit, predict
from review_walk_yaw_pulse_capacity import apply_kernel, upper_bound_and_oracle
from review_walk_yaw_rebound_feedback import fixture
from record_io import write
from s1_reference_inputs import ROOT, read, sha


OUT = ROOT/'records/reports/common-walk-yaw-history-probe'


def main():
    path = OUT/'analysis.json'
    proof = read(path)
    plan = ROOT/'configs/experiments/common_walk_yaw_history_probe.json'
    spec = read(plan)
    assert proof['protocol_sha256'] == sha(plan) and not proof['heldout_used']
    assert proof['response']['passes_history_isolation_gate'] and proof['matched_applied_targets']
    record = ROOT/proof['record']
    forward = spec['environment']['groups']['forward']
    signs = np.asarray(spec['phase']['pulse_signs'])[forward]
    yaw = {}
    for role in ('reference', 'actuator-pulse'):
        file = record/f'trace-{role}.npz'
        assert sha(file) == proof['trace_sha256'][role]
        with np.load(file) as archive:
            assert archive['active'][:6, forward].all()
            yaw[role] = archive['yaw_rate'][:6, forward].astype(np.float64)
    kernel = (signs*(yaw['actuator-pulse']-yaw['reference'])).mean(axis=1)/spec['phase']['peak_rad_s']
    np.testing.assert_allclose(kernel[:2]*spec['phase']['peak_rad_s'],
                               [proof['response'][f'actuator_{ms}ms_rad_s'] for ms in (20, 40)], atol=1e-12)
    source = WALK/'development-yaw-review.json'
    history = read(source)
    assert not history['heldout_used']
    x, target = fixture(DEVELOPMENT[0], history['source_sha256'])
    models = [fit(x, np.sign(target[..., i]), (0, 1, 2, 3)) for i in range(6)]

    def gradient(observed):
        signs = np.clip(np.stack([predict(observed, model) for model in models], axis=-1), -1., 1.)
        return signs@kernel

    scale = float(np.quantile(np.abs(gradient(x)), .9))
    assert scale > 0
    caps = (.06, .12)
    results = {}
    for seed in DEVELOPMENT[1:]:
        x, target = fixture(seed, history['source_sha256'])
        direction = -np.clip(gradient(x)/scale, -1., 1.)
        yaw = target[..., 0]
        rows = {}
        for cap in caps:
            bound, _ = upper_bound_and_oracle(yaw, kernel, cap)
            gain = np.abs(yaw)-np.abs(yaw+apply_kernel(cap*direction, kernel))
            rows[str(cap)] = dict(mean_gain_rad_s=float(gain.mean()), linear_upper_bound_rad_s=float(bound.mean()))
        results[str(seed)] = rows
    required = spec['history_probe']['admission']['original_20s_tracking_gain_gate_rad_s']
    passing = [cap for cap in caps if all(row[str(cap)]['mean_gain_rad_s'] >= required for row in results.values())]
    write(OUT/'saved-actuator-feedback-screen.json', dict(status='passed-saved-linear-response-check',
          proof_sha256=sha(path), source_sha256=sha(source), script_sha256=sha(Path(__file__)),
          kernel=kernel.tolist(), fit_seed=DEVELOPMENT[0], check_seeds=list(DEVELOPMENT[1:]),
          forecast_models=models, gradient_scale=scale, results=results, minimum_gain_rad_s=required,
          nominated_cap_rad_s=min(passing, default=None), heldout_used=False, learning_updates=0,
          physical_substeps=0, scope='Linear open-loop replay of archived observations. Nomination still requires CPU target checks and a new closed-loop run.'))
    print('WALK_ACTUATOR_FEEDBACK_SCREEN', results, 'nominated', min(passing, default=None), flush=True)


if __name__ == '__main__':
    main()
