"""Estimate a tracking-gain ceiling under the measured short linear response."""
from pathlib import Path

import numpy as np

from review_common_walk_yaw import DEVELOPMENT, OUT as WALK
from review_common_walk_yaw_forecast import load, predict
from record_io import write
from s1_reference_inputs import ROOT, read, sha


OUT = ROOT/'records/reports/common-walk-yaw-pulse'


def apply_kernel(command, kernel):
    response = np.zeros_like(command)
    for lag, weight in enumerate(kernel):
        response[lag:] += weight*command[:len(command)-lag]
    return response


def upper_bound_and_oracle(target, kernel, limit):
    # Convexity of the absolute loss gives an upper bound on gain using
    # the transpose of the finite convolution. This oracle sees future yaw.
    sign = np.sign(target)
    gradient = np.zeros_like(target)
    for lag, weight in enumerate(kernel):
        gradient[:len(target)-lag] += weight*sign[lag:]
    bound = limit*np.abs(gradient).mean(axis=0)
    change = apply_kernel(-limit*np.sign(gradient), kernel)
    achieved = (np.abs(target)-np.abs(target+change)).mean(axis=0)
    assert np.all(achieved <= bound+1e-12)
    return bound, achieved


def main():
    pulse_path = OUT/'analysis.json'
    plan = ROOT/'configs/experiments/common_walk_yaw_pulse.json'
    pulse = read(pulse_path)
    assert pulse['status'] == 'passed-integrity' and pulse['protocol_sha256'] == sha(plan)
    assert pulse['response']['immediate_effect_verified'] and not pulse['response']['can_test_60ms_feedback']
    assert not pulse['heldout_used']
    evidence = WALK/'development-yaw-review.json'
    forecast_path = WALK/'development-yaw-forecast.json'
    raw, forecast = read(evidence), read(forecast_path)
    assert forecast['source_sha256'] == sha(evidence) and not forecast['heldout_used']
    gyro = forecast['results']['0.02']['models']['gyro']
    train, _ = load(DEVELOPMENT[0], raw['source_sha256'])
    scale = float(np.quantile(np.abs(predict(train, gyro)), .9))
    assert scale > 0
    observed = pulse['response']['windows']
    settings = read(plan)
    cap = settings['phase']['peak_rad_s']
    threshold = settings['gate']['tracking_gate_unchanged_rad_s']
    results = {}
    for seed in DEVELOPMENT[1:]:
        x, (target, _) = load(seed, raw['source_sha256'])
        feedback = -cap*np.clip(predict(x, gyro)/scale, -1., 1.)
        rows = {}
        for reference in ('reference-a', 'reference-b'):
            kernel = np.asarray([observed[str(ms)]['paired'][reference]['mean_rad_s']
                                 for ms in range(20, 121, 20)])/cap
            bound, oracle = upper_bound_and_oracle(target, kernel, cap)
            change = apply_kernel(feedback, kernel)
            gain = (np.abs(target)-np.abs(target+change)).mean(axis=0)
            ceiling = float(bound.mean())
            rows[reference] = dict(kernel=kernel.tolist(), mean_gain_upper_bound_rad_s=ceiling,
                                   acausal_oracle_gain_rad_s=float(oracle.mean()),
                                   gyro_replay_gain_rad_s=float(gain.mean()),
                                   minimum_cap_for_tracking_gate_rad_s=cap*threshold/ceiling)
        results[str(seed)] = rows
    current = all(row['mean_gain_upper_bound_rad_s'] >= threshold for rows in results.values() for row in rows.values())
    write(OUT/'saved-feedback-capacity.json', dict(status='passed-saved-linear-response-check',
          pulse_sha256=sha(pulse_path), forecast_sha256=sha(forecast_path), script_sha256=sha(Path(__file__)),
          diagnostic_limit_rad_s=cap, tracking_gate_rad_s=threshold, gyro_forecast_scale_rad_s=scale,
          current_cap_reaches_gate_in_model=current, cohorts=results, physical_substeps=0,
          learning_updates=0, heldout_used=False,
          scope='Six observed mean impulse samples form a truncated linear surrogate. The ceiling applies to this surrogate, not the nonlinear closed-loop plant.'))
    print('WALK_PULSE_CAPACITY', {seed: {ref: (row['mean_gain_upper_bound_rad_s'], row['gyro_replay_gain_rad_s'],
            row['minimum_cap_for_tracking_gate_rad_s']) for ref, row in rows.items()} for seed, rows in results.items()},
          'current-cap', current, flush=True)


if __name__ == '__main__':
    main()
