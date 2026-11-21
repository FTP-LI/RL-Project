"""Apply the predeclared adaptation progress gate to saved development traces.

No simulator or training is run. The gate decides whether direct continuation
is worth another reviewed protocol, not whether the seven-motion task passed.
"""
from pathlib import Path
import json
import numpy as np
import yaml
from s1_reference_inputs import ROOT, read, sha
from analyze_official_bootstrap import metrics

OUT = ROOT / 'records/reports/official-adaptation'


def main():
    progress = read(OUT / 'evaluation-progress.json')
    assert progress['status'] == 'collected-and-verified'
    current = Path(progress['evaluation'])
    previous = Path(read(ROOT / 'records/reports/official-actor-bootstrap/evaluation-progress.json')['evaluation'])
    cfg = lambda p: yaml.load((p / 'env.yaml').read_text(), Loader=yaml.BaseLoader)
    assert cfg(current) == cfg(previous), 'Evaluation configuration changed'
    a, b = read(current / 'evaluation.json'), read(previous / 'evaluation.json')
    assert a['source_sha256'] == b['source_sha256'] and a['packages'] == b['packages']
    review_path = ROOT / 'records/reports/official-actor-bootstrap/continuation-review.json'
    gate = read(review_path)['next_progress_gate']
    rows = {'bootstrap': metrics(previous), 'adaptation': metrics(current)}
    delta = {k: {f: rows['adaptation'][k][f] - rows['bootstrap'][k][f]
                 for f in ('linear_error', 'yaw_error', 'signed_speed', 'inward_fraction_after1s')}
             for k in rows['adaptation']}
    checks = {
        'stand_pass': rows['adaptation']['stop']['aggregate_tracking_pass'],
        'posture_preserved': all(x['inward_fraction_after1s'] <= gate['posture_inward_fraction_max']
                                for x in rows['adaptation'].values()),
        'survival_preserved': all(x['survivors'] / x['episodes'] >= gate['survival_min']
                                 for x in rows['adaptation'].values()),
        'moving_progress': any(delta[k]['linear_error'] <= -gate['required_error_reduction_in_at_least_one_moving_command']
                               for k in ('forward', 'backward')),
        'no_other_command_regression': all(delta[k]['linear_error'] <= gate['max_error_increase_other_command']
                                           for k in ('stop', 'forward', 'backward')),
        'no_yaw_regression': all(x['yaw_error'] <= gate['max_yaw_error_increase'] for x in delta.values()),
    }
    assert all(np.isfinite(v) for row in delta.values() for v in row.values())
    result = dict(status='passed', rows=rows, delta_candidate_minus_bootstrap=delta,
                  gate=gate, checks=checks, progress_gate_passed=all(checks.values()),
                  decision='review-next-protocol' if all(checks.values()) else 'end-direct-continuation-route',
                  same_evaluation_configuration=True, evaluation_episode_count=96,
                  seven_command_acceptance=False, heldout_used=False, multi_seed_confirmed=False,
                  automatic_learning_authorized=False,
                  review_sha256=sha(review_path), script_sha256=sha(Path(__file__)),
                  evaluation_sha256=sha(current / 'evaluation.json'),
                  parent_evaluation_sha256=sha(previous / 'evaluation.json'),
                  scope='Development progress only; no causal reward ablation or completed gait claim')
    (OUT / 'screen-analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
