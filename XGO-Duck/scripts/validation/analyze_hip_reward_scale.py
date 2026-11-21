"""Measure candidate hip-position penalty scales on verified saved episodes.

The preceding diagnostic evaluated the unchanged official position penalty.
Here, magnitudes 1, 2 and 4 are compared to the existing pose reward ceiling
to make a subsequent weight choice reviewable. No candidate is selected, no
training reward is changed, and this arithmetic is not a policy evaluation.
The old matched pilot and official reference have different training histories.
"""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'records/reports/hip-posture-candidates'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    analysis_path = OUT / 'analysis.json'
    episode_path = OUT / 'episodes.csv'
    analysis = json.loads(analysis_path.read_text())
    assert analysis['status'] == 'passed'
    assert analysis['unweighted_candidate_costs'] is True
    assert sha(episode_path) == analysis['episode_csv_sha256']
    bindings = {analysis_path: sha(analysis_path), episode_path: sha(episode_path)}
    for relative, expected in analysis['inputs_sha256'].items():
        file = ROOT / relative
        assert sha(file) == expected, relative
        bindings[file] = expected
    with episode_path.open() as stream:
        episodes = list(csv.DictReader(stream))
    assert len(episodes) == analysis['episode_rows'] == 960
    ceiling = analysis['pose_weight']
    margin = analysis['limit_proximity_margin_rad']
    assert ceiling > 0 and margin == .15
    config_path = ROOT / 'records/experiments/head_command_repair_control_20261101T213026_s42/params/env.yaml'
    config = yaml.load(config_path.read_text(), Loader=yaml.BaseLoader)
    assert config['scale_rewards_by_dt'] == 'true'
    assert float(config['rewards']['pose']['weight']) == ceiling
    step_dt = int(config['decimation']) * float(config['sim']['mujoco']['timestep'])
    assert step_dt == .02
    bindings[config_path] = sha(config_path)
    results = []
    for model in analysis['provenance']:
        for command in ('stop', 'forward', 'backward', 'turn_left', 'turn_right'):
            rows = [r for r in episodes if r['model'] == model
                    and r['condition'] == command
                    and r['window'] == 'after_first_second']
            assert len(rows) == 32 and {int(r['env']) for r in rows} == set(range(32))
            costs = np.array([float(r['limit_proximity']) for r in rows])
            assert np.isfinite(costs).all() and (costs >= 0).all()
            for magnitude in (1.0, 2.0, 4.0):
                results.append(dict(model=model, condition=command, episodes=32,
                    candidate_weight=-magnitude,
                    mean_unweighted_cost_rad=float(costs.mean()),
                    mean_weighted_reward_rate=-magnitude * float(costs.mean()),
                    mean_step_reward=-magnitude * float(costs.mean()) * step_dt,
                    cost_fraction_of_pose_ceiling=magnitude * float(costs.mean()) / ceiling))
    file = OUT / 'weight-scale.csv'
    with file.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for path, expected in bindings.items():
        assert sha(path) == expected, path
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)),
        inputs_sha256={str(p.relative_to(ROOT)): v for p, v in bindings.items()},
        candidate_weights=[-1.0, -2.0, -4.0], existing_pose_reward_ceiling=ceiling,
        reward_scale_by_dt=True, control_step_seconds=step_dt,
        cost_at_two_hard_stops_rad=2 * margin,
        weighted_magnitude_at_two_hard_stops=[2 * margin * w for w in (1.0, 2.0, 4.0)],
        results=results, table_sha256=sha(file), selected_weight=None,
        training_started=False, learning_improvement_verified=False,
        scope='Arithmetic reward-rate scale on existing first episodes after one second; '
              'not discounted return, live training reward or a simulated intervention')
    (OUT / 'weight-scale.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('inputs_sha256', 'results')}, indent=2))
    print(json.dumps([r for r in results if r['condition'] == 'forward'], indent=2))


if __name__ == '__main__':
    main()
