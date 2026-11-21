"""Check failure/reset accounting with outcomes fixed independently of simulation."""
import json
from pathlib import Path
import numpy as np
from evaluation_metrics import FirstEpisodeMetrics


def main():
    metrics = FirstEpisodeMetrics(np.zeros((5, 2)), 3, 0.02)
    # 0 falls early, 1 reaches horizon, 2 times out early, 3 falls on the last
    # step, 4 leaves terrain at horizon (even though mjlab calls this timeout).
    for step in range(1, 4):
        fell = np.array([step == 1, False, False, step == 3, False])
        truncated = np.array([False, step == 3, step == 2, False, step == 3])
        bounds = np.array([False, False, False, False, step == 3])
        velocity = np.full((5, 2), 0.1)
        velocity[~metrics.active] = 999  # Reset-episode data must not leak in.
        metrics.update(step, np.full((5, 2), step), velocity, np.zeros(5),
                       np.zeros(3), np.full((5, 14), 0.2), np.ones((5, 2)),
                       np.ones((5, 2), bool), fell, truncated, fell,
                       np.zeros(5, bool), bounds)
    report = metrics.result()
    assert report['full_episode_survival'] == 0.2
    assert report['fall_count_excluding_timeouts'] == 2
    assert [r['steps'] for r in report['episodes']] == [1, 3, 2, 3, 3]
    assert report['episodes'][2]['outcome'] == 'early_timeout'
    assert report['episodes'][4]['outcome'] == 'out_of_terrain_bounds'
    assert abs(report['linear_xy_mae_m_s'] - np.sqrt(0.02)) < 1e-12
    assert abs(report['action_delta_rms'] - 0.2) < 1e-12
    assert abs(report['contact_foot_slip_m_s'] - 1.0) < 1e-12
    assert abs(report['episodes'][0]['final_displacement_m'] - np.sqrt(2)) < 1e-12
    standing = FirstEpisodeMetrics(np.zeros((1,2)), 1, 0.02)
    standing.update(1, np.zeros((1,2)), np.zeros((1,2)), np.zeros(1),
                    np.array([0.15,0,0]), np.zeros((1,14)), np.zeros((1,2)),
                    np.ones((1,2),bool), np.zeros(1,bool), np.ones(1,bool),
                    np.zeros(1,bool), np.zeros(1,bool), np.zeros(1,bool))
    assert standing.result()['linear_xy_mae_m_s'] == 0.15
    assert standing.result()['linear_xy_mae_m_s'] > 0.08
    out = Path(__file__).resolve().parents[2]/'records/evaluations'
    out.mkdir(exist_ok=True, parents=True)
    (out/'metric-accounting-check.json').write_text(json.dumps(dict(status='passed',
        scope='Synthetic outcome accounting, not policy evaluation', result=report), indent=2)+'\n')
    print('Metric accounting passed: early failure, post-reset exclusion, horizon timeout, final-step fall, terrain timeout.')


if __name__ == '__main__':
    main()
