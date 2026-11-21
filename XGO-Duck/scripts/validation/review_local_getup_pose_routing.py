"""Score one post-hoc pose route on the already verified same-release traces."""
from pathlib import Path

import numpy as np

from evaluate_local_getup_ppo import PLAN
from getup_coverage_metrics import POSES, group_rows
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import classify_release


def main():
    spec = read(PLAN)
    out = (ROOT/spec['training_evidence']).parent
    analysis = out/'evaluation-analysis.json'
    saved = read(analysis)
    assert saved['protocol_sha256'] == sha(PLAN) and saved['first_inputs_and_delayed_motor_exact']
    record = ROOT/saved['record']
    parent, _, higher = spec['branches']
    left, right = saved['arms'][parent], saved['arms'][higher]
    with np.load(record/f'first-control-{parent}.npz') as data:
        inputs = data['actor_input']
    with np.load(record/f'first-control-{higher}.npz') as data:
        np.testing.assert_array_equal(inputs, data['actor_input'])
    assert spec['model_interface']['observation_names'].split(',')[:2] == ['base_ang_vel', 'projected_gravity']
    p = spec['environment']
    gravity = inputs[:, 3:6]
    assert gravity.shape == (p['num_envs'], 3)
    scene = record/'env.yaml'
    assert 'n_min: -0.03' in scene.read_text() and 'n_max: 0.03' in scene.read_text()
    magnitude = np.linalg.norm(gravity, axis=1)
    # The saved policy observation includes additive sensor noise. Classify
    # the direction, keeping the original orientation threshold unchanged.
    np.testing.assert_allclose(magnitude, 1., atol=np.sqrt(3)*.03+2e-4, rtol=0)
    observed = classify_release(gravity/magnitude[:, None], p['criteria']['release_alignment_min'])
    actual = [episode['release_pose'] for episode in left['timing']['episodes']]
    flags = np.asarray(observed) == 'right-side'
    timely, final, combined = [], [], 0
    for i, select in enumerate(flags):
        model = right if select else left
        row = model['timing']['episodes'][i]
        assert row['env_index'] == i and row['release_pose'] == actual[i]
        hold = model['end_hold']['stable_by_environment'][i]
        timely.append(row)
        final.append(dict(row, hold_start_seconds=model['end_hold']['start_seconds'] if hold else None,
                          success=bool(hold and row['release_pose_valid'])))
        combined += int(row['hold_start_seconds'] is not None and hold and actual[i] in POSES)
    stats = [group_rows(rows, p['success_rate_min'], p['actual_release_minimum']) for rows in (timely, final)]
    write(out/'posthoc-pose-routing.json', dict(status='scored-saved-pose-route-only', evaluation_sha256=sha(analysis),
          script_sha256=sha(Path(__file__)), right_side_role=higher, default_role=parent,
          cue='Normalize the first common projected-gravity actor input at release, then latch one branch.',
          scene_sha256=sha(scene), observed_gravity_norm_range=[float(magnitude.min()), float(magnitude.max())],
          branch_worlds=np.flatnonzero(flags).tolist(), mismatched_right_side=np.flatnonzero(
              flags != (np.asarray(actual) == 'right-side')).tolist(),
          combined_successes=combined, timely=stats[0], final_hold=stats[1],
          new_physical_substeps=0, learning_updates=0, heldout_used=False, independent_motion_acceptance=False))
    print('POSTHOC_GETUP_ROUTE', combined, int(flags.sum()), [row['functional_coverage_gate'] for row in stats], flush=True)


if __name__ == '__main__':
    main()
