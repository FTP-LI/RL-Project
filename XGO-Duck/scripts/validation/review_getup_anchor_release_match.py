"""Check whether saved recovery arms reached comparable physical release poses."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from getup_coverage_metrics import POSES, evaluate
from record_io import write
from recovery_end_hold import evaluate_end_hold


OUT = ROOT / 'records/reports/getup-teacher-anchor-pair'
ROLES = ('original', 'control', 'candidate')
LIMITS = dict(root_position_m=.003, root_orientation_deg=3., joint_angle_deg=3.,
              root_linear_speed_m_s=.05, angular_speed_rad_s=.25)


def load(role):
    proof_path = OUT / f'{role}-analysis.json'
    proof = read(proof_path)
    folder = ROOT / proof['record']
    trace_path = folder / 'coverage-trace.npz'
    plan = read(folder / 'protocol.json')
    assert sha(trace_path) == proof['trace_sha256'] and proof['status'] == 'passed'
    assert plan['seed'] == 1030 and proof['learning_updates'] == 0
    with np.load(trace_path) as saved:
        trace = {name: saved[name] for name in saved.files}
    checked = evaluate(trace, plan)
    end = evaluate_end_hold(trace, plan, checked['episodes'])
    for key in ('requested', 'actual', 'overall'):
        assert checked[key] == proof[key] and end[key] == proof['end_hold'][key]
    k = np.flatnonzero(np.isclose(trace['time'], 2., atol=1e-8, rtol=0)).item()
    pose = np.array([row['release_pose'] for row in checked['episodes']])
    requested = np.array([row['requested_pose'] for row in checked['episodes']])
    timely = np.array([row['hold_start_seconds'] is not None for row in checked['episodes']])
    end_flags = np.array(end['stable_by_environment'])
    return dict(qpos=trace['qpos'][k], qvel=trace['qvel'][k], pose=pose, requested=requested,
                pass_both=timely & end_flags, trace_sha256=sha(trace_path), proof_sha256=sha(proof_path))


def compare(first, second):
    np.testing.assert_array_equal(first['requested'], second['requested'])
    q0, q1, v0, v1 = first['qpos'], second['qpos'], first['qvel'], second['qvel']
    quat0, quat1 = q0[:, 3:7].astype(float), q1[:, 3:7].astype(float)
    denom = np.linalg.norm(quat0, axis=1) * np.linalg.norm(quat1, axis=1)
    angle = np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(quat0*quat1, axis=1))/denom, 0., 1.)))
    errors = dict(root_position_m=np.linalg.norm(q0[:, :3]-q1[:, :3], axis=1), root_orientation_deg=angle,
        joint_angle_deg=np.rad2deg(np.max(abs(q0[:, 7:]-q1[:, 7:]), axis=1)),
        root_linear_speed_m_s=np.linalg.norm(v0[:, :3]-v1[:, :3], axis=1),
        angular_speed_rad_s=np.max(abs(v0[:, 3:]-v1[:, 3:]), axis=1))
    same_class = first['pose'] == second['pose']
    exact = (q0 == q1).all(axis=1) & (v0 == v1).all(axis=1)
    close = same_class & np.logical_and.reduce([v <= LIMITS[key] for key, v in errors.items()])
    groups = {}
    for pose in POSES:
        included = (first['pose'] == pose) & close
        a, b = first['pass_both'][included], second['pass_both'][included]
        groups[pose] = dict(worlds=int(included.sum()), first_successes=int(a.sum()), second_successes=int(b.sum()),
                            second_only=int((b & ~a).sum()), first_only=int((a & ~b).sum()))
    return dict(identical_physical_release=int(exact.sum()), same_actual_release_class=int(same_class.sum()),
        close_physical_release=int(close.sum()), close_by_actual_pose=groups,
        distance_quantiles={key: np.quantile(v, [.5, .9, 1.]).tolist() for key, v in errors.items()})


def main():
    data = {role: load(role) for role in ROLES}
    result = dict(status='passed-saved-data-only', seed=1030, match_limits=LIMITS,
        comparisons={role: compare(data['candidate'], data[role]) for role in ('original', 'control')},
        provenance={role: {key: data[role][key] for key in ('trace_sha256', 'proof_sha256')} for role in ROLES},
        script_sha256=sha(Path(__file__)), new_physical_steps=0, learning_updates=0, heldout_used=False,
        scope='Descriptive same-index physical release matching. Physical proximity cannot recover hidden state or establish a causal policy effect.')
    write(OUT / 'release-match-review.json', result)
    print('GETUP_ANCHOR_RELEASE_MATCH', result['comparisons'], flush=True)


if __name__ == '__main__':
    main()
