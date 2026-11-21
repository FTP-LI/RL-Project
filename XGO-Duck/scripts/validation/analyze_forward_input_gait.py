"""Separate the observed cadence and contact-phase yaw changes of the admitted input."""

import numpy as np

from audit_forward_stride_response import REPORT, ROOT, read, sha, write


def read_branch(folder, role, receipt):
    branch = folder/'clips'/role
    source = branch/'trace.npz'
    score = ROOT/receipt['raw']
    assert sha(score) == receipt['raw_sha256']
    samples = read(score)
    windows = np.asarray(samples['stride_windows_from_ramp'])
    assert (windows >= 0).all()
    branch_meta = read(folder/'guarded-evaluation.json')['branches'][role]
    assert sha(source) == branch_meta['trace_sha256']['trace.npz']
    with np.load(source, allow_pickle=False) as state:
        yaw = np.abs(state['angular_velocity'][16:, :, 2]).astype(np.float64)
        feet = state['foot_contact']
        position = state['qpos'][15:, :, :2].astype(np.float64)
    labels = feet[..., 0].astype(np.int8)+2*feet[..., 1].astype(np.int8)
    ages = np.ones_like(labels, dtype=np.int32)
    for k in range(1, len(labels)):
        ages[k] = np.where(labels[k] == labels[k-1], ages[k-1]+1, 1)
    labels, ages = labels[16:], ages[16:]
    world = np.arange(windows.shape[0])[:, None]
    displacement = np.linalg.norm(position[windows[..., 1], world]-position[windows[..., 0], world], axis=-1)/2
    result = dict(
        median_complete_stride_period_seconds_by_anchor=(np.median((windows[..., 1]-windows[..., 0])/2, axis=0)*.02).tolist(),
        median_planar_net_displacement_per_stride_m_by_anchor=np.median(displacement, axis=0).tolist(),
        median_scene_abs_yaw_after_ramp_rad_s=float(np.median(yaw.mean(axis=0))),
        pooled_yaw_by_support_rad_s={},
    )
    for side, value in (('left', 1), ('right', 2)):
        for name, part in (('early', ages <= 3), ('middle', (ages >= 4) & (ages <= 8)), ('late', ages >= 9)):
            result['pooled_yaw_by_support_rad_s'][f'{side}_{name}'] = float(np.where((labels == value) & part, yaw, 0).mean())
    for name, value in (('flight', 0), ('double', 3)):
        result['pooled_yaw_by_support_rad_s'][name] = float(np.where(labels == value, yaw, 0).mean())
    assert abs(sum(result['pooled_yaw_by_support_rad_s'].values()) - yaw.mean()) < 1e-9
    return dict(result, trace_sha256=sha(source), score_sha256=sha(score))


def main():
    result = REPORT/'stride-comparison.json'
    project = read(ROOT/'project.json')['goal_execution']['forward_stride_yaw_response']
    assert sha(result) == project['result_sha256']
    data = read(result)
    roles = ('reference-a', 'reference-b', 'forward-input-0130', 'forward-input-0115')
    cases = {}
    for name, case in data['cases'].items():
        folder = ROOT/'records/evaluations'/case['run_id']
        assert sha(folder/'guarded-evaluation.json') == case['collection_sha256']
        arms = {role:read_branch(folder, role, case['arms'][role]) for role in roles}
        baseline = [arms[role]['pooled_yaw_by_support_rad_s'] for role in roles[:2]]
        candidates = {}
        for role in roles[2:]:
            motion = arms[role]['pooled_yaw_by_support_rad_s']
            candidates[role] = {phase: (baseline[0][phase]+baseline[1][phase])/2-motion[phase] for phase in motion}
        cases[name] = dict(arms=arms, pooled_abs_yaw_reduction_by_support_rad_s=candidates)
    write(REPORT/'input-gait-phase-analysis.json', dict(source_sha256=sha(result), cases=cases,
           phase_basis='Time and world pooled sums divided by all controls times all worlds after the actual 15-control input ramp.',
           physical_steps=0, learning_updates=0, hardware_io=False))
    for name, row in cases.items():
        arms, effect = row['arms'], row['pooled_abs_yaw_reduction_by_support_rad_s']
        print('STRIDE_INPUT_MECHANICS', name, {role:arm['median_complete_stride_period_seconds_by_anchor'] for role,arm in arms.items()},
              {role:arm['median_planar_net_displacement_per_stride_m_by_anchor'] for role,arm in arms.items()},
              {role:sum(phases.values()) for role,phases in effect.items()}, flush=True)


if __name__ == '__main__':
    main()
