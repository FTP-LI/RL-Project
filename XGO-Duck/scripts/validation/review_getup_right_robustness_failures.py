"""Summarize right-side failure patterns from the independent development scores."""

from pathlib import Path

import numpy as np

import run_getup_right_robustness_development as study
from standup_metrics import projected_gravity


def main():
    out, read, sha, write = study.OUT, study.read, study.sha, study.write
    analysis_path = out / 'analysis.json'
    decision = read(out / 'decision.json')
    release = read(out / 'release-verification.json')
    assert decision['analysis_sha256'] == release['analysis_sha256'] == sha(analysis_path)
    analysis, spec = read(analysis_path), read(study.PLAN)
    assert analysis['protocol_sha256'] == sha(study.PLAN) and not analysis['heldout_used']
    record = study.ROOT / analysis['record']
    result = out / 'right-failure-review.json'
    assert not result.exists()
    p, routing = spec['environment'], spec['dual_route']
    criteria, start = p['criteria'], spec['prefix_controls']
    with np.load(record / 'release-coverage.npz', allow_pickle=False) as archive:
        poses = archive['poses']
    with np.load(record / 'initial.npz', allow_pickle=False) as archive:
        requested = archive['requested_pose']
    right = np.flatnonzero(poses == 'right-side')
    chosen = release['selected_world']
    assert right.size == release['counts']['right-side'] and chosen == right[0]
    summaries, pass_matrix = {}, []
    for role in spec['branches']:
        arm = analysis['arms'][role]
        timed = np.array([row['hold_start_seconds'] is not None for row in arm['timely']['episodes']])
        final = np.array(arm['final']['stable_by_environment'], dtype=bool)
        score = timed & final
        pass_matrix.append(score[right])
        failures = right[~score[right]]
        source = record / f'trace-{role}.npz'
        assert sha(source) == arm['trace_sha256']
        with np.load(source, allow_pickle=False) as archive:
            time = archive['time']
            quat = archive['quaternion'][:, failures]
            feet = archive['foot_contact'][:, failures].all(axis=-1)
            head = archive['native_head_leg_contact'][:, failures]
        assert len(time) == p['maximum_control_steps'] - start
        tilt = np.rad2deg(np.arccos(np.clip(-projected_gravity(quat)[..., 2], -1, 1)))
        over = tilt > criteria['tilt_max_deg']
        final_window = time > criteria['max_seconds'] - criteria['hold_seconds']
        assert final_window.sum() == round(criteria['hold_seconds'] / criteria['step_dt'])
        early = (time <= criteria['freeze_seconds'] + 1.)
        never_upright = over.all(axis=0)
        persistent = (feet[final_window] & head[final_window] & over[final_window]).all(axis=0)
        rows = []
        for j, world in enumerate(failures.tolist()):
            rows.append(dict(world=world, timely_hold=bool(timed[world]), final_hold=bool(final[world]),
                tilt_at_3s_deg=float(tilt[early][-1, j]), minimum_active_tilt_deg=float(tilt[:, j].min()),
                final_tilt_deg=float(tilt[-1, j]), never_reached_upright_tilt=bool(never_upright[j]),
                final_head_leg_fraction=float(head[final_window, j].mean()),
                final_dual_feet_fraction=float(feet[final_window, j].mean()),
                persistent_dual_feet_head_leg_and_tilt=bool(persistent[j])))
        assert len(right) - len(failures) == arm['combined']['actual']['right-side']['timely_holds']
        cross_groups = {}
        for start_pose in study.base.shared.POSES:
            for release_pose in study.base.shared.POSES:
                group = (requested == start_pose) & (poses == release_pose)
                if group.any():
                    cross_groups[f'{start_pose}/{release_pose}'] = dict(worlds=int(group.sum()),
                        passed=int((score & group).sum()))
        summaries[role] = dict(pass_count=int(score[right].sum()), failed=len(failures),
            chosen_passed=bool(score[chosen]), never_reached_upright_tilt=int(never_upright.sum()),
            persistent_dual_feet_head_leg_and_tilt=int(persistent.sum()),
            median_final_tilt_deg=float(np.median(tilt[-1])) if len(failures) else None,
            trace_sha256=arm['trace_sha256'], requested_actual=cross_groups, failures=rows)
    overlap = np.stack(pass_matrix).sum(axis=0)
    raw = read(record / 'evaluation.json')['conditions']
    raw_world = raw[0]['video']['environment_index']
    assert all(row['video']['environment_index'] == raw_world for row in raw)
    write(result, dict(status='completed-development-right-failure-review', seed=spec['environment']['seed'],
        analysis_sha256=sha(analysis_path), script_sha256=sha(Path(__file__)), actual_right=len(right),
        all_pass=int((overlap == len(summaries)).sum()), all_fail=int((overlap == 0).sum()),
        all_fail_worlds=right[overlap == 0].tolist(),
        split=int(((overlap > 0) & (overlap < len(summaries))).sum()), arms=summaries,
        raw_video_world=raw_world, raw_video_release_pose=str(poses[raw_world]), preselected_right_world=chosen,
        alignment=routing['alignment'], learning_updates=0, physical_substeps=0, heldout_used=False))
    print('RIGHT_ROBUSTNESS_FAILURE_REVIEW', len(right), {role: value['failed'] for role, value in summaries.items()},
          'common failures', int((overlap == 0).sum()), flush=True)


if __name__ == '__main__':
    main()
