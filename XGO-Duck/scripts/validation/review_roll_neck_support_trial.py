"""Summarize trigger timing, actual motor readback and the two null repeats."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT / 'records/reports/roll-neck-support/log-recovery'


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    width = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [float(max(0., center-width)), float(min(1., center+width))]


def duration(trace, step, world):
    if not trace['head_contact'][step, world]:
        return None
    future = trace['head_contact'][step:, world] & trace['active'][step:, world]
    gap = np.flatnonzero(~future)
    return int(gap[0]) if len(gap) else len(future)


def main():
    analysis_path = OUT / 'readback/analysis.json'
    analysis, run = read(analysis_path), read(OUT / 'coordinator.json')
    verification = read(OUT / 'readback/verification.json')
    assert run['status'] == 'stopped-review-required' and verification['original_receipts_preserved']
    assert sha(analysis_path) == verification['analysis_sha256']
    assert not analysis['heldout_used'] and analysis['learning_updates'] == 0
    record = ROOT / analysis['record']
    prefix_path = record / 'trace-shared-prefix.npz'
    assert sha(prefix_path) == analysis['prefix_sha256']
    keys = ('head_contact', 'active', 'qpos', 'qvel', 'neck_cue', 'neck_phase_deg', 'motor')
    with np.load(prefix_path) as store:
        prefix = {key: store[key] for key in keys}
    summary, data = {}, {}
    for role, proof in analysis['arms'].items():
        path = record / f'trace-{role}.npz'
        assert sha(path) == proof['trace_sha256']
        with np.load(path) as store:
            data[role] = {key: np.concatenate((prefix[key], store[key]), axis=0) for key in keys}
        episodes = proof['episodes']
        assert len(episodes) == 128
        success = proof['successes']
        summary[role] = dict(successes=success, attempts=len(episodes), wilson95=wilson(success, len(episodes)),
            headtop_sequence=sum(row['head_contact_seconds'] is not None for row in episodes),
            supported_completion=sum(row['completion_seconds'] is not None for row in episodes),
            timely_landing_hold=sum(row['hold_end_seconds'] is not None for row in episodes),
            joint_limit_episodes=sum(row['joint_violation_samples'] > 0 for row in episodes),
            penetration_limit_episodes=sum(row['head_penetration_max_m'] > .005 for row in episodes),
            camera_success=episodes[0]['success'])
    neck = 'neck-support'
    a, b, c = (data[role] for role in ('reference-a', 'reference-b', neck))
    worlds = analysis['arms'][neck]['targets']['modified_worlds']
    rows = []
    for world in worlds:
        k = int(np.flatnonzero(c['neck_cue'][:, world])[0])
        pose = c['qpos'][k-1, world]
        span = {role: duration(t, k-1, world) for role, t in data.items()}
        candidate_score = analysis['arms'][neck]['episodes'][world]
        values = c['motor'][k, :, world]
        rows.append(dict(world=world, first_request_seconds=(k-1)*.02, control=k,
            phase_degrees=float(c['neck_phase_deg'][k, world]), candidate_success=candidate_score['success'],
            max_qpos_component_difference_at_request=dict(
                reference_repeat=float(np.max(abs(a['qpos'][k-1, world]-b['qpos'][k-1, world]))),
                against_a=float(np.max(abs(pose-a['qpos'][k-1, world]))),
                against_b=float(np.max(abs(pose-b['qpos'][k-1, world])))),
            remaining_head_contact_controls=span,
            next_neck_speed_change_rad_s=float(c['qvel'][k, world, run['neck_dof']]-c['qvel'][k-1, world, run['neck_dof']]),
            delayed_neck_target_by_substep_rad=values[:, 0].tolist(),
            smoothed_neck_target_by_substep_rad=values[:, 1].tolist(),
            voltage_by_substep_v=values[:, 2].tolist(), motor_torque_by_substep_nm=values[:, 3].tolist()))
    assert len(rows) == len(worlds)
    align = {}
    for role in ('reference-a', 'reference-b'):
        differences = [row['max_qpos_component_difference_at_request'][f'against_{role[-1]}'] for row in rows]
        spans = [row['remaining_head_contact_controls'][neck]-row['remaining_head_contact_controls'][role]
                 for row in rows if row['remaining_head_contact_controls'][role] is not None]
        align[role] = dict(exact_pose_matches=sum(value == 0 for value in differences),
            median_max_qpos_component_difference=float(np.median(differences)) if differences else None,
            both_supporting_at_cue=len(spans),
            median_remaining_support_change_s=float(np.median(spans)*.02) if spans else None)
    flags_a = [row['success'] for row in analysis['arms']['reference-a']['episodes']]
    flags_b = [row['success'] for row in analysis['arms']['reference-b']['episodes']]
    disagreements = [i for i, (left, right) in enumerate(zip(flags_a, flags_b, strict=True)) if left != right]
    result = dict(status='passed-saved-data-only', sources_sha256=sha(analysis_path), script_sha256=sha(Path(__file__)),
        summary=summary, triggered_worlds=len(worlds), rows=rows, matched_head_support=align,
        official_disagreement_worlds=disagreements, new_physical_steps=0, heldout_used=False, learning_updates=0,
        scope='Descriptive same-world comparison at the candidate trigger; restored starts match, later contact can diverge.')
    write(OUT / 'mechanism-review.json', result)
    print('ROLL_NECK_MECHANISM', summary, align, 'repeat-disagreements', len(disagreements), flush=True)


if __name__ == '__main__':
    main()
