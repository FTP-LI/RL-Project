"""Quantify how much positive rotation occurs under load between scoring samples."""

import json

import numpy as np

from review_roll_forward_input_stop import ROOT, arrays, read, sha
from roll_metrics import progress


REPORT = ROOT/'records/reports/roll-forward-input-integration/support-window-force'


def main():
    evidence = REPORT/'force-readback.json'
    receipt = read(evidence)
    assert all(sha(ROOT/name)==digest for name,digest in receipt['source_sha256'].items())
    run = ROOT/'records/evaluations'/receipt['run_id']
    clock = arrays(run/'contact-clock.npz')
    trace = arrays(run/'trace.npz')
    protocol = run/'protocol.json'
    assert sha(protocol)==receipt['protocol_sha256']
    plan = read(protocol)
    nominal_weight = read(run/'contact-clock.json')['nominal_weight_n']
    assert nominal_weight*plan['force_capture']['load_threshold_nominal_weight_fraction']==receipt['threshold_n']
    first,last = (plan['force_capture'][key] for key in ('first_control','last_control'))
    roll_start = plan['initial_stand_controls']
    cut = slice(first+1,last+1)
    prefix = slice(first-roll_start,last-roll_start)
    collected = np.concatenate((clock['before'][:1],clock['after']))
    within_clock,_,_ = progress(dict(quaternion=collected,ground_contact=np.ones(collected.shape[:2],dtype=bool)))
    delta = np.diff(within_clock,axis=0).reshape(last-first,5,plan['worlds'])
    series = {name:trace[name][roll_start:] for name in ('quaternion','ground_contact')}
    control_angle,_,_ = progress(series)
    control_delta = np.diff(control_angle,axis=0)[prefix]
    # The physics recorder serializes raw qpos. The scoring reader's body
    # quaternion is a separately computed field. Align like-for-like first.
    qpos_angle,_,_ = progress(dict(series,quaternion=trace['qpos'][roll_start:,:,3:7]))
    qpos_delta = np.diff(qpos_angle,axis=0)[prefix]
    np.testing.assert_allclose(delta.sum(axis=1),qpos_delta,atol=1e-10,rtol=0.)
    load = clock['normal_force_n'].reshape(last-first,5,plan['worlds'],4)
    loaded = load>receipt['threshold_n']
    gap = ~trace['ground_contact'][cut] & trace['active'][cut] & (trace['phase_next'][cut]==1) & (control_delta>0.)
    rolled = (trace['phase_next'][roll_start:]==2).any(axis=0)
    assert rolled.sum()==receipt['original_roll_success']
    groups = {}
    positive = np.maximum(delta,0.)
    for label, selected in (('roll-pass',rolled),('roll-fail',~rolled)):
        rows, worlds = np.nonzero(gap & selected[None,:])
        assert len(rows)==receipt['groups'][label]['endpoint_gap_samples']
        values = loaded[rows,:,worlds]
        assert np.bincount(values[...,0].sum(axis=1),minlength=6).tolist()==receipt['groups'][label]['load_count_histogram']
        degrees = np.rad2deg(positive[rows,:,worlds])
        under_load = (degrees*values[...,0]).sum(axis=1)
        total = degrees.sum(axis=1)
        usable = total>0.
        endpoint_delta = np.rad2deg(control_delta[rows,worlds])
        continuous = load[rows,:,worlds,0]
        has_later_solve = rows < (last-first-1)
        next_load = loaded[rows[has_later_solve]+1,0,worlds[has_later_solve],0]
        by_world = []
        for world in np.unique(worlds):
            mask = worlds==world
            by_world.append(dict(world=int(world),endpoint_gap_intervals=int(mask.sum()),
                endpoint_excluded_rotation_deg=float(endpoint_delta[mask].sum()),
                positive_rotation_under_load_deg=float(under_load[mask].sum())))
        groups[label] = dict(endpoint_gap_intervals=len(rows),worlds_with_endpoint_gaps=len(by_world),
            positive_rotation_under_load_sum_deg=float(under_load.sum()),
            substep_positive_rotation_sum_deg=float(total.sum()),
            positive_angle_fraction_under_load=float(under_load.sum()/total.sum()) if total.size else None,
            median_interval_fraction_under_load=float(np.median(under_load[usable]/total[usable])) if usable.any() else None,
            endpoint_excluded_rotation_median_per_world_deg=float(np.median([row['endpoint_excluded_rotation_deg']
                                                                            for row in by_world])) if by_world else None,
            nominal_weight_fraction_median_by_substep=np.median(continuous/nominal_weight,axis=0).tolist(),
            head_load_interval_count=int(values[...,1].any(axis=1).sum()),
            either_foot_load_interval_count=int(values[...,2:].any(axis=(1,2)).sum()),
            captured_next_solve_intervals=len(next_load),next_solve_loaded_intervals=int(next_load.sum()),by_world=by_world)
    result = REPORT/'contact-sample-timing.json'
    result.write_text(json.dumps(dict(run_id=receipt['run_id'], source_sha256=sha(evidence),
        angle_basis='Positive physical substep displacement under solver load. Original gate uses integrated endpoint contact.',
        force_threshold_n=receipt['threshold_n'],force_window_seconds=receipt['force_window_seconds'],
        score_vs_qpos_max_control_angle_difference_rad=float(np.abs(control_delta-qpos_delta).max()),
        groups=groups,original_roll_success=receipt['original_roll_success'],original_roll_minimum=receipt['minimum_roll_success'],
        new_physical_steps=0,new_learning_updates=0),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    failed = groups['roll-fail']
    print('SUPPORT_TIMING_REVIEWED',failed['positive_angle_fraction_under_load'],failed['next_solve_loaded_intervals'],
          failed['captured_next_solve_intervals'],failed['head_load_interval_count'],failed['either_foot_load_interval_count'])


if __name__=='__main__':
    main()
