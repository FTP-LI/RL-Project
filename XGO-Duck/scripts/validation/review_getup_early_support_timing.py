"""Measure early supine support and tilt events on the two saved development cohorts."""
from pathlib import Path

import numpy as np

from evaluate_getup_pose_router import OUT, PLAN, ROLES, SEEDS
from record_io import write
from review_getup_pose_route_forks import outcomes
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity


def first_confirmed(time, condition, width):
    indices = np.flatnonzero(np.convolve(condition, np.ones(width, np.int32), mode='valid') == width)
    return float(time[indices[0]+width-1]) if len(indices) else None


def review(seed):
    source = OUT/f'seed{seed}-analysis.json'
    proof = read(source)
    assert proof['status'] == 'passed-independent-verification' and proof['master_sha256'] == sha(PLAN)
    assert not proof['heldout_used'] and seed == proof['seed']
    record = ROOT/proof['record']
    routed = proof['arms'][ROLES[-1]]
    cases = routed['timing']['episodes']
    passed = outcomes(routed)
    assert len(cases) == len(passed) and passed.sum() == routed['combined_successes']
    select = [row['env_index'] for row in cases if row['release_pose'] == 'supine']
    criteria = read(record/'protocol.json')['environment']['criteria']
    prefix, branch = record/'trace-shared-prefix.npz', record/f'trace-{ROLES[-1]}.npz'
    assert sha(prefix) == proof['prefix_sha256'] and sha(branch) == routed['trace_sha256']
    keys = ('time', 'active', 'quaternion', 'foot_contact', 'native_head_leg_contact')
    with np.load(prefix) as a, np.load(branch) as b:
        trace = {key:np.concatenate((a[key], b[key])) for key in keys}
    time = trace['time']
    np.testing.assert_allclose(np.diff(time), .02, atol=1e-7, rtol=0)
    assert np.isclose(criteria['freeze_seconds'],2.)
    early = (time > criteria['freeze_seconds']+1e-8) & (time <= 3.+1e-8)
    upright = np.rad2deg(np.arccos(np.clip(-projected_gravity(trace['quaternion'])[...,2],-1.,1.)))
    rows = []
    for world in select:
        valid = trace['active'][early,world]
        head = trace['native_head_leg_contact'][early,world]
        feet = trace['foot_contact'][early,world].all(axis=-1)
        tilt = upright[early,world]
        t = time[early]
        rows.append(dict(world=world, requested=cases[world]['requested_pose'], completed_recovery=bool(passed[world]),
            first_head_leg_contact=first_confirmed(t, valid & head, 1),
            sustained_two_feet=first_confirmed(t, valid & feet, 5),
            sustained_two_feet_clear=first_confirmed(t, valid & feet & ~head, 5),
            sustained_upright=first_confirmed(t, valid & (tilt <= criteria['tilt_max_deg']), 5),
            sustained_upright_feet_clear=first_confirmed(t, valid & feet & ~head & (tilt <= criteria['tilt_max_deg']), 5),
            initial_head_contact_fraction=float(head[valid].mean()),
            original_hold_start_seconds=cases[world]['hold_start_seconds']))
    summary = {}
    for label, success in (('recovered',True),('missed',False)):
        subset = [row for row in rows if row['completed_recovery'] == success]
        events = {}
        for name in ('first_head_leg_contact','sustained_two_feet','sustained_two_feet_clear',
                     'sustained_upright','sustained_upright_feet_clear'):
            times = [row[name] for row in subset if row[name] is not None]
            events[name] = dict(count=len(times), median_confirmation_seconds=float(np.median(times)) if times else None)
        summary[label] = dict(worlds=len(subset), confirmed_by_3s=events)
    return dict(seed=seed, source_sha256=sha(source), release_seconds=2., original_tilt_limit_degrees=criteria['tilt_max_deg'],
                summary=summary, worlds=rows)


if __name__ == '__main__':
    assert tuple(SEEDS) == (1062,1063)
    results = [review(seed) for seed in SEEDS]
    write(OUT/'early-support-timing.json',dict(status='passed-saved-development-review',cohorts=results,
          window_seconds=[2.,3.],event_confirmation_controls=5,source_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),
          physical_substeps=0,learning_updates=0,heldout_used=False,
          scope='Events are confirmed at recorded 20 ms controls. Same saved states, no substep timing or counterfactual inference.'))
    print('GETUP_EARLY_SUPPORT_TIMING',{r['seed']:r['summary'] for r in results},flush=True)
