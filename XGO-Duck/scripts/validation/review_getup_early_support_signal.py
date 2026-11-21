"""Screen one bounded early recovery signal on archived development states."""
from pathlib import Path

import numpy as np

from evaluate_getup_pose_router import OUT, PLAN, ROLES, SEEDS
from record_io import write
from review_getup_pose_route_forks import outcomes
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import classify_release, projected_gravity


def review(seed):
    proof_path = OUT/f'seed{seed}-analysis.json'
    proof = read(proof_path)
    assert proof['status'] == 'passed-independent-verification' and proof['seed'] == seed and not proof['heldout_used']
    assert proof['master_sha256'] == sha(PLAN)
    folder = ROOT/proof['record']
    case = proof['arms'][ROLES[-1]]
    prefix_path, branch_path = folder/'trace-shared-prefix.npz', folder/f'trace-{ROLES[-1]}.npz'
    assert sha(prefix_path) == proof['prefix_sha256'] and sha(branch_path) == case['trace_sha256']
    latch_path = folder/'route-readback.npz'
    assert sha(latch_path) == proof['route']['saved_readback_sha256']
    with np.load(latch_path) as source:
        gravity = source['first_gravity'].astype(float)
    assert np.isfinite(gravity).all() and (np.linalg.norm(gravity,axis=1)>0).all()
    criteria = read(folder/'protocol.json')['environment']['criteria']
    directions = classify_release(gravity/np.linalg.norm(gravity,axis=1,keepdims=True),criteria['release_alignment_min'])
    selected = np.asarray(directions) == 'supine'
    labels = np.asarray([r['release_pose'] for r in case['timing']['episodes']])
    actual = labels == 'supine'
    keys = ('time','active','quaternion','foot_contact','native_head_leg_contact')
    with np.load(prefix_path) as a, np.load(branch_path) as b:
        trace = {key:np.concatenate((a[key],b[key])) for key in keys}
    window = (trace['time'] >= 2.3-1e-8)&(trace['time'] <= 3.+1e-8)
    tilt = np.rad2deg(np.arccos(np.clip(-projected_gravity(trace['quaternion'])[...,2],-1.,1.)))
    progress = np.clip((90.-tilt)/(90.-criteria['tilt_max_deg']),0.,1.)
    contact = trace['native_head_leg_contact']
    # Grade the approach to upright; permit partial foot support and award
    # clearance without adding a contact penalty to the original objective.
    signal = progress*trace['foot_contact'].mean(axis=-1)*(~contact)
    signal *= window[:,None]&trace['active']&selected[None]
    assert np.isfinite(signal).all() and 0. <= signal.min() <= signal.max() <= 1.
    assert not signal[~window].any() and not signal[:,~selected].any()
    observed = signal[window].mean(axis=0)
    success = outcomes(case)
    had_contact = contact[(trace['time']>2.) & (trace['time']<=3.)].any(axis=0)
    groups = {}
    for label, mask in (('supine_recovered',actual&success),('supine_missed',actual&~success),
                        ('supine_recovered_with_transient_head_contact',actual&success&had_contact),
                        ('other_pose',~actual)):
        assert mask.any()
        values = observed[mask]
        groups[label] = dict(worlds=int(mask.sum()),median=float(np.median(values)),
                             nonzero=int(np.count_nonzero(values)),quartiles=np.quantile(values,[.25,.75]).tolist())
    return dict(seed=seed,source_sha256=sha(proof_path),same_release_latch=int((selected==actual).sum()),
                latched_other_pose_as_supine=int((selected&~actual).sum()),missed_actual_supine=int((actual&~selected).sum()),
                groups=groups,signal_by_world=observed.tolist())


if __name__ == '__main__':
    assert tuple(SEEDS) == (1062,1063)
    rows = [review(seed) for seed in SEEDS]
    write(OUT/'early-support-signal.json',dict(status='passed-saved-development-review',cohorts=rows,
          script_sha256=sha(Path(__file__)),source_sha256=sha(PLAN),window_seconds=[2.3,3.],physical_substeps=0,
          heldout_used=False,learning_updates=0,
          scope='One prospective bounded reward component scored on saved states. Reward-clock, value and training behavior remain separate.'))
    print('GETUP_EARLY_SUPPORT_SIGNAL',[(r['seed'],r['same_release_latch'],r['groups']) for r in rows],flush=True)
