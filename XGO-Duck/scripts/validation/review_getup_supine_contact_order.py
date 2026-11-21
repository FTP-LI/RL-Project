"""Compare contact and tilt timing for paired specified-supine outcomes."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from review_getup_nominal_supine import REPORTS, ROLES
from standup_metrics import projected_gravity


OUT = ROOT/'records/reports/getup-side-to-supine-replay/evaluation'
FIELDS = ('time','quaternion','foot_contact','native_head_leg_contact','angular_velocity')
DWELL = 5


def first_dwell(mask):
    start = np.flatnonzero(np.convolve(mask.astype(int),np.ones(DWELL,dtype=int),mode='valid')==DWELL)
    return int(start[0]) if len(start) else None


def first_time(time,mask,dwell=False):
    hit = first_dwell(mask) if dwell else next(iter(np.flatnonzero(mask)),None)
    return float(time[hit]) if hit is not None else None


def locate(time,one):
    return dict(
        first_head_leg=first_time(time,one['contact']),
        first_sustained_head_leg=first_time(time,one['contact'],True),
        first_sustained_both_feet=first_time(time,one['feet'],True),
        first_tilt_30=first_time(time,one['tilt']<=30.),
        first_tilt_15=first_time(time,one['tilt']<=15.),
        first_sustained_upright_without_head_leg=first_time(time,
            (one['tilt']<=15.)&~one['contact']&one['feet'],True),
        signed_pitch_velocity_at_3000_ms=float(one['pitch'][np.argmin(abs(time-3.))]),
    )


def review(folder):
    source = ROOT/'records/reports'/folder/'analysis.json'
    proof = read(source)
    record = ROOT/proof['record']
    assert proof['status']=='passed-independent-verification' and not proof['heldout_used']
    assert not proof['learning_updates'] and sha(record/'protocol.json')==proof['protocol_sha256']
    prefix = record/'trace-shared-prefix.npz'
    assert sha(prefix)==proof['prefix_sha256']
    with np.load(prefix) as data:
        initial = {key:data[key] for key in FIELDS}
    per_role, outcomes = {}, {}
    requested = None
    for role in ROLES:
        arm = proof['arms'][role]
        episodes = arm['timing']['episodes']
        chosen = [x['env_index'] for x in episodes if x['requested_pose']=='supine']
        assert len(chosen)==32 and all(episodes[world]['release_pose']=='supine' for world in chosen)
        if requested is None:
            requested = chosen
        assert requested==chosen
        outcomes[role] = np.array([episodes[world]['success'] and arm['end_hold']['stable_by_environment'][world] for world in chosen])
        path = record/f'trace-{role}.npz'
        assert sha(path)==arm['trace_sha256']
        with np.load(path) as data:
            trace = {key:np.concatenate((initial[key],data[key])) for key in FIELDS}
        take = trace['time']>2.+1e-8
        time = trace['time'][take]
        gravity = projected_gravity(trace['quaternion'][take][:,chosen])
        tilt = np.rad2deg(np.arccos(np.clip(-gravity[...,2],-1.,1.)))
        per_role[role] = [dict(tilt=tilt[:,i],contact=trace['native_head_leg_contact'][take,world],
            feet=trace['foot_contact'][take,world].all(axis=-1),pitch=trace['angular_velocity'][take,world,1])
            for i,world in enumerate(chosen)]
    comparison = []
    for i,world in enumerate(requested):
        successful = [role for role in ROLES if outcomes[role][i]]
        failed = [role for role in ROLES if not outcomes[role][i]]
        if not failed:
            continue
        times = {role:locate(time,per_role[role][i]) for role in ROLES}
        pairs = []
        for good in successful:
            for bad in failed:
                difference = per_role[bad][i]['tilt']-per_role[good][i]['tilt']
                diverged = first_time(time,difference>=5.,True)
                contact = times[bad]['first_sustained_head_leg']
                pairs.append(dict(successful=good,failed=bad,first_5deg_gap=diverged,
                    gap_minus_failed_sustained_contact_seconds=None if diverged is None or contact is None else diverged-contact))
        comparison.append(dict(world=world,successful=successful,failed=failed,events=times,paired=pairs))
    return dict(source_sha256=sha(source),record=proof['record'],worlds=32,comparisons=comparison)


def main():
    evidence = OUT/'nominal-supine-contact-order.json'
    assert not evidence.exists()
    batches = {name:review(name) for name in REPORTS}
    write(evidence,dict(status='passed-paired-saved-review',scope='Descriptive timing. Earlier onset does not identify causality.',
        batches=batches,source_sha256=sha(Path(__file__)),physical_substeps=0,learning_updates=0,heldout_used=False))
    print('SUPINE_CONTACT_ORDER',[(name,len(row['comparisons']),sum(len(case['paired']) for case in row['comparisons']))
                                 for name,row in batches.items()],flush=True)


if __name__=='__main__':
    main()
