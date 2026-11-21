"""Localize any paired recovery changes on the saved independent evaluation."""
from pathlib import Path

import numpy as np

from evaluate_getup_early_support_ppo import OUT, PLAN, ROLES
from record_io import write
from review_getup_anchor_same_release import describe
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity


KEYS = ('time','native_head_leg_contact','quaternion','joint_envelope_ok','foot_contact','height','angular_velocity')
POSES = ('prone','supine','left-side','right-side','other')


def measurements(trace):
    time = trace['time']
    selected = (time>=2.3-1e-7)&(time<=3.+1e-7)
    assert selected.sum()==36
    tilt = np.rad2deg(np.arccos(np.clip(-projected_gravity(trace['quaternion'])[...,2],-1.,1.)))
    return dict(tilt_3s_deg=tilt[np.argmin(abs(time-3.))],mean_early_tilt_deg=tilt[selected].mean(0),
        early_head_leg_fraction=trace['native_head_leg_contact'][selected].mean(0),
        early_foot_support_fraction=trace['foot_contact'][selected].mean((0,2)))


def main():
    source = OUT/'analysis.json'
    target = OUT/'paired-effect.json'
    proof = read(source)
    assert not target.exists() and proof['status']=='passed-independent-verification'
    assert proof['protocol_sha256']==sha(PLAN) and proof['learning_updates']==0 and not proof['heldout_used']
    spec = read(PLAN)
    record = ROOT/proof['record']
    prefix = record/'trace-shared-prefix.npz'
    assert sha(prefix)==proof['prefix_sha256']
    with np.load(prefix) as archive:
        initial = {key:archive[key] for key in KEYS}
    traces, values, outcome = {}, {}, {}
    poses = None
    for role in ROLES:
        arm = proof['arms'][role]
        file = record/f'trace-{role}.npz'
        assert sha(file)==arm['trace_sha256']
        with np.load(file) as archive:
            trace = {key:np.concatenate((initial[key],archive[key])) for key in KEYS}
        traces[role] = trace
        values[role] = measurements(trace)
        episodes = arm['timing']['episodes']
        assert [ep['env_index'] for ep in episodes]==list(range(len(episodes)))
        actual = np.array([ep['release_pose'] for ep in episodes])
        if poses is None:
            poses = actual
        np.testing.assert_array_equal(poses,actual)
        timing = np.array([ep['hold_start_seconds'] is not None for ep in episodes])
        outcome[role] = timing&np.array(arm['end_hold']['stable_by_environment'])&(actual!='other')
        assert outcome[role].sum()==arm['combined_successes']
    control,candidate = (outcome[role] for role in ROLES[1:])
    masks = dict(all_supine=poses=='supine',rescued=~control&candidate,
                 regressed=control&~candidate,common_failures=~control&~candidate,
                 all_three_failed=~outcome[ROLES[0]]&~control&~candidate)
    groups = {}
    for label,selected in masks.items():
        telemetry = {role:describe(traces[role],selected,spec['environment']['criteria']) for role in ROLES}
        effects = {}
        if selected.any():
            for key in values['control']:
                reference = values['control'][key][selected]
                intervention = values['candidate'][key][selected]
                delta = intervention-reference
                effects[key] = dict(control_median=float(np.median(reference)),candidate_median=float(np.median(intervention)),
                                    paired_delta_median=float(np.median(delta)))
        groups[label] = dict(worlds=np.flatnonzero(selected).tolist(),
            by_actual_pose={pose:int((selected&(poses==pose)).sum()) for pose in POSES},
            measures=effects,telemetry=telemetry)
    result = dict(status='passed-saved-evaluation-review',evaluation_sha256=sha(source),source_sha256=sha(Path(__file__)),
        admission=proof['selection'],groups=groups,physical_substeps=0,learning_updates=0,heldout_used=False,
        scope='Same-release development comparison. Early and late changes are descriptive until replicated.')
    write(target,result)
    print('EARLY_SUPPORT_EFFECT',{k:len(v['worlds']) for k,v in groups.items()},flush=True)


if __name__=='__main__':
    main()
