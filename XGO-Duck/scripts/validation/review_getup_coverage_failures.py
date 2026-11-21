"""Read complete failed recovery trajectories; no new simulation or learning.

Separate actual release coverage from recovery, and record pre-policy numeric
divergence despite an exactly matched starting state. No causal reward claim
can be inferred from a one-seed, one-count outcome difference.
"""
from pathlib import Path
import json
import numpy as np
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from standup_metrics import projected_gravity,classify_release

OUT=ROOT/'records/reports/getup-release-coverage/fixed-evaluation'


def main():
    m=read(OUT/'analysis.json');assert m['status']=='passed' and not m['stage_gate_passed']
    values={};releases={};early={};outcomes={};sources={};starts={}
    for role,v in m['roles'].items():
        record=ROOT/v['record'];p=read(record/'protocol.json');raw=record/'coverage-trace.npz'
        assert sha(raw)==v['trace_sha256']
        with np.load(raw) as f:t={k:f[k] for k in f.files}
        with np.load(record/'paired-start.npz') as f:starts[role]={k:f[k] for k in f.files}
        rows=read(record/'coverage.json')['episodes'];failed=[r for r in rows if r['hold_start_seconds'] is None]
        release=classify_release(projected_gravity(t['quaternion'][99]),p['criteria']['release_alignment_min'])
        assert release==[r['release_pose'] for r in rows]
        groups={pose:[r['env_index'] for r in failed if r['requested_pose']==pose] for pose in ('prone','supine','left-side','right-side')}
        cases=[]
        for row in failed:
            i=row['env_index'];g=projected_gravity(t['quaternion'][:,i]);tilt=np.degrees(np.arccos(np.clip(-g[:,2],-1,1)))
            # A stable-looking single frame is not an uninterrupted3s hold.
            late=slice(349,550)
            cases.append(dict(environment_index=i,requested_pose=row['requested_pose'],release_pose=row['release_pose'],
                final_height_m=float(t['height'][-1,i]),final_tilt_deg=float(tilt[-1]),
                late_median_height_m=float(np.median(t['height'][late,i])),late_median_tilt_deg=float(np.median(tilt[late])),
                late_joint_violation_samples=int((~t['joint_envelope_ok'][late,i]).sum()),
                late_both_feet_fraction=float(t['foot_contact'][late,i].all(-1).mean()),
                native_per_world_head_leg_contact_not_recorded=True))
        values[role]=dict(failed_episodes=len(failed),failed_initial_groups=groups,
            failed_actual_groups={pose:sum(r['release_pose']==pose for r in failed) for pose in ('prone','supine','left-side','right-side','other')},cases=cases)
        releases[role]=release;early[role]=t['qpos'][:100].copy();outcomes[role]=np.array([r['hold_start_seconds'] is not None for r in rows]);sources[role]=sha(raw)
    assert set(starts['original'])==set(starts['control'])
    for key,value in starts['original'].items():np.testing.assert_array_equal(value,starts['control'][key],err_msg=key)
    old=outcomes['original'];new=outcomes['control'];difference=np.abs(early['original']-early['control'])
    result=dict(status='reviewed-no-new-training',roles=values,
        paired_initial_requested_outcomes=dict(both_hold=int((old&new).sum()),control_only=int((~old&new).sum()),
            official_only=int((old&~new).sum()),neither=int((~old&~new).sum())),
        pre_policy_comparison=dict(initial_state_exact=True,maximum_qpos_difference_first2s=float(difference.max()),
            first_control_above_1e_6=next((int(i+1) for i,v in enumerate(difference.max((1,2))) if v>1e-6),None),
            release_classes_differ=sum(a!=b for a,b in zip(releases['original'],releases['control'])),
            units='Mixed qpos coordinates; do not describe maximum as radians',
            interpretation='First2s actions are the same physical freeze. Contact trajectories are not required bitwise equal across processes; mechanism of numeric divergence is not uniquely established.'),
        source_trace_sha256=sources,script_sha256=sha(Path(__file__)),physics_steps_added=0,learning_updates=0,
        gate_unchanged=True,automatic_retry=False,
        next_step='Inspect failed side-to-supine trajectories and native-contact evidence before proposing a new bounded adaptation; retain both closed protocols')
    write(OUT/'failure-review.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('roles','source_trace_sha256')},indent=2))
    for role,v in values.items():print(role,json.dumps({k:a for k,a in v.items() if k!='cases'}))


if __name__=='__main__':main()
