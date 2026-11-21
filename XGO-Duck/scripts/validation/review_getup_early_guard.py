"""CPU review of measured triggers, contact and final posture after early guard.

Use the verified fixed11s first episodes; retain all original failures.
Matched world identifiers do not imply bitwise dynamics or per-case causality.
No new physics, policies, thresholds, training or heldout evaluations.
"""
import numpy as np
from pathlib import Path
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from standup_metrics import projected_gravity

OUT=ROOT/'records/reports/getup-early-guard'


def main():
    a=read(OUT/'analysis.json');assert a['status']=='passed'
    m=read(ROOT/'configs/experiments/getup_early_guard_study.json');roles={};stored={}
    for role,proof in a['roles'].items():
        folder=ROOT/proof['record'];path=folder/'coverage-trace.npz';assert sha(path)==proof['trace_sha256']
        with np.load(path) as f:
            t={k:f[k] for k in ('guard_latched','neck_home_active','native_head_leg_contact','height','quaternion',
                'joint_positions','initial_qpos','initial_qvel','qpos','requested_targets','applied_targets')}
        final=np.asarray(proof['end_hold']['stable_by_environment'],bool)
        assert int(final.sum())==proof['overall']['timely_holds']
        applied=t['neck_home_active'].any(0);shadow=t['guard_latched'].any(0)
        neck=m['arms'][role]['neck_index']
        tilt=np.degrees(np.arccos(np.clip(-projected_gravity(t['quaternion'])[...,2],-1,1)))
        groups={}
        for name,mask in [('failed',~final),('succeeded',final),('triggered_failed',applied&~final),('triggered_succeeded',applied&final)]:
            n=int(mask.sum())
            groups[name]=dict(episodes=n,
                late_head_leg_episodes=int(t['native_head_leg_contact'][399:,mask].any(0).sum()),
                end_height_m=None if n==0 else float(np.median(t['height'][399:,mask])),
                end_tilt_deg=None if n==0 else float(np.median(tilt[399:,mask])),
                neck_actual_deg=None if n==0 else float(np.median(np.degrees(t['joint_positions'][399:,mask,neck]))),
                neck_requested_deg=None if n==0 else float(np.median(np.degrees(t['requested_targets'][399:,mask,neck]))),
                neck_applied_deg=None if n==0 else float(np.median(np.degrees(t['applied_targets'][399:,mask,neck]))))
        first=[]
        for i in np.flatnonzero(shadow):first.append(dict(world=int(i),seconds=int(np.flatnonzero(t['guard_latched'][:,i])[0])*.02,applied=bool(applied[i]),end_stable=bool(final[i])))
        roles[role]=dict(overall=proof['overall'],end_hold=proof['end_hold']['overall'],
            shadow_detected=int(shadow.sum()),applied_episodes=int(applied.sum()),trigger_times=first,groups=groups,
            trace_sha256=sha(path),record=proof['record'])
        stored[role]=dict(final=final,initial_qpos=t['initial_qpos'],initial_qvel=t['initial_qvel'],qpos=t['qpos'][:100])
    c,b=stored['control'],stored['candidate'];delta=abs(c['qpos']-b['qpos']).max((1,2));different=np.flatnonzero(delta>0)
    paired=dict(wins=int((~c['final']&b['final']).sum()),losses=int((c['final']&~b['final']).sum()),
        initial_qpos_exact=bool(np.array_equal(c['initial_qpos'],b['initial_qpos'])),
        initial_qvel_exact=bool(np.array_equal(c['initial_qvel'],b['initial_qvel'])),
        freeze_first_divergence=None if len(different)==0 else int(different[0])+1,
        freeze_qpos_max_abs=float(delta.max()),scope='Descriptive world-index overlap only; pre-intervention solver divergence limits causal attribution')
    result=dict(status='passed-saved-data-only',roles=roles,paired=paired,
        source_result_sha256=sha(OUT/'analysis.json'),script_sha256=sha(Path(__file__)),
        new_physics_steps=0,new_learning_steps=0,heldout_used=False,hardware_io=False)
    write(OUT/'mechanism-review.json',result)
    for role,r in roles.items():print('EARLY_GUARD_REVIEW',role,r['overall']['timely_holds'],r['applied_episodes'],r['groups'],flush=True)
    print('WORLD_OVERLAP',paired,flush=True)


if __name__=='__main__':main()
