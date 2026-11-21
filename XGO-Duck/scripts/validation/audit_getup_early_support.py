"""Locate recovery failure onset in already verified physical trajectories.

Read-only exploratory diagnosis. Keep the original final-hold labels; never
change scores, replay physics, fit a predictor, or tune on heldout episodes.
Separate requested and applied targets from actual joints. Contact booleans
are sensor evidence, not support forces; their within-control timing differs
as stated by the frozen audit definition.
"""
from pathlib import Path
import csv,json,time
import numpy as np
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from standup_metrics import projected_gravity

PLAN=ROOT/'configs/experiments/getup_early_support_audit.json'
OUT=ROOT/'records/reports/getup-early-support'


def first_run(x,times,length,lo=100,hi=550):
    """Return interval onset for the first complete run, not its confirmation."""
    x=np.asarray(x,bool);run=np.zeros(x.shape[1],np.int64)
    start=np.full(x.shape[1],np.nan)
    for i in range(lo,hi):
        run=np.where(x[i],run+1,0)
        hit=(run==length)&np.isnan(start)
        start[hit]=times[i-length+1]-.02
    return start


def dist(x,limits):return np.maximum(np.maximum(limits[:,0]-x,x-limits[:,1]),0.)


def distribution(x):
    a=np.asarray(x);a=a[np.isfinite(a)]
    return dict(observed=len(a),median=None if not len(a) else float(np.median(a)),
        q25=None if not len(a) else float(np.quantile(a,.25)),q75=None if not len(a) else float(np.quantile(a,.75)))


def main():
    started=time.monotonic();p=read(PLAN);assert sha(ROOT/p['parent_protocol'])==p['parent_sha256']
    master=read(ROOT/p['parent_protocol']);criteria=master['arms']['original']['criteria']
    limits=np.asarray(master['training_arms']['control']['joint_limits'],np.float32)
    neck=master['arms']['original']['neck_index'];dt=.02;times=np.arange(1,551)*dt
    # A25-sample run must start at2.0s and be confirmed at2.5s.
    test=np.zeros((550,2),bool);test[100:125,0]=True
    np.testing.assert_allclose(first_run(test,times,25)[0],2.,atol=1e-12)
    assert np.isnan(first_run(test,times,26)[0]) and np.isnan(first_run(test,times,25)[1])
    roles={};rows=[]
    for role,source in p['sources'].items():
        path=ROOT/source['trace'];proof=read(ROOT/source['proof'])
        assert sha(path)==source['sha256'] and sha(ROOT/source['proof'])==source['proof_sha256']
        assert proof['status']=='passed';stable=np.asarray(proof['end_hold']['stable_by_environment'],bool)
        with np.load(path) as f:
            t={k:f[k] for k in ('time','height','quaternion','angular_velocity','joint_positions','foot_contact',
                'joint_envelope_ok','native_head_leg_contact','native_contact_eligible','requested_targets','applied_targets')}
        np.testing.assert_allclose(t['time'],times,atol=1e-8,rtol=0)
        assert t['height'].shape==(550,320) and t['foot_contact'].shape==(550,320,2)
        assert all(np.isfinite(v).all() for v in t.values())
        tilt=np.degrees(np.arccos(np.clip(-projected_gravity(t['quaternion'])[...,2],-1,1)))
        feet=t['foot_contact'].all(-1);contact=t['native_head_leg_contact']
        raw=dist(t['requested_targets'],limits);applied=dist(t['applied_targets'],limits)
        kinematic=(abs(t['height']-criteria['nominal_height_m'])<=criteria['height_tolerance_m'])
        kinematic &= (tilt<=criteria['tilt_max_deg']) & feet & t['joint_envelope_ok']
        kinematic &= np.linalg.norm(t['angular_velocity'],axis=-1)<=criteria['angular_speed_max_rad_s']
        np.testing.assert_array_equal(kinematic[400:].all(0),stable)
        onset={'head_leg_0.5s':first_run(contact,times,25),
            'eligible_head_leg_0.5s':first_run(contact&t['native_contact_eligible'],times,25),
            'neck_request_outside_0.5s':first_run(raw[...,neck]>1e-6,times,25),
            'both_feet_0.5s':first_run(feet,times,25),
            'first_original_stable_sample':first_run(kinematic,times,1)}
        groups={}
        for group,mask in (('all',np.ones(320,bool)),('end_stable',stable),('end_failed',~stable)):
            points=[]
            for second in p['points_seconds']:
                i=round(second/dt)-1
                points.append(dict(seconds=second,episodes=int(mask.sum()),
                    height_m=distribution(t['height'][i,mask]),tilt_deg=distribution(tilt[i,mask]),
                    neck_actual_deg=distribution(np.degrees(t['joint_positions'][i,mask,neck])),
                    neck_requested_deg=distribution(np.degrees(t['requested_targets'][i,mask,neck])),
                    neck_applied_deg=distribution(np.degrees(t['applied_targets'][i,mask,neck])),
                    both_feet=int(feet[i,mask].sum()),head_leg=int(contact[i,mask].sum()),
                    original_stable=int(kinematic[i,mask].sum())))
            windows=[]
            for lo,hi in p['windows_seconds']:
                sl=slice(round(lo/dt),round(hi/dt))
                windows.append(dict(seconds=[lo,hi],episodes=int(mask.sum()),
                    both_feet_fraction=float(feet[sl,mask].mean()),
                    head_leg_fraction=float(contact[sl,mask].mean()),
                    any_request_outside_fraction=float((raw[sl,mask]>1e-6).any(-1).mean()),
                    any_applied_outside_fraction=float((applied[sl,mask]>1e-6).any(-1).mean())))
            events={key:dict(distribution=distribution(v[mask]),
                started_by3s=int((np.isfinite(v[mask])&(v[mask]<=3.+1e-8)).sum()),
                started_before5s=int((np.isfinite(v[mask])&(v[mask]<5.-1e-8)).sum())) for key,v in onset.items()}
            groups[group]=dict(episodes=int(mask.sum()),points=points,windows=windows,events=events)
        coverage=read(ROOT/proof['record']/'coverage.json')['episodes'];assert len(coverage)==320
        for i,row in enumerate(coverage):
            assert row['env_index']==i
            rows.append(dict(role=role,world=i,requested_pose=row['requested_pose'],release_pose=row['release_pose'],
                end_stable=bool(stable[i]),**{key:None if not np.isfinite(v[i]) else float(v[i]) for key,v in onset.items()}))
        roles[role]=groups
        print('EARLY_SUPPORT',role,'failed',int((~stable).sum()),'contact0.5s starts<=3',
            groups['end_failed']['events']['head_leg_0.5s']['started_by3s'],
            'contact at3',groups['end_failed']['points'][3]['head_leg'],flush=True)
    OUT.mkdir(exist_ok=True)
    with (OUT/'episodes.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='passed-saved-data-only',roles=roles,episodes=len(rows),
        audit_definition_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),csv_sha256=sha(OUT/'episodes.csv'),
        original_final_labels_recomputed_exact=True,sources=p['sources'],sensor_timing=p['sensor_timing'],
        new_physics_steps=0,new_learning_steps=0,heldout_used=False,hardware_io=False,
        scope=p['causal_scope'],elapsed_seconds=time.monotonic()-started)
    write(OUT/'analysis.json',result);print('EARLY_SUPPORT_AUDIT_DONE',len(rows),'saved episodes',flush=True)


if __name__=='__main__':main()
