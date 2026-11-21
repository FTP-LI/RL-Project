"""Attribute observed roll progress loss; preserve every frozen verdict.

Decompose geometric-minus-qualified angle into positive rotation without
ground contact and positive rotation discounted for lateral tilt. This is a
trajectory accounting identity, not a causal intervention or new pass rule.
"""
from pathlib import Path
import csv
import json
import numpy as np
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from roll_metrics import progress,rotation

OUT=ROOT/'records/reports/roll-support-diagnosis'


def main():
    OUT.mkdir(exist_ok=True)
    reports=[]
    for role in ('original','control','candidate'):
        reports.append(ROOT/f'records/reports/roll-top-contact/{role}/evaluation-analysis.json')
    for role in ('original','adapted'):
        reports.append(ROOT/f'records/reports/roll-adaptation-screen/{role}-analysis.json')
    rows=[];sources=[]
    for report in reports:
        saved=read(report);assert saved['status']=='passed'
        folder=ROOT/saved['evaluator'];plan=read(folder/'protocol.json')
        path=folder/'forward-roll-trace.npz';assert sha(path)==saved['trace_sha256']
        with np.load(path) as value:t={k:value[k].copy() for k in value.files}
        geometric,supported,r=progress(t)
        delta=np.diff(geometric,axis=0,prepend=geometric[:1])
        flat=np.clip((.866-np.abs(r[...,2,1]))/.366,0,1);flat=flat**2*(3-2*flat)
        positive=np.maximum(delta,0)
        air_loss=positive*(~t['ground_contact'])
        side_loss=positive*t['ground_contact']*(1-flat)
        np.testing.assert_allclose(geometric-supported,np.cumsum(air_loss+side_loss,axis=0),atol=1e-9,rtol=0)
        tilt=np.arccos(np.clip(r[...,2,2],-1,1))
        stable=(abs(t['height']-plan['criteria']['nominal_height_m'])<=plan['criteria']['height_tolerance_m'])
        stable&=(tilt<=np.deg2rad(plan['criteria']['tilt_max_deg']))
        stable&=(np.linalg.norm(t['angular_velocity'],axis=2)<=plan['criteria']['angular_speed_max_rad_s'])
        stable&=t['foot_contact'].all(axis=2)&t['joint_envelope_ok']&t['active']
        for item in saved['rows']:
            i=item['env_index'];valid=t['active'][:,i]
            peak=int(np.argmax(np.where(valid,np.minimum(geometric[:,i],supported[:,i]),-np.inf)))
            row=dict(source=saved['evaluator'],role=saved['role'],seed=plan['seed'],env_index=i,
                success=item['success'],head_top_sequence=item['head_contact_seconds'] is not None,
                geometric_max_deg=float(np.rad2deg(geometric[valid,i]).max()),
                qualified_max_deg=float(np.rad2deg(np.minimum(geometric[:,i],supported[:,i])[valid]).max()),
                qualified_peak_seconds=float(t['time'][peak]),
                airborne_loss_at_peak_deg=float(np.rad2deg(air_loss[:peak+1,i].sum())),
                lateral_loss_at_peak_deg=float(np.rad2deg(side_loss[:peak+1,i].sum())),
                standing_stable_final_second_fraction=float(stable[-50:,i].mean()),
                actual_joint_violation_samples=item['joint_violation_samples'],head_depth_m=item['head_penetration_max_m'])
            rows.append(row)
        sources.append(dict(report=str(report.relative_to(ROOT)),report_sha256=sha(report),trace_sha256=sha(path)))
    failed=[x for x in rows if not x['success']]
    head_ok_failed=[x for x in failed if x['head_top_sequence']]
    assert len(rows)==160
    stats=lambda key:dict(median=float(np.median([x[key] for x in head_ok_failed])),
                         minimum=float(min(x[key] for x in head_ok_failed)),maximum=float(max(x[key] for x in head_ok_failed)))
    result=dict(status='passed',episodes=len(rows),failures=len(failed),
        failed_missing_head_top=sum(not x['head_top_sequence'] for x in failed),
        failed_with_head_top=len(head_ok_failed),
        qualified_angle_in_head_top_failures_deg=stats('qualified_max_deg'),
        airborne_loss_in_head_top_failures_deg=stats('airborne_loss_at_peak_deg'),
        lateral_loss_in_head_top_failures_deg=stats('lateral_loss_at_peak_deg'),
        all_failed_stably_standing_final_second=all(x['standing_stable_final_second_fraction']==1. for x in failed),
        head_top_failures_airborne_loss_exceeds_lateral=sum(x['airborne_loss_at_peak_deg']>x['lateral_loss_at_peak_deg'] for x in head_ok_failed),
        rows=rows,sources=sources,script_sha256=sha(Path(__file__)),learning_updates=0,physics_steps=0,
        old_gates_unchanged=True,scope='Observed trajectory decomposition; no causal or improved-policy claim')
    write(OUT/'analysis.json',result)
    with (OUT/'episodes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','sources')},indent=2))


if __name__=='__main__':main()
