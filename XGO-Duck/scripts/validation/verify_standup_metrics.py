"""Boundary checks for recovery timing and pose classification; CPU only."""
from pathlib import Path
import json
import numpy as np
from standup_metrics import summarize,projected_gravity,classify_release

ROOT=Path(__file__).resolve().parents[2]


def main():
    plan=dict(step_dt=.02,max_seconds=11,freeze_seconds=2,deadline_seconds=8,hold_seconds=3,
        nominal_height_m=.139158,height_tolerance_m=.02,tilt_max_deg=15,angular_speed_max_rad_s=1,
        release_alignment_min=.75)
    q=np.sqrt(.5)
    poses=np.array([[q,0,q,0],[q,0,-q,0],[q,-q,0,0],[q,q,0,0]])
    assert classify_release(projected_gravity(poses))==['prone','supine','left-side','right-side']
    n=6;h=550
    trace=dict(time=np.arange(1,h+1)*.02,height=np.full((h,n),.139158),
        quaternion=np.tile([1.,0,0,0],(h,n,1)),angular_velocity=np.zeros((h,n,3)),
        foot_contact=np.ones((h,n,2),dtype=bool),joint_envelope_ok=np.ones((h,n),dtype=bool),
        active=np.ones((h,n),dtype=bool),requested_pose=['prone']*n)
    trace['quaternion'][:100]=poses[0]
    # Exactly 8.00 s onset + 3.00 s hold succeeds. One step later fails.
    trace['quaternion'][100:400,0]=poses[0]
    trace['quaternion'][100:401,1]=poses[0]
    # An interruption leaves neither side a complete hold.
    trace['quaternion'][100:400,2]=poses[0];trace['foot_contact'][450,2,0]=False
    # Wrong initial stratum, missing final support, and a terminated trajectory.
    trace['quaternion'][:100,3]=poses[1]
    trace['foot_contact'][100:,4,0]=False
    trace['active'][200:,5]=False
    result=summarize(trace,plan)
    assert [x['success'] for x in result['episodes']]==[True,False,False,False,False,False]
    assert result['episodes'][0]['hold_start_seconds']==8.
    assert result['episodes'][0]['hold_end_seconds']==11.
    out=ROOT/'records/reports/standup-reference';out.mkdir(exist_ok=True)
    report=dict(status='passed',checks=['four-pose-axis-convention','deadline-exactly-included',
        'one-step-late-rejected','broken-hold-rejected','wrong-stratum-rejected','missing-support-rejected','termination-excluded'],
        simulation_started=False,plan=plan,synthetic_results=result)
    (out/'metric-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('plan','synthetic_results')},indent=2))


if __name__=='__main__':main()
