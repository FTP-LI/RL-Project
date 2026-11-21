"""CPU synthetic acceptance/rejection tests for phase-aware roll metrics.

These test policy-independent semantic failures, timing boundaries and rigid
orientation invariance. They make no claim about any real policy's ability.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import numpy as np
from roll_metrics import summarize,progress

ROOT=Path(__file__).resolve().parents[2]
PLAN=dict(step_dt=.02,max_seconds=9.,deadline_seconds=7.,hold_seconds=2.,
          nominal_height_m=.139158,height_tolerance_m=.02,tilt_max_deg=15.,
          angular_speed_max_rad_s=1.,completion_tolerance_deg=30.,
          backward_tolerance_deg=30.,head_penetration_max_m=.005)


def trace(finish=4.,direction=1.,yaw=0.):
    t=np.arange(451)*.02
    angle=np.clip((t-1)/(finish-1),0,1)*2*np.pi*direction
    c,s=np.cos(angle/2),np.sin(angle/2);cy,sy=np.cos(yaw/2),np.sin(yaw/2)
    q=np.stack((cy*c,-sy*s,cy*s,sy*c),axis=1)[:,None,:]
    # Only the head-top orientation is relevant to the synthetic latch.
    head=np.tile([np.sqrt(.5),-np.sqrt(.5),0.,0.],(len(t),1,1))
    feet=np.repeat(((t<1.1)|(t>=finish))[:,None,None],2,axis=2)
    return dict(time=t,quaternion=q,head_quaternion=head,height=np.full((len(t),1),.139158),
        angular_velocity=np.zeros((len(t),1,3)),active=np.ones((len(t),1),bool),
        foot_contact=feet,ground_contact=np.ones((len(t),1),bool),
        head_contact=((np.abs(angle)>np.deg2rad(20))&(np.abs(angle)<np.deg2rad(170)))[:,None],
        head_penetration_m=np.zeros((len(t),1)),joint_envelope_ok=np.ones((len(t),1),bool))


def main():
    cases=[]
    def check(name,t,expected):
        result=summarize(t,PLAN);assert bool(result['successes'])==expected,(name,result)
        cases.append(dict(case=name,expected=expected,result=result['episodes'][0]))
    base=trace();check('forward-supported-headtop-then-hold',base,True)
    check('hold-onset-at-exact-seven-seconds',trace(finish=7.02),True)
    check('hold-onset-late-by-one-control-step',trace(finish=7.04),False)
    check('yaw-invariant',trace(yaw=1.7),True)
    alternate=deepcopy(base);alternate['quaternion'][::2]*=-1;check('quaternion-sign-invariant',alternate,True)
    check('backward-revolution-rejected',trace(direction=-1.),False)
    flight=deepcopy(base);flight['ground_contact'][:]=False;check('unsupported-flip-rejected',flight,False)
    nohead=deepcopy(base);nohead['head_contact'][:]=False;check('no-head-contact-rejected',nohead,False)
    side=deepcopy(base);side['quaternion']=side['quaternion'][...,[0,2,1,3]];check('side-roll-rejected',side,False)
    stand=deepcopy(base);stand['quaternion'][:]=[1,0,0,0];check('standing-without-roll-rejected',stand,False)
    twice=deepcopy(base)
    a=np.clip((twice['time']-1)/3,0,1)*4*np.pi
    twice['quaternion'][:,0,:]=np.stack((np.cos(a/2),np.zeros_like(a),np.sin(a/2),np.zeros_like(a)),axis=1)
    check('repeated-revolutions-rejected',twice,False)
    penetration=deepcopy(base);penetration['head_penetration_m'][100]=.0051;check('head-penetration-rejected',penetration,False)
    limit=deepcopy(base);limit['joint_envelope_ok'][100]=False;check('actual-joint-limit-rejected',limit,False)
    timeout=deepcopy(base);timeout['active'][250:]=False;check('early-timeout-rejected',timeout,False)
    reset=deepcopy(timeout);reset['active'][350:]=True
    try:summarize(reset,PLAN)
    except AssertionError:cases.append(dict(case='reset-cannot-erase-failure',expected='assertion',result='assertion'))
    else:raise AssertionError('Reset was accepted')
    result=dict(status='passed',plan=PLAN,cases=cases,
        metric_sha256=hashlib.sha256((ROOT/'scripts/validation/roll_metrics.py').read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='CPU semantic tests only; no simulation or policy success')
    folder=ROOT/'records/reports/roll-reference';folder.mkdir(exist_ok=True)
    (folder/'metric-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='passed',cases=len(cases),plan=PLAN),indent=2))


if __name__=='__main__':main()
