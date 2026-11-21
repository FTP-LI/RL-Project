"""Independent CPU reconstruction of roll geometry, contact order and holds.

Uses quaternion unwrapping and consecutive-sample accumulation, separately
from the collector's wrapped increments and sliding windows. Also verifies
input/action ordering, physical limits, protocol hashes and every video frame.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import imageio_ffmpeg
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'records/reports/roll-reference'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def independent(trace,plan,limits,tolerance):
    dt=plan['step_dt'];n=trace['height'].shape[1];h=round(plan['max_seconds']/dt)+1
    active=trace['active'];assert active.shape==(h,n)
    assert not np.any(np.diff(active.astype(int),axis=0)>0)
    np.testing.assert_allclose(trace['time'],np.arange(h)*dt,atol=1e-7,rtol=0)
    q=trace['quaternion'];np.testing.assert_allclose(np.linalg.norm(q,axis=-1),1,atol=2e-4,rtol=0)
    w,x,y,z=np.moveaxis(q,-1,0)
    r00=1-2*(y*y+z*z);r01=2*(x*y-w*z);r10=2*(x*y+w*z);r11=1-2*(x*x+z*z)
    r20=2*(x*z-w*y);r21=2*(y*z+w*x);r22=1-2*(x*x+y*y)
    angle=np.unwrap(np.arctan2(-r20,(r00*r11-r10*r01)/np.maximum(np.hypot(r01,r11),1e-6)),axis=0)
    angle-=angle[0]
    dq=np.diff(angle,axis=0,prepend=angle[:1])
    flat=np.clip((.866-np.abs(r21))/.366,0,1);flat=flat**2*(3-2*flat)
    supported=np.cumsum(np.minimum(dq,0)+np.maximum(dq,0)*flat*trace['ground_contact'],axis=0)
    hq=trace['head_quaternion'];hw,hx,hy,hz=np.moveaxis(hq,-1,0)
    top=2*(hy*hz+hw*hx)
    bounded=((trace['joint_positions']>=limits[:,0]-tolerance)&(trace['joint_positions']<=limits[:,1]+tolerance)).all(axis=2)
    assert np.array_equal(trace['joint_envelope_ok'],bounded)
    tilt=np.arccos(np.clip(r22,-1,1))
    stable=(np.abs(trace['height']-plan['nominal_height_m'])<=plan['height_tolerance_m'])
    stable&=(tilt<=np.deg2rad(plan['tilt_max_deg']))
    stable&=(np.linalg.norm(trace['angular_velocity'],axis=2)<=plan['angular_speed_max_rad_s'])
    stable&=np.all(trace['foot_contact'],axis=2)&bounded&active
    lower=np.deg2rad(360-plan['completion_tolerance_deg']);upper=np.deg2rad(360+plan['completion_tolerance_deg'])
    hold=round(plan['hold_seconds']/dt);rows=[]
    for i in range(n):
        foot=np.flatnonzero(np.all(trace['foot_contact'][:,i],axis=1)&active[:,i])
        head_candidates=np.flatnonzero((angle[:,i]>np.deg2rad(20))&(angle[:,i]<np.deg2rad(170))&
                                      trace['head_contact'][:,i]&(top[:,i]<-.3)&active[:,i])
        head_candidates=head_candidates[head_candidates>(foot[0] if len(foot) else h)]
        head=int(head_candidates[0]) if len(head_candidates) else None
        finish_candidates=np.flatnonzero((np.minimum(angle[:,i],supported[:,i])>=lower)&active[:,i])
        finish_candidates=finish_candidates[finish_candidates>=(head if head is not None else h)]
        finish=int(finish_candidates[0]) if len(finish_candidates) else None
        streak=0;end=None;onset=None
        if finish is not None:
            for k in range(finish+1,h):
                streak=streak+1 if stable[k,i] and lower<=angle[k,i]<=upper else 0
                if streak>=hold and (k-hold)*dt<=plan['deadline_seconds']+1e-8:
                    end=k;onset=(k-hold)*dt;break
        last=h if end is None else end+1;mask=active[:last,i]
        maxdepth=float(trace['head_penetration_m'][:last,i][mask].max(initial=0))
        good=bool(end is not None and
            np.all(angle[:last,i][mask]<=upper) and
            np.all(angle[:last,i][mask]>=-np.deg2rad(plan['backward_tolerance_deg'])) and
            bounded[:last,i][mask].all() and maxdepth<=plan['head_penetration_max_m'] and
            tilt[0,i]<=np.deg2rad(plan['tilt_max_deg']) and
            abs(trace['height'][0,i]-plan['nominal_height_m'])<=plan['height_tolerance_m'])
        rows.append(dict(env_index=i,success=good,
            head_contact_seconds=None if head is None else head*dt,
            completion_seconds=None if finish is None else finish*dt,
            hold_start_seconds=onset,hold_end_seconds=None if end is None else end*dt,
            geometric_max_deg=float(np.rad2deg(angle[:,i][active[:,i]]).max(initial=0)),
            supported_max_deg=float(np.rad2deg(supported[:,i][active[:,i]]).max(initial=0)),
            head_penetration_max_m=maxdepth,joint_violation_samples=int((~bounded[:,i]&active[:,i]).sum())))
    return rows


def main():
    state=read(OUT/'progress.json');assert state['status']=='completed'
    folder=Path(state['path']);report=read(folder/'evaluation.json');plan=read(folder/'protocol.json')
    assert sha(folder/'protocol.json')==report['protocol_sha256']
    assert sha(ROOT/plan['model'])==plan['model_sha256']==report['model_sha256']
    for name,digest in plan['source_sha256'].items():assert sha(folder/Path(name).name)==digest
    item=report['conditions'][0];trace_path=folder/'forward-roll-trace.npz'
    assert sha(trace_path)==item['trace_sha256']
    with np.load(trace_path) as trace:
        assert np.isfinite(trace['observations'][trace['active']]).all()
        assert not trace['truncated'][1:-1][trace['active'][1:-1]].any()
        np.testing.assert_allclose(trace['observations'][:,:,48:][trace['active']],0,atol=1e-6,rtol=0)
        np.testing.assert_allclose(trace['applied_targets'][1:],np.asarray(report['policy_home'])+trace['raw_policy_actions'][1:],atol=1e-6,rtol=0)
        # Unreset first episodes retain the raw model history, not shifted HOME actions.
        np.testing.assert_allclose(trace['observations'][1:,:,34:48][trace['active'][1:]],
                                  trace['raw_policy_actions'][:-1][trace['active'][1:]],atol=1e-7,rtol=0)
        assert np.all(trace['head_penetration_m']>=0)
        rows=independent(trace,plan['criteria'],np.asarray(report['joint_limits']),np.deg2rad(plan['actual_joint_limit_tolerance_deg']))
    assert len(rows)==plan['episodes']
    for observed,saved in zip(rows,item['episodes']):
        for key,value in observed.items():
            if value is None or isinstance(value,(bool,int)):assert value==saved[key],(key,value,saved[key])
            else:np.testing.assert_allclose(value,saved[key],rtol=0,atol=2e-4,err_msg=key)
    n=len(rows);k=sum(r['success'] for r in rows);p=k/n;z=1.959963984540054;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    assert item['successes']==k and item['numeric_gate_passed']==(p>=plan['success_fraction_min'])
    video=folder/item['video']['path'];assert sha(video)==item['video']['sha256']
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);frames=sum(1 for _ in reader)
    assert meta['fps']==25 and meta['size']==(960,720) and frames==len(item['video']['frame_times_seconds'])
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        numeric_gate_passed=p>=plan['success_fraction_min'],attempts=n,successes=k,
        success_fraction=p,wilson_95=[mid-half,mid+half],episodes=rows,
        decoded_frames=frames,trace_sha256=sha(trace_path),video_sha256=sha(video),
        report_sha256=sha(folder/'evaluation.json'),protocol_sha256=sha(folder/'protocol.json'),
        verifier_sha256=sha(Path(__file__)),scope=plan['scope'])
    for path in (OUT/'analysis.json',folder/'verification.json'):path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))


if __name__=='__main__':main()
