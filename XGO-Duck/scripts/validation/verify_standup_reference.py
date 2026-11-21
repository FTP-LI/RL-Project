"""Independently recompute recovery holds, interface records and video counts.

Uses captured trajectories only: no ONNX inference, simulation or training.
The stable-run test is implemented with consecutive-sample accumulation,
separately from the evaluator's sliding-window metric implementation.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import imageio_ffmpeg
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'records/reports/standup-reference'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    state=read(OUT/'progress.json');assert state['status']=='completed'
    folder=Path(state['path']);report=read(folder/'evaluation.json');plan=read(folder/'protocol.json')
    assert sha(folder/'protocol.json')==report['protocol_sha256']
    assert report['model_sha256']==plan['model_sha256']==sha(ROOT/plan['model'])
    for name,digest in plan['source_sha256'].items():assert sha(folder/Path(name).name)==digest
    assert [c['condition'] for c in report['conditions']]==plan['poses']
    criteria=plan['criteria'];dt=criteria['step_dt'];h=round(criteria['max_seconds']/dt);n=plan['episodes_per_pose']
    freeze=round(criteria['freeze_seconds']/dt);hold=round(criteria['hold_seconds']/dt)
    ranges=np.asarray(report['joint_limits']);tol=np.deg2rad(plan['applied_joint_limit_tolerance_deg'])
    rows=[]
    for item in report['conditions']:
        pose=item['condition'];trace_path=folder/f'{pose}-trace.npz';assert sha(trace_path)==item['trace_sha256']
        with np.load(trace_path) as t:
            active=t['active'];assert active.shape==(h,n) and active.dtype==bool
            assert not np.any(np.diff(active.astype(int),axis=0)>0)
            np.testing.assert_allclose(t['time'],np.arange(1,h+1)*dt,rtol=0,atol=1e-7)
            assert not t['truncated'][:-1][active[:-1]].any()
            assert np.isfinite(t['observations'][active]).all() and np.isfinite(t['raw_policy_actions']).all()
            np.testing.assert_allclose(t['observations'][:,:,48:][active],0,rtol=0,atol=1e-6)
            np.testing.assert_allclose(t['raw_policy_actions'][:freeze],0,rtol=0,atol=0)
            np.testing.assert_allclose(t['observations'][:freeze,:,34:48],0,rtol=0,atol=0)
            expected=np.broadcast_to(np.asarray(report['simulation_home']),(freeze,n,14))
            np.testing.assert_allclose(t['applied_targets'][:freeze],expected,rtol=0,atol=1e-7)
            np.testing.assert_allclose(t['applied_targets'][freeze:],np.asarray(report['policy_home'])+t['raw_policy_actions'][freeze:],rtol=0,atol=1e-6)
            bounded=((t['joint_positions']>=ranges[:,0]-tol)&(t['joint_positions']<=ranges[:,1]+tol)).all(axis=2)
            assert np.array_equal(bounded,t['joint_envelope_ok'])
            q=t['quaternion'];np.testing.assert_allclose(np.linalg.norm(q,axis=2),1,atol=2e-4,rtol=0)
            w,x,y,z=np.moveaxis(q,-1,0)
            gravity=np.stack((2*(w*y-x*z),-2*(w*x+y*z),2*(x*x+y*y)-1),axis=-1)
            tilt=np.arccos(np.clip(1-2*(x*x+y*y),-1,1))
            stable=(np.abs(t['height']-criteria['nominal_height_m'])<=criteria['height_tolerance_m'])
            stable&=(tilt<=np.deg2rad(criteria['tilt_max_deg']))
            stable&=(np.linalg.norm(t['angular_velocity'],axis=2)<=criteria['angular_speed_max_rad_s'])
            stable&=np.all(t['foot_contact'],axis=2)&bounded&active
            count=0;wrong=0;details=[]
            for i in range(n):
                g=gravity[freeze-1,i];axis=int(np.argmax(np.abs(g)))
                actual='other' if axis==2 or abs(g[axis])<criteria['release_alignment_min'] else \
                    (('prone' if g[0]>0 else 'supine') if axis==0 else ('left-side' if g[1]>0 else 'right-side'))
                streak=0;start=None
                for k in range(freeze,h):
                    streak=streak+1 if stable[k,i] else 0
                    onset=(k-hold+1)*dt
                    if streak>=hold and onset<=criteria['deadline_seconds']+1e-8:
                        start=onset;break
                valid=actual==pose;success=valid and start is not None
                saved=item['episodes'][i]
                assert saved['release_pose']==actual and saved['release_pose_valid']==valid
                assert saved['success']==success and saved['joint_violation_samples']==int((~bounded[freeze:,i]).sum())
                if start is None:assert saved['hold_start_seconds'] is None
                else:np.testing.assert_allclose([saved['hold_start_seconds'],saved['hold_end_seconds']],[start,start+criteria['hold_seconds']],atol=1e-7,rtol=0)
                count+=success;wrong+=not valid
                details.append(dict(env_index=i,success=success,release_pose=actual,hold_start_seconds=start))
            assert item['successes']==count and item['wrong_release_pose']==wrong
            assert item['numeric_gate_passed']==(count/n>=plan['condition_success_rate_min'])
            # Wilson interval describes development trials, not trained-seed robustness.
            p=count/n;z95=1.959963984540054;denom=1+z95*z95/n
            middle=(p+z95*z95/(2*n))/denom;half=z95*np.sqrt(p*(1-p)/n+z95*z95/(4*n*n))/denom
        video=folder/item['video']['path'];assert sha(video)==item['video']['sha256']
        reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);frames=sum(1 for _ in reader)
        assert frames==len(item['video']['frame_times_seconds']) and meta['size']==(960,720) and meta['fps']==25
        rows.append(dict(condition=pose,successes=count,attempts=n,success_fraction=count/n,
            wilson_95=[middle-half,middle+half],wrong_release_pose=wrong,
            trace_sha256=sha(trace_path),video_sha256=sha(video),decoded_frames=frames,episodes=details))
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),report_sha256=sha(folder/'evaluation.json'),
        protocol_sha256=sha(folder/'protocol.json'),verifier_sha256=sha(Path(__file__)),conditions=rows,
        all_conditions_passed=all(c['success_fraction']>=plan['condition_success_rate_min'] for c in rows),
        scope='Independent saved-trace and media checks; development screening only, no walking-transition or hardware claim')
    (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'conditions':[{k:v for k,v in c.items() if k!='episodes'} for c in rows]},indent=2))


if __name__=='__main__':main()
