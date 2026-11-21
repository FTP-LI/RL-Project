"""Diagnose the fixed roll reference and render true phase keyframes.

Do not change the original 90% gate or resample failures. Separate missing
supported rotation/contact order from failure to stand. The six-frame sheet
shows the continuously recorded environment 0, not the four unrecorded cases.
No simulation, inference, learned weight update, or hardware is performed.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from roll_metrics import progress,rotation

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'records/reports/roll-reference'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    state=read(OUT/'progress.json');assert state['status']=='completed'
    folder=Path(state['path']);report=read(folder/'evaluation.json');verified=read(OUT/'analysis.json')
    assert verified['status']=='passed' and verified['report_sha256']==sha(folder/'evaluation.json')
    plan=read(folder/'protocol.json');criteria=plan['criteria'];item=report['conditions'][0]
    with np.load(folder/'forward-roll-trace.npz') as t:
        assert sha(folder/'forward-roll-trace.npz')==verified['trace_sha256']
        geom,supported,r=progress(t);dq=np.diff(geom,axis=0,prepend=geom[:1])
        flat=np.clip((.866-abs(r[...,2,1]))/.366,0,1);flat=flat**2*(3-2*flat)
        no_support=np.maximum(dq,0)*(~t['ground_contact'])
        side_loss=np.maximum(dq,0)*t['ground_contact']*(1-flat)
        tilt=np.arccos(np.clip(r[...,2,2],-1,1))
        stable=(abs(t['height']-criteria['nominal_height_m'])<=criteria['height_tolerance_m'])
        stable&=tilt<=np.deg2rad(criteria['tilt_max_deg'])
        stable&=np.linalg.norm(t['angular_velocity'],axis=2)<=criteria['angular_speed_max_rad_s']
        stable&=np.all(t['foot_contact'],axis=2)&t['joint_envelope_ok']&t['active']
        hold=round(criteria['hold_seconds']/criteria['step_dt']);rows=[]
        for i,e in enumerate(item['episodes']):
            first=None
            for start in range(1,len(t['time'])-hold+1):
                onset=float(t['time'][start]-criteria['step_dt'])
                if onset>criteria['deadline_seconds']+1e-8:break
                if stable[start:start+hold,i].all():first=onset;break
            rows.append(dict(env_index=i,original_success=e['success'],original_reasons=e['reasons'],
                geometric_max_deg=float(np.rad2deg(geom[:,i]).max()),
                supported_max_deg=float(np.rad2deg(supported[:,i]).max()),
                unsupported_positive_rotation_deg=float(np.rad2deg(no_support[:,i]).sum()),
                lateral_flatness_discount_deg=float(np.rad2deg(side_loss[:,i]).sum()),
                stable_hold_ignoring_roll_start_seconds=first,
                head_contact_samples=int(t['head_contact'][:,i].sum()),
                joint_violation_samples=e['joint_violation_samples'],
                head_penetration_max_m=e['head_penetration_max_m']))
        first=item['episodes'][0]
        middle=np.flatnonzero(geom[:,0]>=np.pi)
        wanted=[0.,first['head_contact_seconds'] or .3,
            float(t['time'][middle[0]]) if len(middle) else .6,
            first['completion_seconds'] or 1.,first['hold_start_seconds'] or 2.,first['hold_end_seconds'] or 4.]
        labels=['初始站立','头顶部接触','翻转中段','完成旋转','开始持稳','持稳结束']
        video_times=np.asarray(item['video']['frame_times_seconds'])
        # A milestone frame must be at or after the measured event. Nearest
        # frame selection could label 0.96 s as completion measured at 0.98 s.
        picks=[min(int(np.searchsorted(video_times,x-1e-9)),len(video_times)-1) for x in wanted]
        trace_times=t['time'].copy();angles=geom[:,0].copy();supported0=supported[:,0].copy()
    video=folder/item['video']['path'];assert sha(video)==verified['video_sha256']
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);frames={};count=0
    for i,raw in enumerate(reader):
        if i in picks:frames[i]=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(meta['size'][1],meta['size'][0],3).copy())
        count+=1
    assert count==verified['decoded_frames']
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',19)
    title=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',27)
    sheet=Image.new('RGB',(1500,960),'white');draw=ImageDraw.Draw(sheet)
    draw.text((14,8),'官方前滚翻',font=title,fill='#15334b')
    draw.text((14,45),'环境 0 的真实录像；该例成功，不能代表四个失败场景。原定 90% 门槛未通过。',font=font,fill='#15334b')
    sources=[]
    for k,(index,label) in enumerate(zip(picks,labels)):
        x=10+(k%3)*500;y=82+(k//3)*438
        sheet.paste(frames[index].resize((480,360),Image.Resampling.LANCZOS),(x,y))
        actual=float(video_times[index]);j=int(np.argmin(abs(trace_times-actual)))
        text=f'{label} · t={actual:.2f} s'
        draw.text((x,y+366),text,font=font,fill='#15334b')
        draw.text((x,y+393),f'累计 {np.rad2deg(angles[j]):.1f}° / 有效支撑 {np.rad2deg(supported0[j]):.1f}°',font=font,fill='#15334b')
        sources.append(dict(frame_index=index,frame_time_seconds=actual,requested_time_seconds=wanted[k],label=label))
    target=ROOT/'docs/reports/assets/roll-phase-keyframes.jpg';sheet.save(target,quality=92)
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        original_gate_passed=verified['numeric_gate_passed'],original_successes=verified['successes'],attempts=verified['attempts'],
        stable_holds_ignoring_rotation_gate=sum(r['stable_hold_ignoring_roll_start_seconds'] is not None for r in rows),
        failed_episodes=[r for r in rows if not r['original_success']],episodes=rows,
        report_sha256=sha(folder/'evaluation.json'),trace_sha256=verified['trace_sha256'],
        source_video_sha256=sha(video),image=str(target.relative_to(ROOT)),image_sha256=sha(target),
        selected_frames=sources,script_sha256=sha(Path(__file__)),original_verdict_rewritten=False,
        scope='Post-hoc diagnostic only; stable-hold diagnostic cannot replace the full forward-roll gate')
    (OUT/'failure-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))


if __name__=='__main__':main()
