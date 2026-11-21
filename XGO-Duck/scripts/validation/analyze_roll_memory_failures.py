"""Explain closed roll failures and compare the two preassigned env0 videos.

Reconstruct supported/total rotation from every saved quaternion/contact.
Compose actual frames with counters, never generated poses. The control and
candidate videos were both chosen as env0 before seeing outcomes.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from s1_reference_inputs import ROOT,read,sha
from record_io import write

OUT=ROOT/'records/reports/roll-action-memory/startup-recovery'


def curves(t):
    w,x,y,z=np.moveaxis(t['quaternion'],-1,0)
    r00=1-2*(y*y+z*z);r01=2*(x*y-w*z);r10=2*(x*y+w*z);r11=1-2*(x*x+z*z)
    r20=2*(x*z-w*y);r21=2*(y*z+w*x)
    angle=np.unwrap(np.arctan2(-r20,(r00*r11-r10*r01)/np.maximum(np.hypot(r01,r11),1e-6)),axis=0)
    angle-=angle[0]
    delta=np.diff(angle,axis=0,prepend=angle[:1]);flat=np.clip((.866-abs(r21))/.366,0,1)
    flat=flat**2*(3-2*flat)
    supported=np.cumsum(np.minimum(delta,0)+np.maximum(delta,0)*flat*t['ground_contact'],axis=0)
    return np.rad2deg(angle),np.rad2deg(supported)


def main():
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',20)
    large=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',28)
    sheet=Image.new('RGB',(1540,680),'white');draw=ImageDraw.Draw(sheet)
    draw.text((14,8),'滚翻接触对照',font=large,fill='#15334b')
    draw.text((14,45),'同一 seed 1003、预先指定的环境 0；角度依次为几何转角 / 有效支撑转角。',font=font,fill='#15334b')
    results=[]
    for row,arm in enumerate(('control','candidate')):
        a=read(OUT/arm/'analysis.json');p=read(OUT/arm/'progress.json');folder=Path(p['path']);report=read(folder/'evaluation.json')
        assert a['report_sha256']==sha(folder/'evaluation.json')
        source=folder/'forward-roll-trace.npz';assert sha(source)==a['trace_sha256']
        with np.load(source) as value:t={k:value[k].copy() for k in value.files}
        geom,support=curves(t)
        for i,saved in enumerate(a['rows']):
            mask=t['active'][:,i]
            np.testing.assert_allclose([geom[:,i][mask].max(),support[:,i][mask].max()],
                [saved['geometric_max_deg'],saved['supported_max_deg']],atol=2e-4,rtol=0)
        failed=[r for r in a['rows'] if not r['success']]
        reasons=dict(missing_head_top_sequence=sum(r['head_contact_seconds'] is None for r in failed),
            insufficient_supported_rotation=sum(r['head_contact_seconds'] is not None and r['supported_max_deg']<330 for r in failed),
            joint_limit_failures=sum(r['joint_violation_samples']>0 for r in failed),
            head_depth_failures=sum(r['head_penetration_max_m']>.005 for r in failed))
        assert reasons['missing_head_top_sequence']+reasons['insufficient_supported_rotation']==len(failed)
        movie=report['conditions'][0]['video'];video=folder/movie['path'];assert sha(video)==movie['sha256']
        times=np.asarray(movie['frame_times_seconds']);indices=[int(np.argmin(abs(times-t))) for t in (0.,.32,.64,1.2,3.)]
        reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);frames={};count=0
        for i,raw in enumerate(reader):
            if i in indices:frames[i]=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(720,960,3).copy())
            count+=1
        assert count==len(times)==226 and meta['fps']==25
        y=90+row*290;label='原输出' if arm=='control' else '输出滤波'
        draw.text((14,y),label+'｜环境 0 '+('成功' if a['rows'][0]['success'] else '未通过'),font=font,fill='#15334b')
        annotations=[]
        for column,index in enumerate(indices):
            x=10+column*306;sheet.paste(frames[index].resize((300,225),Image.Resampling.LANCZOS),(x,y+30))
            step=round(times[index]/.02);text=f'{times[index]:.2f}s | {geom[step,0]:.0f}° / {support[step,0]:.0f}°'
            draw.text((x,y+258),text,font=font,fill='#15334b');annotations.append(dict(video_frame=index,time=float(times[index]),trace_step=step,geometry_deg=float(geom[step,0]),support_deg=float(support[step,0])))
        results.append(dict(arm=arm,successes=a['successes'],episodes=32,failure_reasons=reasons,failed_episodes=failed,
            trace_sha256=sha(source),video=str(video.relative_to(ROOT)),video_sha256=sha(video),decoded_frames=count,selected_frames=annotations))
    image=ROOT/'docs/reports/assets/roll-output-memory-comparison.jpg';sheet.save(image,quality=92)
    result=dict(status='recomputed-rendered-awaiting-visual-inspection',checked_at=datetime.now(timezone.utc).isoformat(),
        arms=results,image=str(image.relative_to(ROOT)),image_sha256=sha(image),script_sha256=sha(Path(__file__)),
        learning_updates=0,environment_steps=0,original_verdicts_unchanged=True,
        scope='Failure classification and preassigned env0 illustration; numerical outcomes use all32 episodes per arm')
    write(OUT/'failure-review.json',result)
    print(json.dumps({k:([{x:y for x,y in r.items() if x not in ('failed_episodes','selected_frames')} for r in v] if k=='arms' else v) for k,v in result.items()},indent=2))


if __name__=='__main__':main()
