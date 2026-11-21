"""Extract event-aligned real frames from the already verified state replay.

Fixed3/4s images may miss the brief correction. Use first applied trigger and
the actual pre5s correction peak instead, rounding to the next existing25fps
frame. Label both event and sampled times; do not interpolate or rerender.
"""
from pathlib import Path
import math
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from s1_reference_inputs import ROOT,read,sha
from record_io import write

OUT=ROOT/'records/reports/getup-early-guard'


def main():
    proof=read(OUT/'replay/verification.json');assert proof['status']=='passed'
    summary=read(OUT/'analysis.json');index=read(OUT/'replay-selection.json')['environment_index']
    record=ROOT/summary['roles']['candidate']['record'];source=record/'coverage-trace.npz'
    assert sha(source)==summary['roles']['candidate']['trace_sha256']
    neck=read(ROOT/'configs/experiments/getup_early_guard_study.json')['arms']['candidate']['neck_index']
    with np.load(source) as f:
        active=f['neck_home_active'][:,index];extra=f['applied_targets'][:,index,neck]-f['requested_targets'][:,index,neck]
    first=int(np.flatnonzero(active)[0])+1
    candidates=np.where(active[:250],extra[:250],-np.inf);peak=int(np.argmax(candidates))+1
    assert extra[peak-1]>0 and first>=116
    events={0:['初始跌落'],100:['冻结结束'],400:['恢复截止'],550:['观察结束']}
    for step,label in [(first,'首次触发'),(peak,'修正峰值')]:
        frame_step=2*math.ceil(step/2)
        events.setdefault(frame_step,[]).append(f'{label} {step*.02:.2f}s')
    steps=sorted(events);cols=len(steps);plate=Image.new('RGB',(cols*330,810),'white');draw=ImageDraw.Draw(plate)
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',20);small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',17)
    draw.text((12,8),'早期修正关键时刻',font=font,fill='#15334b')
    draw.text((12,38),f'保存状态回放 · 环境{index} · 事件后首个已有视频帧 · 无插值',font=small,fill='#15334b')
    sources=[]
    for row,v in enumerate(proof['videos']):
        video=ROOT/v['path'];assert sha(video)==v['sha256'];reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader)
        frames={};count=0
        for i,raw in enumerate(reader):
            if i*2 in steps:frames[i*2]=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(480,640,3).copy())
            count+=1
        assert count==276 and meta['fps']==25 and len(frames)==cols
        y=84+row*355;label='官方原样对照' if v['role']=='control' else '早期卡住修正'
        draw.text((12,y),label,font=font,fill='#15334b')
        for col,step in enumerate(steps):
            x=col*330+5;plate.paste(frames[step].resize((320,240),Image.Resampling.LANCZOS),(x,y+70))
            draw.text((x,y+30),f'采样 t={step*.02:.2f}s · 步{step}',font=small,fill='#15334b')
            draw.text((x,y+52),'；'.join(events[step]),font=small,fill='#15334b')
            sources.append(dict(role=v['role'],video=v['path'],video_sha256=v['sha256'],step=step,frame_index=step//2,
                qpos_sha256=v['qpos_sha256_by_frame'][step//2]))
    target=ROOT/'docs/reports/assets/getup-early-guard-events.jpg';plate.save(target,quality=94)
    result=dict(status='rendered-awaiting-visual-review',image=str(target.relative_to(ROOT)),image_sha256=sha(target),
        first_applied_trigger_seconds=first*.02,peak_added_target_seconds=peak*.02,
        peak_added_target_deg=float(np.degrees(extra[peak-1])),sample_steps=steps,sources=sources,
        script_sha256=sha(Path(__file__)),physics_steps=0,optimizer_steps=0,new_robot_frames=False)
    write(OUT/'event-frame-review.json',result);print('EARLY_GUARD_EVENT_FRAMES',result['first_applied_trigger_seconds'],result['peak_added_target_seconds'],result['peak_added_target_deg'])


if __name__=='__main__':main()
