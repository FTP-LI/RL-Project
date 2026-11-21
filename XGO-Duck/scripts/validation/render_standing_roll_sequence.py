"""Six authentic frames from the preassigned continuous preparation example.

No interpolation or generated poses. Predeclared times cover preparation,
handoff, roll and hold; the illustrated environment0 is not all32 episodes.
"""
from pathlib import Path
import json
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from roll_metrics import progress


def main():
    out=ROOT/'records/reports/roll-stand-preparation'
    result=read(out/'prepared-analysis.json');assert result['status']=='passed' and result['physical_boundary_exact']
    folder=ROOT/result['evaluator'];report=read(folder/'evaluation.json');item=report['conditions'][0]
    video=folder/item['video']['path'];assert sha(video)==result['video_sha256']==item['video']['sha256']
    times=np.asarray(item['video']['frame_times_seconds'])
    wanted=[-.8,-.04,0.,.32,1.2,3.2];indices=[int(np.argmin(abs(times-t))) for t in wanted]
    assert len(set(indices))==6
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');metadata=next(reader);frames={};count=0
    for i,raw in enumerate(reader):
        if i in indices:frames[i]=Image.frombytes('RGB',metadata['size'],raw)
        count+=1
    assert count==len(times)==result['decoded_frames']
    with np.load(folder/'forward-roll-trace.npz') as f:trace={k:f[k].copy() for k in f.files}
    assert sha(folder/'forward-roll-trace.npz')==result['trace_sha256']
    geometric,supported,_=progress(trace)
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',21)
    small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    heading=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',31)
    canvas=Image.new('RGB',(1840,1250),'#edf2f5');draw=ImageDraw.Draw(canvas)
    draw.text((18,15),'站立后滚翻',font=heading,fill='#15334b')
    case=result['rows'][0];case_label='成功' if case['success'] else '未通过'
    draw.text((18,65),f'官方策略 · seed 1007 · 预选环境 0：{case_label}；完整试验 {result["successes"]}/32',font=font,fill='#15334b')
    draw.text((18,105),'单条连续物理轨迹；切换不重置关节或机身。单个成功样本不代表整组通过。',font=font,fill='#15334b')
    labels=['准备','准备末段','切换','滚翻','恢复姿态','后段'];rows=[]
    for j,(index,label) in enumerate(zip(indices,labels)):
        x=10+(j%3)*610;y=155+(j//3)*540;t=float(times[index])
        canvas.paste(frames[index].resize((600,450),Image.Resampling.LANCZOS),(x,y))
        counter=round((t+1)*50)
        caption=f'{label} · t={t:.2f} s · 物理步 {counter}'
        draw.text((x,y+460),caption,font=font,fill='#15334b')
        detail='站立准备；尚未触发滚翻'
        if t>=0:
            step=round(t/.02);g=float(np.rad2deg(geometric[step,0]));s=float(np.rad2deg(supported[step,0]))
            detail=f'几何转角 {g:.1f}° · 有效支撑转角 {s:.1f}°'
        draw.text((x,y+495),detail,font=small,fill='#15334b')
        rows.append(dict(frame_index=index,time_relative_to_roll_trigger=t,physical_counter=counter,caption=caption,detail=detail))
    path=ROOT/'docs/reports/assets/standing-roll-sequence.jpg';canvas.save(path,quality=93)
    manifest=dict(status='rendered-awaiting-visual-review',image=str(path.relative_to(ROOT)),image_sha256=sha(path),
        source_video=str(video.relative_to(ROOT)),source_video_sha256=sha(video),source_trace_sha256=result['trace_sha256'],
        decoded_frames=count,environment_index=0,environment_success=case['success'],group_successes=result['successes'],
        group_episodes=32,frames=rows,generated_robot_images=False,extra_physics_steps=0,script_sha256=sha(Path(__file__)))
    write(out/'sequence-image.json',manifest);print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
