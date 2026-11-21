"""Compose true forward frames: official teacher, failed hip run and bootstrap.

Uses the same preselected 2/8/16 second timestamps and environment 0 as S1.
Full source frames stay intact; labels are outside the frame. Quantitative
claims still use all 32 episodes, not these illustrative examples.
"""
from pathlib import Path
import json
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from s1_reference_inputs import ROOT,read,sha
from analyze_walking_suite import signed_foot_gap_mm

OUT=ROOT/'records/reports/official-actor-bootstrap'


def main():
    assert read(OUT/'training-verification.json')['status']=='passed'
    reference=read(ROOT/'records/reports/s1-evaluation/progress.json')['models']['official-reference']
    failed=read(ROOT/'records/reports/s2-hip-repair/progress.json')['models']['candidate']
    current=dict(five=read(OUT/'evaluation-progress.json')['evaluation'])
    models=[('official','官方参照',reference),('failed','既有内收模型',failed),('bootstrap','官方初始化后',current)]
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',21)
    title=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',32)
    sheet=Image.new('RGB',(1980,1680),'white');draw=ImageDraw.Draw(sheet)
    draw.text((25,14),'初始化姿态对照',font=title,fill='#183047');sources=[]
    for row,(role,label,model) in enumerate(models):
        folder=Path(model['five']);report=read(folder/'evaluation.json')
        assert read(folder/'verification.json')['status']=='passed'
        item=next(x for x in report['conditions'] if x['condition']=='forward')
        video=folder/item['video']['path'];assert sha(video)==item['video']['sha256']
        indices={item['video']['frame_times_seconds'].index(t):t for t in (2.,8.,16.)}
        reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);w,h=meta['size']
        frames={i:Image.fromarray(np.frombuffer(raw,np.uint8).reshape(h,w,3).copy()) for i,raw in enumerate(reader) if i in indices}
        names=report['model_metadata']['joint_names'].split(',');hips=[names.index(s+'_hip_roll') for s in ('left','right')]
        path=folder/'forward-trace.npz'
        with np.load(path,allow_pickle=False) as trace:
            gaps=signed_foot_gap_mm(trace['foot_positions'],trace['root_quaternion'])
            for col,(index,t) in enumerate(indices.items()):
                k=round(t/report['step_dt'])-1;x=15+col*655;y=65+row*520
                angles=np.rad2deg(trace['joint_positions'][k,0,hips]);gap=float(gaps[k,0])
                sheet.paste(frames[index].resize((640,480),Image.Resampling.LANCZOS),(x,y))
                annotation=f'{label} · {t:.0f} s | 髋角 {angles[0]:.1f}° / {angles[1]:.1f}° | 脚距 {gap:.1f} mm'
                assert draw.textbbox((0,0),annotation,font=font)[2]<=650
                draw.text((x+3,y+482),annotation,font=font,fill='#183047')
                sources.append(dict(role=role,time_seconds=t,frame_index=index,trace_index=k,
                    video=str(video.relative_to(ROOT)),video_sha256=sha(video),trace_sha256=sha(path),
                    hip_angles_deg=angles.tolist(),signed_foot_projection_mm=gap))
    draw.text((25,1633),'环境 0；前进指令 0.15 m/s。脚距是躯干横轴投影；效果判断使用全部 32 个 episode。',font=font,fill='#183047')
    path=ROOT/'docs/reports/assets/official-bootstrap-keyframes.jpg';sheet.save(path,quality=92)
    result=dict(status='rendered-awaiting-visual-inspection',panel=str(path.relative_to(ROOT)),sha256=sha(path),
        script_sha256=sha(Path(__file__)),frames=sources,generated_robot_images=False)
    (OUT/'media-provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2))


if __name__=='__main__':main()
