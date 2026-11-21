"""Complete replay of a shared pre-intervention jam from saved physical states.

The fixed case rule is independent of success. Both original results remain
failed. No interpolation, new episodes, optimizer, state reset or integration.
Frames are explicitly labeled as saved-state replay, separate from source
camera recordings. Run graphics in the one visible VS Code task terminal.
"""
from pathlib import Path
import os,json,sys,hashlib
import numpy as np
import torch
import mujoco
import mediapy
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from mjlab.scene import Scene
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import read,sha
from record_io import write
from getup_neck_relief_pair import checked,prepare
from common_scene_contract import model_arrays,compare_arrays

OUT=ROOT/'records/reports/getup-neck-relief/replay'


def main():
    assert os.isatty(1),'Use original visible VS Code shared task terminal'
    assert not OUT.exists(),'Keep existing published replay'
    master=checked();selection=read(OUT.parent/'replay-selection.json');index=selection['environment_index'];assert index is not None
    summary=read(OUT.parent/'analysis.json');OUT.mkdir()
    cfg,_,_=prepare(master['arms']['control']);model=Scene(cfg.env.scene,device='cpu').compile()
    root=[i for i in range(model.nbody) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '').endswith('trunk_base')];assert len(root)==1
    data=mujoco.MjData(model);camera=mujoco.MjvCamera();camera.distance=.8;camera.azimuth=135;camera.elevation=-15
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',17);small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',15);heading=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',21)
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,640);model.vis.global_.offheight=max(model.vis.global_.offheight,480)
    plate=Image.new('RGB',(1980,820),'white');draw=ImageDraw.Draw(plate)
    draw.text((12,8),'末段解困过程',font=heading,fill='#15334b')
    draw.text((12,42),f'保存状态回放 · 环境{index} · 选取不依赖成功结果 · 两组均未及时持稳',font=heading,fill='#15334b')
    samples=(0,100,250,324,400,550);labels=('初始跌落','冻结结束','末段开始','颈部响应','恢复截止','观察结束')
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe());videos=[]
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        for row,(role,label) in enumerate((('control','官方原样对照'),('candidate','末段颈部解困'))):
            record=ROOT/summary['roles'][role]['record'];source=record/'coverage-trace.npz';assert sha(source)==selection['source_trace_sha256'][role]
            with np.load(record/'scene-static-fields.npz') as f:compare_arrays(model_arrays(model),{k:f[k] for k in f.files})
            with np.load(source) as f:
                q=f['qpos'][:,index].copy();v=f['qvel'][:,index].copy();initial=f['initial_qpos'][index].copy();initial_v=f['initial_qvel'][index].copy()
                correction=f['neck_correction'][:,index].copy();contact=f['native_head_leg_contact'][:,index].copy()
            output=OUT/f'{role}-env{index}-complete.mp4';hashes=[];steps=[];y=95+row*355
            draw.text((12,y),label,font=heading,fill='#15334b')
            with mediapy.VideoWriter(str(output),shape=(480,640),fps=25,qp=22) as writer:
                for step in range(0,551,2):
                    pose=initial if step==0 else q[step-1];velocity=initial_v if step==0 else v[step-1]
                    data.qpos[:]=pose;data.qvel[:]=velocity;mujoco.mj_forward(model,data);np.testing.assert_array_equal(data.qpos,pose)
                    camera.lookat[:]=data.xpos[root[0]];renderer.update_scene(data,camera=camera)
                    frame=Image.fromarray(renderer.render().copy());d=ImageDraw.Draw(frame)
                    d.rectangle((0,0,640,78),fill='#11202d')
                    delta=0. if step==0 else float(np.degrees(correction[step-1]))
                    d.text((8,4),f'保存状态回放 | {role} | 环境{index} | t={step*.02:.2f} s',font=font,fill='white')
                    d.text((8,29),f'颈部目标修正 +{delta:.1f}° · 原始独立评估未通过',font=font,fill='white')
                    d.text((8,54),'连续保存物理状态 · 无插值 · 额外物理步0',font=small,fill='white')
                    writer.add_image(np.asarray(frame));hashes.append(hashlib.sha256(pose.tobytes()).hexdigest());steps.append(step)
                    if step in samples:
                        col=samples.index(step);x=10+col*330;plate.paste(frame.resize((320,240),Image.Resampling.LANCZOS),(x,y+65))
                        draw.text((x,y+36),f'{step*.02:g} s · {labels[col]}',font=heading,fill='#15334b')
                        note='接触：初态' if step==0 else ('原始末子步：头腿接触' if contact[step-1] else '原始末子步：头腿无接触')
                        draw.text((x,y+310),note,font=small,fill='#15334b')
            reader=imageio_ffmpeg.read_frames(str(output),pix_fmt='rgb24');meta=next(reader);count=sum(1 for _ in reader)
            assert count==276 and meta['fps']==25 and len(steps)==276
            expected=[hashlib.sha256((initial if step==0 else q[step-1]).tobytes()).hexdigest() for step in steps];assert expected==hashes
            videos.append(dict(path=str(output.relative_to(ROOT)),sha256=sha(output),environment_index=index,role=role,
                frames=count,fps=25,control_steps=steps,qpos_sha256_by_frame=hashes,episode_times_seconds=[s*.02 for s in steps],
                source_trace_sha256=sha(source),original_source_camera=False,physics_steps=0))
    picture=ROOT/'docs/reports/assets/getup-neck-relief-shared-jam.jpg';plate.save(picture,quality=94)
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',videos=videos,image=str(picture.relative_to(ROOT)),image_sha256=sha(picture),
        selection_sha256=sha(OUT.parent/'replay-selection.json'),script_sha256=sha(Path(__file__)),
        physics_steps=0,optimizer_steps=0,interpolation=False,original_source_camera=False,
        scope='Saved-state visualization only; original full-episode results remain failed; contact labels refer to recorded native last-substep sensor, geometry uses saved end-of-control qpos')
    write(OUT/'verification.json',result);print('GETUP_SHARED_JAM_REPLAY_VERIFIED',index,'2 x276frames; physics_steps=0',flush=True)


if __name__=='__main__':main()
