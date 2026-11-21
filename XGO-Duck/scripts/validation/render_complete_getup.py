"""Complete8s playback of saved physical states, without a new rollout.

The original two3s recordings remain unchanged. Fill the whole timeline by
rendering every second saved control state, not by interpolating or joining
the clips across a gap. Label this as state replay rather than source-camera
footage. Run EGL only in the single visible VS Code task terminal.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import sys
import numpy as np
import mujoco
import torch
import mediapy
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import read,sha
from getup_contact_training import prepare
from common_scene_contract import model_arrays,compare_arrays
from mjlab.scene import Scene
from record_io import write

OUT=ROOT/'records/reports/action-video'


def main():
    assert os.isatty(1),'Use the existing visible VS Code shared task terminal'
    assert not (OUT/'replay.json').exists(),'Keep published replay and provenance'
    proof=read(ROOT/'records/reports/getup-value-bootstrap/startup-recovery/analysis.json')
    assert proof['status']=='passed';record=ROOT/proof['record'];source=record/'value-data.npz'
    assert sha(source)==proof['dataset_sha256']
    plan=read(record/'protocol.json');cfg,_,_=prepare(plan)
    model=Scene(cfg.env.scene,device='cpu').compile()
    with np.load(record/'scene-static-fields.npz') as f:compare_arrays(model_arrays(model),{k:f[k] for k in f.files})
    with np.load(source) as f:
        qpos=f['qpos'].copy();qvel=f['qvel'].copy();contacts=(f['head_leg']&f['eligible']).copy()
    with np.load(record/'live-startup.npz') as f:initial_q=f['qpos'].copy();initial_v=f['qvel'].copy()
    assert qpos.shape==(400,32,21) and np.isfinite(qpos).all() and np.isfinite(qvel).all()
    scores=contacts[300:,:32].sum(0);selected=[0]
    if scores.max()>0 and int(np.argmax(scores))!=0:selected.append(int(np.argmax(scores)))
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,640)
    model.vis.global_.offheight=max(model.vis.global_.offheight,480)
    root=[i for i in range(model.nbody) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '').endswith('trunk_base')]
    assert len(root)==1
    data=mujoco.MjData(model);camera=mujoco.MjvCamera();camera.distance=.8;camera.azimuth=135;camera.elevation=-15
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',17)
    small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',15)
    platefont=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',20)
    snapshots=(0,100,150,250,400);labels=('初始跌落','冻结结束','主动控制 1 s','主动控制 3 s','训练段结束')
    plate=Image.new('RGB',(1620,90+len(selected)*330),'white');draw=ImageDraw.Draw(plate)
    draw.text((12,8),'完整起身过程',font=platefont,fill='#15334b')
    draw.text((12,40),'保存状态回放 · 覆盖 0–8 s · 无新增物理步 · 不代表独立成功率评价',font=platefont,fill='#15334b')
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe());rows=[]
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        for row,index in enumerate(selected):
            path=OUT/f'getup-env{index}-complete.mp4';steps=[];hashes=[]
            with mediapy.VideoWriter(str(path),shape=(480,640),fps=25,qp=22) as video:
                for step in range(0,401,2):
                    state=initial_q[index] if step==0 else qpos[step-1,index]
                    velocity=initial_v[index] if step==0 else qvel[step-1,index]
                    data.qpos[:]=state;data.qvel[:]=velocity
                    mujoco.mj_forward(model,data);np.testing.assert_array_equal(data.qpos,state)
                    camera.lookat[:]=data.xpos[root[0]];renderer.update_scene(data,camera=camera)
                    pixels=Image.fromarray(renderer.render().copy());d=ImageDraw.Draw(pixels)
                    d.rectangle((0,0,640,78),fill='#11202d')
                    phase='冻结与落地' if step<100 else '策略主动控制'
                    d.text((8,4),f'保存状态回放 | 环境 {index} | t={step*.02:.2f} s | 步 {step}',font=font,fill='white')
                    d.text((8,29),phase+' · 固定官方策略，探索标准差 0.05',font=font,fill='white')
                    d.text((8,54),'无新增仿真、无插值；来自本轮连续物理状态',font=small,fill='white')
                    video.add_image(np.asarray(pixels));steps.append(step)
                    hashes.append(__import__('hashlib').sha256(state.tobytes()).hexdigest())
                    if step in snapshots:
                        col=snapshots.index(step);x=10+col*322;y=100+row*330
                        plate.paste(pixels.resize((310,233),Image.Resampling.LANCZOS),(x,y+30))
                        draw.text((x,y),f'{step*.02:g} s · {labels[col]}',font=platefont,fill='#15334b')
                        draw.text((x,y+270),f'环境 {index} · 原始控制步 {step}',font=platefont,fill='#15334b')
            rows.append(dict(environment_index=index,path=str(path.relative_to(ROOT)),sha256=sha(path),
                requested_pose=('prone','supine','left-side','right-side')[plan['split']['pose_codes'][index]],
                selection='Original recorded env0' if row==0 else 'Largest late eligible head-leg-contact count among32saved worlds',
                late_contact_steps=int(scores[index]),frames=len(steps),fps=25,common_steps=steps,
                episode_times_seconds=[s*.02 for s in steps],qpos_sha256_by_frame=hashes,
                original_source_camera=False,source='exact-saved-physical-state-replay',physics_steps=0))
            print('COMPLETE_SAVED_GETUP',index,'0..8s frames=',len(steps),'physics_steps=0',flush=True)
    image=ROOT/'docs/reports/assets/getup-complete-sequence.jpg';plate.save(image,quality=94)
    assert not torch.cuda.is_initialized()
    result=dict(status='rendered-awaiting-verification',videos=rows,image=str(image.relative_to(ROOT)),image_sha256=sha(image),
        source_record=proof['record'],source_dataset_sha256=sha(source),initial_state_sha256=sha(record/'live-startup.npz'),
        policy_sha256=sha(ROOT/'configs/environments/action_recording.json'),script_sha256=sha(Path(__file__)),
        physics_steps=0,optimizer_steps=0,cuda_initialized=False,original3second_clips_preserved=True,
        interpolation=False,scope='Exact saved-state geometry replay; no new evaluation, learning or success-rate claim')
    write(OUT/'replay.json',result);print(json.dumps({k:v for k,v in result.items() if k!='videos'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    from evaluate_standup_reference import Tee
    OUT.mkdir(exist_ok=True)
    with (OUT/'replay-execution.log').open('x',buffering=1) as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:main()
        finally:sys.stdout,sys.stderr=stdout,stderr
