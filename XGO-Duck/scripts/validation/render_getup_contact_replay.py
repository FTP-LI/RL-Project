"""Render exact saved qpos as a labelled diagnostic plate, without dynamics.

Run in the shared visible VS Code terminal because this uses the existing EGL
renderer. Every pose is a recorded post-step state, with its original root x/y
and all joint coordinates; mj_forward updates geometry but never integrates.
This is explicitly state replay, not original-camera footage or a new episode.
"""
from pathlib import Path
import json
import os
import numpy as np
import mujoco
import torch
from PIL import Image,ImageDraw,ImageFont
from mjlab.scene import Scene
import recover_getup_contact_release as repair
from verify_getup_contact_diagnosis import ROOT,OUT,read,sha,load
from standup_metrics import projected_gravity

def main():
    assert os.isatty(1),'Use the existing visible VS Code task terminal'
    proof=read(OUT/'independent-contact-check.json');assert proof['status']=='passed'
    result=read(OUT/'analysis.json');index=15;times=(3.,6.,10.)
    assert index in result['arms']['full']['failed_environment_indices']
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',22)
    small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    title=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',28)
    sheet=Image.new('RGB',(1950,1220),'white');draw=ImageDraw.Draw(sheet)
    draw.text((15,8),'起身碰撞对照',font=title,fill='#15334b')
    draw.text((15,48),'保存状态重放，不积分；同一环境 15。仅展示配对个案，成功率按全部 32 次计算。',font=font,fill='#15334b')
    rows=[]
    for row,arm in enumerate(repair.base.ARMS):
        p=repair.base.checked(arm);folder=ROOT/result['arms'][arm]['record'];extra=load(folder/'physical-contact-trace.npz');t=load(folder/'supine-trace.npz')
        model=Scene(repair.config(p).scene,device='cpu').compile();data=mujoco.MjData(model)
        model.vis.global_.offwidth=max(model.vis.global_.offwidth,640);model.vis.global_.offheight=max(model.vis.global_.offheight,480)
        label='完整碰撞：未持稳' if arm=='full' else '原起身掩码：持稳（仅诊断）'
        y=100+row*555;draw.text((15,y),label,font=font,fill='#15334b')
        tilt=np.rad2deg(np.arccos(np.clip(-projected_gravity(t['quaternion'])[...,2],-1,1)))
        neck=read(folder/'evaluation.json')['joint_names'].index('neck_pitch')
        with mujoco.Renderer(model,height=480,width=640) as renderer:
            for col,time in enumerate(times):
                k=round(time/.02)-1;assert abs(t['time'][k]-time)<1e-7
                data.qpos[:]=extra['qpos'][k,index];data.qvel[:]=extra['qvel'][k,index]
                saved=data.qpos.copy();mujoco.mj_forward(model,data);np.testing.assert_array_equal(data.qpos,saved)
                q=t['quaternion'][k,index];w,x,yq,z=q
                yaw=np.arctan2(2*(w*z+x*yq),1-2*(yq*yq+z*z))
                camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:]=saved[:3]+[0,0,.015];camera.distance=.62;camera.azimuth=np.rad2deg(yaw)+90;camera.elevation=-18
                renderer.update_scene(data,camera=camera)
                im=Image.fromarray(renderer.render().copy());sheet.paste(im,(10+650*col,y+35))
                text=f't={time:g} s · 躯干 {tilt[k,index]:.1f}° · 颈部 {np.rad2deg(t["joint_positions"][k,index,neck]):.1f}°'
                draw.text((12+650*col,y+515),text,font=small,fill='#15334b')
                rows.append(dict(arm=arm,environment_index=index,time_seconds=time,trace_index=k,
                    qpos_sha256=__import__('hashlib').sha256(saved.tobytes()).hexdigest(),source=str((folder/'physical-contact-trace.npz').relative_to(ROOT))))
    target=ROOT/'docs/reports/assets/getup-contact-replay.jpg';sheet.save(target,quality=94)
    assert not torch.cuda.is_initialized()
    manifest=dict(status='rendered-awaiting-visual-review',image=str(target.relative_to(ROOT)),image_sha256=sha(target),frames=rows,
        original_source_camera=False,exact_saved_qpos=True,root_xy_retained=True,physics_steps=0,learning_updates=0,
        renderer='MuJoCo EGL geometry replay in visible task; no ManagerBasedRlEnv or new episode',
        source_check_sha256=sha(OUT/'independent-contact-check.json'),script_sha256=sha(Path(__file__)))
    (OUT/'replay-figure.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='frames'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    from datetime import datetime,timezone
    import sys,traceback
    from evaluate_standup_reference import Tee
    log=OUT/f'replay-visible-{datetime.now().strftime("%Y%m%dT%H%M%S")}.log'
    event=dict(argv=sys.argv,started_at=datetime.now(timezone.utc).isoformat(),foreground_tty=os.isatty(1),physics_steps=0)
    with log.open('x',buffering=1) as file:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(stdout,file),Tee(stderr,file)
        try:
            print('VISIBLE_SAVED_STATE_REPLAY',json.dumps(event),flush=True);main();event['exit_code']=0
        except BaseException as e:event.update(exit_code=1,error=repr(e));traceback.print_exc();raise
        finally:
            event['ended_at']=datetime.now(timezone.utc).isoformat();event['log']=str(log.relative_to(ROOT))
            with (OUT/'replay-commands.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            sys.stdout,sys.stderr=stdout,stderr
