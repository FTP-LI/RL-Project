"""Render an added stall directly from its two saved trajectories."""
from hashlib import sha256
from pathlib import Path

from mjlab.scene import Scene
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from s1_reference_inputs import ROOT, read, sha
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from run_walk_backward_start_enrichment import ARMS, OUT, PLAN, environment
from collect_walk_backward_onset import config


def main():
    path = OUT/'decision.json'
    decision = read(path)
    spec = read(PLAN)
    assert decision['status']=='passed-reset-pair-integrity' and decision['protocol_sha256']==sha(PLAN)
    assert not decision['heldout_used']
    added = set(decision['near_static_in_difficult_slots']['candidate'])-set(decision['near_static_in_difficult_slots']['control'])
    world = min(added)
    target = OUT/'keyframes.jpg'
    assert not target.exists()
    cfg,_ = config(environment(spec,'control',decision['seed']))
    model = Scene(cfg.scene,device='cpu').compile()
    source_scene = OUT/'cpu-scene-fields.npz'
    assert sha(source_scene)==spec['reference_sha256'][source_scene.relative_to(ROOT).as_posix()]
    with np.load(source_scene) as saved:
        compare_arrays(model_arrays(model),{name:saved[name] for name in saved.files})
    model.vis.global_.offwidth = max(model.vis.global_.offwidth,640)
    model.vis.global_.offheight = max(model.vis.global_.offheight,480)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance,camera.azimuth,camera.elevation = .78,100,-15
    trunk = model.body('robot/trunk_base').id
    controls = [1,100,200,300]
    font = '/mnt/c/Windows/Fonts/msyh.ttc'
    title,label,detail = (ImageFont.truetype(font,size) for size in (26,19,15))
    plate = Image.new('RGB',(1320,673),'white')
    draw = ImageDraw.Draw(plate)
    ink = '#193147'
    draw.text((12,7),'后退初态对照',font=title,fill=ink)
    draw.text((12,42),f'保存状态回放 · 场景 {world}',font=label,fill=ink)
    frames = []
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        for row,(arm,name) in enumerate(zip(ARMS,('普通重置','困难初态重置'),strict=True)):
            receipt = OUT/arm/f'seed{decision["seed"]}-analysis.json'
            assert sha(receipt)==decision['score_sha256'][arm]
            proof = read(receipt)
            trace = ROOT/proof['record']/'trace-backward.npz'
            assert sha(trace)==proof['source_sha256']
            with np.load(trace) as archive:
                k = np.asarray(controls)-1
                assert archive['active'][k,world].all()
                poses,velocity,contact = (archive[key][k,world] for key in ('qpos','velocity','contact'))
            y = 78+row*296
            draw.text((12,y),name,font=label,fill=ink)
            for col,(control,q,vel,feet) in enumerate(zip(controls,poses,velocity,contact,strict=True)):
                data.qpos[:] = q
                mujoco.mj_kinematics(model,data)
                np.testing.assert_array_equal(data.qpos,q)
                assert data.time==0.
                camera.lookat[:] = data.xpos[trunk]
                renderer.update_scene(data,camera=camera)
                x = 12+col*326
                frame = Image.fromarray(renderer.render().copy()).resize((312,234),Image.Resampling.LANCZOS)
                plate.paste(frame,(x,y+25))
                draw.text((x,y+263),f'{control*.02:.2f} s · 脚触地 {int(feet.sum())}/2',font=detail,fill=ink)
                draw.text((x,y+280),f'后退速度 {-vel[0]:+.3f} m/s',font=detail,fill=ink)
                frames.append(dict(arm=arm,control=control,backward_speed_m_s=float(-vel[0]),contact=feet.tolist(),
                                   qpos_sha256=sha256(q.tobytes()).hexdigest()))
    plate.save(target,quality=94)
    write(target.with_suffix('.json'),dict(status='rendered-saved-state-keyframes',image_sha256=sha(target),
          decision_sha256=sha(path),script_sha256=sha(Path(__file__)),world=world,frames=frames,
          original_camera=False,physical_substeps=0,learning_updates=0))
    print('BACKWARD_ENRICHMENT_KEYFRAMES',world,len(frames),flush=True)


if __name__=='__main__':
    main()
