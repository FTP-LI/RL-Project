"""Render one archived shared stall for the original and two roll branches."""
from hashlib import sha256
from pathlib import Path

from mjlab.scene import Scene
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_walk_backward_natural_clearance import onset
from run_walk_backward_roll_pair import OUT, PLAN
from s1_reference_inputs import ROOT, read, sha


def main():
    proof_path = OUT/'seed1103-analysis.json'
    proof = read(proof_path)
    decision = read(OUT/'seed1103-decision.json')
    assert proof['status']=='passed-integrity' and decision['analysis_sha256']==sha(proof_path)
    assert proof['protocol_sha256']==sha(PLAN) and not proof['heldout_used']
    assert proof['common_near_static']
    world = min(proof['common_near_static'])
    output = OUT/'seed1103-keyframes.jpg'
    assert not output.exists()
    spec = read(PLAN)
    record = ROOT/proof['record']
    p = onset.environment(spec,1103)
    cfg,_ = onset.config(p)
    model = Scene(cfg.scene,device='cpu').compile()
    scene = OUT/'cpu-scene-fields.npz'
    assert sha(scene)==spec['reference_sha256'][scene.relative_to(ROOT).as_posix()]
    with np.load(scene) as saved:
        compare_arrays(model_arrays(model),{name:saved[name] for name in saved.files})
    root = model.body('robot/trunk_base').id
    prefix = record/'trace-shared-prefix.npz'
    assert sha(prefix)==proof['shared_prefix_sha256']
    keys = ('qpos','contact','velocity','active')
    with np.load(prefix) as data:
        before = {name:data[name][-1,world] for name in keys}
    controls = [100,107,115,130]
    labels = ('共同状态','修正升幅结束','修正保持结束','单脚支撑观察截止')
    roles = {'backward':'原策略','body-left':'向左支撑目标','body-right':'向右支撑目标'}
    model.vis.global_.offwidth = max(model.vis.global_.offwidth,640)
    model.vis.global_.offheight = max(model.vis.global_.offheight,480)
    camera = mujoco.MjvCamera()
    camera.distance,camera.azimuth,camera.elevation = .78,100,-15
    state = mujoco.MjData(model)
    font_path = '/mnt/c/Windows/Fonts/msyh.ttc'
    title,label,caption = (ImageFont.truetype(font_path,size) for size in (26,19,15))
    plate = Image.new('RGB',(1320,973),'white')
    draw = ImageDraw.Draw(plate)
    color = '#193147'
    draw.text((12,7),'后退侧倾对照',font=title,fill=color)
    draw.text((12,42),f'保存状态回放 · 共同停滞中的最小场景编号 {world}',font=label,fill=color)
    rows = []
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        for row,(role,name) in enumerate(roles.items()):
            path = record/f'trace-{role}.npz'
            assert sha(path)==proof['trace_sha256'][role]
            with np.load(path) as source:
                after = {field:source[field][np.asarray(controls[1:])-101,world] for field in keys}
            inputs = {field:np.concatenate((np.asarray(before[field])[None],after[field])) for field in keys}
            y = 78+row*296
            draw.text((12,y),name,font=label,fill=color)
            assert inputs['active'].all()
            for col,control in enumerate(controls):
                pose = inputs['qpos'][col]
                state.qpos[:] = pose
                mujoco.mj_kinematics(model,state)
                np.testing.assert_array_equal(state.qpos,pose)
                assert state.time==0.
                camera.lookat[:] = state.xpos[root]
                renderer.update_scene(state,camera=camera)
                frame = Image.fromarray(renderer.render().copy()).resize((312,234),Image.Resampling.LANCZOS)
                x = 12+col*326
                plate.paste(frame,(x,y+25))
                contacts = inputs['contact'][col].tolist()
                speed = float(-inputs['velocity'][col,0])
                draw.text((x,y+263),f'{control*.02:.2f} s · {labels[col]}',font=caption,fill=color)
                draw.text((x,y+280),f'脚触地 {sum(contacts)}/2 · 后退速度 {speed:+.3f} m/s',font=caption,fill=color)
                rows.append(dict(role=role,control=control,contact=contacts,backward_speed_m_s=speed,
                                 qpos_sha256=sha256(pose.tobytes()).hexdigest()))
    plate.save(output,quality=94)
    write(output.with_suffix('.json'),dict(status='rendered-saved-state-keyframes',image_sha256=sha(output),
        analysis_sha256=sha(proof_path),script_sha256=sha(Path(__file__)),world=world,frames=rows,
        original_camera=False,physical_substeps=0,learning_updates=0))
    print('BACKWARD_ROLL_KEYFRAMES',world,len(rows),flush=True)


if __name__=='__main__':
    main()
