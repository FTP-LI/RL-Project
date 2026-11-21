"""Show one shared failure and one success from saved right-side states."""
from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from mjlab.scene import Scene

from s1_reference_inputs import ROOT, read, sha
from common_scene_contract import compare_arrays, model_arrays
from getup_knee_support_supply_recovery import shared
from record_io import write
from review_getup_right_side_failures import DEST, OUT, PROOF


def main():
    proof, review = read(PROOF),read(DEST)
    record = ROOT/proof['record']
    groups = review['actual_right_reference_agreement']
    cases = [min(groups[key]['world_ids']) for key in ('both-pass','both-fail')]
    labels = ('两路参照均通过','两路参照均失败')
    assert not torch.cuda.is_initialized()
    for path,expected in review['source_sha256'].items():
        assert sha(ROOT/path)==expected,path
    plant = ROOT/'configs/experiments/getup_knee_support_supply_recovery.json'
    cfg,_,_ = shared.prepare(read(plant)['environment'])
    model = Scene(cfg.env.scene,device='cpu').compile()
    static = record/'scene-static-fields.npz'
    with np.load(static) as data:
        compare_arrays(model_arrays(model),{name:data[name] for name in data.files})
    root = model.body('robot/trunk_base').id
    model.vis.global_.offwidth = max(model.vis.global_.offwidth,640)
    model.vis.global_.offheight = max(model.vis.global_.offheight,480)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance,camera.azimuth,camera.elevation = .8,135,-15
    snapshots = []
    names = ('qpos','qvel','native_head_leg_contact')
    with np.load(record/'trace-shared-prefix.npz') as prefix,np.load(record/'trace-reference-a.npz') as branch:
        observed = {name:np.concatenate((prefix[name][:,cases],branch[name][:,cases])) for name in names}
    font_path = '/mnt/c/Windows/Fonts/msyh.ttc'
    title,font,small = (ImageFont.truetype(font_path,size) for size in (25,18,15))
    canvas = Image.new('RGB',(1320,646),'white')
    draw = ImageDraw.Draw(canvas)
    color = '#173950'
    draw.text((12,6),'右侧起身',font=title,fill=color)
    draw.text((12,42),'保存状态回放 · 实际右侧释放 · 各组按最小场景编号取例',font=font,fill=color)
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        for row,(world,label) in enumerate(zip(cases,labels,strict=True)):
            y = 78+row*282
            draw.text((12,y),f'{label} · 场景 {world} · 参照一轨迹',font=font,fill=color)
            score = next(item for item in review['rows'] if item['world']==world)['branches']['reference-a']
            for col,seconds in enumerate((2.,2.5,3.,11.)):
                x = 12+col*326
                index = round(seconds/.02)-1
                pose = observed['qpos'][index,row]
                data.qpos[:] = pose
                data.qvel[:] = observed['qvel'][index,row]
                mujoco.mj_forward(model,data)
                np.testing.assert_array_equal(data.qpos,pose)
                camera.lookat[:] = data.xpos[root]
                renderer.update_scene(data,camera=camera)
                frame = Image.fromarray(renderer.render().copy()).resize((312,234),Image.Resampling.LANCZOS)
                canvas.paste(frame,(x,y+26))
                tilt = score['tilt_deg'][review['time_seconds'].index(seconds)]
                contact = bool(observed['native_head_leg_contact'][index,row])
                draw.text((x,y+263),f'{seconds:g} s · 倾角 {tilt:.1f}° · '+('头腿接触' if contact else '头腿分离'),font=small,fill=color)
                snapshots.append(dict(world=world,seconds=seconds,tilt_deg=tilt,head_leg_contact=contact,
                    qpos_sha256=sha256(pose.tobytes()).hexdigest()))
    output = OUT/'right-side-keyframes.jpg'
    assert not output.exists() and not torch.cuda.is_initialized()
    canvas.save(output,quality=93)
    write(OUT/'right-side-keyframes.json',dict(status='rendered-from-saved-states',image=output.relative_to(ROOT).as_posix(),
        image_sha256=sha(output),review_sha256=sha(DEST),static_scene_sha256=sha(static),script_sha256=sha(Path(__file__)),
        snapshots=snapshots,physical_steps=0,learning_updates=0,original_source_camera=False))
    print('RIGHT_SIDE_KEYFRAMES',cases,len(snapshots),flush=True)


if __name__=='__main__':
    main()
