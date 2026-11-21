"""Render the two shared supine failures from their verified saved states."""
from hashlib import sha256

import mujoco
from mjlab.scene import Scene
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common_scene_contract import compare_arrays, model_arrays
from getup_shared_release import prepare
from record_io import write
from review_getup_requested_supine_failure import OUT as REVIEW
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity


def main():
    destination = REVIEW.with_name('requested-supine-keyframes.jpg')
    receipt = destination.with_suffix('.json')
    if destination.exists():
        assert sha(destination) == read(receipt)['image_sha256']
    review = read(REVIEW)
    assert review['status'] == 'passed-saved-trace-review' and not review['heldout_used']
    proof_path = REVIEW.with_name('analysis.json')
    assert sha(proof_path) == review['evaluation_sha256']
    proof = read(proof_path)
    record = ROOT/proof['record']
    protocol = record/'protocol.json'
    assert sha(protocol) == proof['protocol_sha256']
    worlds = review['all_three_failed']
    assert len(worlds) == 2
    cfg,_,_ = prepare(read(protocol)['environment'])
    model = Scene(cfg.env.scene,device='cpu').compile()
    scene = record/'scene-static-fields.npz'
    with np.load(scene) as frozen:
        compare_arrays(model_arrays(model),{name:frozen[name] for name in frozen.files})
    model.vis.global_.offwidth = max(model.vis.global_.offwidth,640)
    model.vis.global_.offheight = max(model.vis.global_.offheight,480)
    data = mujoco.MjData(model)
    root = model.body('robot/trunk_base').id
    camera = mujoco.MjvCamera()
    camera.distance,camera.azimuth,camera.elevation = .8,135,-15
    prefix_path, branch_path = record/'trace-shared-prefix.npz', record/'trace-control.npz'
    assert sha(prefix_path) == proof['prefix_sha256'] and sha(branch_path) == proof['arms']['control']['trace_sha256']
    names = ('qpos','qvel','quaternion','foot_contact','native_head_leg_contact')
    with np.load(prefix_path) as prefix, np.load(branch_path) as branch:
        time = np.concatenate((prefix['time'],branch['time']))
        trace = {name:np.concatenate((prefix[name][:,worlds],branch[name][:,worlds])) for name in names}
    tilt = np.rad2deg(np.arccos(np.clip(-projected_gravity(trace['quaternion'])[...,2],-1.,1.)))
    font_path = '/mnt/c/Windows/Fonts/msyh.ttc'
    heading,font,small = (ImageFont.truetype(font_path,size) for size in (25,19,16))
    canvas = Image.new('RGB',(1320,710),'white')
    draw = ImageDraw.Draw(canvas)
    color = '#173950'
    draw.text((12,7),'仰卧起身失败',font=heading,fill=color)
    draw.text((12,43),'对照组的保存状态回放 · 两个指定仰卧场景三组均失败',font=font,fill=color)
    samples = []
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        for row,world in enumerate(worlds):
            y = 78+row*314
            draw.text((12,y),f'场景 {world}',font=font,fill=color)
            early = np.flatnonzero((time>2.3)&(time<=3.))
            best = early[np.argmin(tilt[early,row])]
            indices = [99,best,149,549]
            labels = ('释放','初段最接近直立','重新倾斜','观察终态')
            assert indices == sorted(set(indices))
            for col,(index,label) in enumerate(zip(indices,labels,strict=True)):
                x = 12+326*col
                pose = trace['qpos'][index,row]
                data.qpos[:] = pose
                data.qvel[:] = trace['qvel'][index,row]
                # Forward kinematics positions the saved state for rendering.
                # No integrator step is taken.
                mujoco.mj_forward(model,data)
                np.testing.assert_array_equal(data.qpos,pose)
                camera.lookat[:] = data.xpos[root]
                renderer.update_scene(data,camera=camera)
                frame = Image.fromarray(renderer.render().copy()).resize((312,234),Image.Resampling.LANCZOS)
                canvas.paste(frame,(x,y+25))
                angle = float(tilt[index,row])
                head = bool(trace['native_head_leg_contact'][index,row])
                feet = int(trace['foot_contact'][index,row].sum())
                draw.text((x,y+262),f'{time[index]:.2f} s · {label} · {angle:.1f}°',font=small,fill=color)
                draw.text((x,y+283),f'脚触地 {feet}/2 · '+('头腿接触' if head else '头腿分离'),font=small,fill=color)
                samples.append(dict(world=world,control=int(index+1),seconds=float(time[index]),tilt_deg=angle,
                    foot_contacts=feet,head_leg_contact=head,qpos_sha256=sha256(pose.tobytes()).hexdigest()))
    canvas.save(destination,quality=94)
    write(receipt,dict(status='rendered-from-saved-states',image_sha256=sha(destination),
        review_sha256=sha(REVIEW),scene_sha256=sha(scene),trace_sha256=sha(branch_path),samples=samples,
        physical_steps=0,learning_updates=0,original_source_camera=False))
    print('REQUESTED_SUPINE_KEYFRAMES',worlds,len(samples),flush=True)


if __name__ == '__main__':
    main()
