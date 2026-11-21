"""Render saved poses for one left-turn scene across the target-return comparison."""

from hashlib import sha256

from mjlab.scene import Scene
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plot_forward_support_continuation import find_font
from preflight_nominal_timestep_low_friction import configured
from review_nominal_skill_composition_handoff import ROOT, read, sha
from roll_metrics import rotation
from record_io import write
from verify_nominal_skill_composition import read_trace


PROJECT = ROOT/'project.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-upright-early-return'


def main():
    goal = read(PROJECT)['goal_execution']
    earlier, current = (goal[key] for key in (
        'nominal_low_friction_upright_recovery','nominal_low_friction_upright_early_return'))
    assert all(item['status']=='reviewed' for item in (earlier,current))
    plan = read(ROOT/current['protocol'],current['protocol_sha256'])
    score = read(ROOT/current['result'],current['result_sha256'])
    left = set(plan['seven_command']['groups']['turn-left'])
    world = min(left.intersection(score['changed_and_rescued_against_both']) or {297})
    cases = ((current,'original-first','提前交还试验的原策略参照'),
             (earlier,'observed-leg-recovery','第550步交还原策略'),
             (current,'early-return-leg-recovery','第450步交还原策略'))
    targets = (450,500,550,600)
    width,height = 300,212
    canvas = Image.new('RGB',(1272,902),'#eef4f8')
    draw = ImageDraw.Draw(canvas)
    title, heading, caption = (ImageFont.truetype(str(find_font()),size) for size in (29,21,17))
    draw.text((18,12),'腿部目标交还时点',font=title,fill='#142f45')
    draw.text((18,51),f'左转分组场景{world}的保存状态回放',font=caption,fill='#415e70')
    cfg = configured(plan)
    model = Scene(cfg.scene,device='cpu').compile()
    work,camera = mujoco.MjData(model),mujoco.MjvCamera()
    frames = []
    with mujoco.Renderer(model,height=height,width=width) as viewer:
        for row,(node,role,label) in enumerate(cases):
            family = ROOT/node['collection']
            raw = read(family,node['collection_sha256'])
            trace = read_trace(family.parent/role,raw['branches'][role]['chunks'],251,600,('qpos','head_contact'),sha)
            poses = trace['qpos'][np.asarray(targets)-251,world]
            angles = rotation(poses[:,3:7])
            if row==0:
                camera.distance,camera.elevation = .64,-12.
                camera.azimuth = 90.+float(np.rad2deg(np.arctan2(angles[0,1,0],angles[0,0,0])))
            tilt = np.rad2deg(np.arccos(np.clip(angles[:,2,2],-1.,1.)))
            top = 85+268*row
            draw.text((18,top),label,font=heading,fill='#142f45')
            for col,(control,pose,angle) in enumerate(zip(targets,poses,tilt,strict=True)):
                work.qpos[:] = pose
                mujoco.mj_forward(model,work)
                np.testing.assert_array_equal(work.qpos,pose)
                assert work.time==0.
                camera.lookat[:] = work.qpos[:3]+[0.,0.,.02]
                viewer.update_scene(work,camera=camera)
                x = 18+312*col
                canvas.paste(Image.fromarray(viewer.render().copy()),(x,top+28))
                head = bool(trace['head_contact'][control-251,world])
                draw.text((x,top+242),f'第{control}步 · 倾角{angle:.1f}° · '+('头接地' if head else '头离地'),
                          font=caption,fill='#142f45')
                frames.append(dict(role=role,control=control,head_contact=head,tilt_deg=float(angle),
                    collection_sha256=node['collection_sha256'],qpos_sha256=sha256(pose.tobytes()).hexdigest()))
    output = REPORT/'return-keyframes.jpg'
    assert not output.exists()
    canvas.save(output,quality=94,subsampling=0)
    write(REPORT/'return-keyframes.json',dict(image_sha256=sha(output),world=world,frames=frames,
        selection='Smallest left turn rescue common to both references, otherwise the prior late response in scene 297.',
        new_physical_steps=0,cross_run_exact_snapshot_claimed=False))
    print('LOW_FRICTION_UPRIGHT_RETURN_KEYFRAMES',world,sha(output),flush=True)


if __name__=='__main__':
    main()
