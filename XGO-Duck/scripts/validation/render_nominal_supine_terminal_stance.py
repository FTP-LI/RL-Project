"""Visualize one predetermined triggering scene using its saved physical poses."""

from hashlib import sha256

from mjlab.scene import Scene
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audit_nominal_supine_terminal_stance import attitude
from plot_forward_support_continuation import find_font
from run_nominal_skill_composition import config
from run_nominal_supine_terminal_stance import KEY, LABELS, PROJECT, REPORT, ROOT, read, sha, write
from verify_nominal_skill_composition import read_trace


def main():
    state = read(PROJECT)['goal_execution'][KEY]
    plan = read(ROOT / state['protocol'],state['protocol_sha256'])
    scores = read(ROOT / state['pair_analysis'],state['pair_analysis_sha256'])
    audit = read(ROOT / state['archive_audit'],state['archive_audit_sha256'])
    raw_path = ROOT / state['collection']
    raw = read(raw_path,state['collection_sha256'])
    source = raw_path.parent
    shared = plan['terminal_stance']['fork_control']
    ids = np.asarray(scores['eligible_worlds'])
    world = min(set(ids) & set(audit['target_worlds']))
    index = np.flatnonzero(ids==world).item()
    common = read_trace(source/'shared',raw['prefix']['chunks'],1,shared,('qpos',),sha)['qpos'][-1,world]
    w,x,y,z = common[3:7]
    heading = np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    plant = Scene(config(plan).scene,device='cpu').compile()
    work = mujoco.MjData(plant)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .62, float(heading+90.), -16.
    width,height = 300,225
    canvas = Image.new('RGB',(1276,933),'#eef4f8')
    draw = ImageDraw.Draw(canvas)
    title, label, row_label = (ImageFont.truetype(str(find_font()),size) for size in (28,17,22))
    draw.text((18,11),'仰卧起身末段关键帧',font=title,fill='#142f45')
    draw.text((18,48),f'场景{world} · 原物理位置回放 · 从跌倒准备开始计时',font=label,fill='#415e70')
    samples = []
    with mujoco.Renderer(plant,height=height,width=width) as viewer:
        for row,arm in enumerate(plan['terminal_stance']['roles']):
            branch = raw['branches'][arm]
            path = source/arm
            trace = read_trace(path,branch['chunks'],shared+1,branch['controls'],('qpos',),sha)['qpos'][:,world]
            telemetry = path/'terminal-stance.npz'
            assert sha(telemetry)==branch['terminal_stance_sha256']
            with np.load(telemetry,allow_pickle=False) as saved:
                contact = saved['head_foot_contact'][:,index]
            top = 81 + 280 * row
            draw.text((18,top),LABELS[arm],font=row_label,fill='#142f45')
            for col,k in enumerate((shared,450,525,600)):
                j = k-shared-1
                source_qpos = common if k==shared else trace[j]
                work.qpos[:] = source_qpos
                mujoco.mj_forward(plant,work)
                np.testing.assert_array_equal(work.qpos,source_qpos)
                camera.lookat[:] = work.qpos[:3]+[0.,0.,.015]
                viewer.update_scene(work,camera=camera)
                left = 18 + 313 * col
                canvas.paste(Image.fromarray(viewer.render().copy()),(left,top+32))
                tilt = float(attitude(source_qpos[3:7])[2])
                touch = k==shared or bool(contact[j])
                draw.text((left,top+258),f'{k*.02:.2f}秒 · 倾角{tilt:.1f}° · {"头脚接触" if touch else "头脚分离"}',
                    font=label,fill='#142f45')
                samples.append(dict(role=arm,control=k,world=int(world),qpos_sha256=sha256(source_qpos.tobytes()).hexdigest(),
                    root_tilt_deg=tilt,head_foot_contact=touch))
    target = REPORT/'witness-keyframes.jpg'
    assert not target.exists()
    canvas.save(target,quality=94,subsampling=0)
    write(REPORT/'witness-keyframes.json',dict(analysis_sha256=state['pair_analysis_sha256'],image_sha256=sha(target),
        collection_sha256=state['collection_sha256'],selection='Smallest archived forward failure in the live eligible set.',
        frames=samples,new_physical_steps=0))
    print('SUPINE_TERMINAL_STANCE_KEYFRAMES',sha(target),world,flush=True)


if __name__=='__main__':
    main()
