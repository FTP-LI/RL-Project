"""Render one scored release across the three frozen evaluation branches."""

from hashlib import sha256

from mjlab.scene import Scene
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audit_nominal_supine_terminal_stance import attitude
from plot_forward_support_continuation import find_font
from run_nominal_skill_composition import config
from run_nominal_supine_upright_band_comparison import KEY, LABELS, PROJECT, REPORT, ROLES, ROOT, read, sha, write
from verify_nominal_skill_composition import read_trace


def main():
    state = read(PROJECT)['goal_execution'][KEY]
    assert state['status'] == 'reviewed'
    plan = read(ROOT/state['protocol'], state['protocol_sha256'])
    scores = read(ROOT/state['analysis'], state['analysis_sha256'])
    data = read(ROOT/state['collection'], state['collection_sha256'])
    folder = ROOT/plan['record']
    source = folder/'release-poses.npz'
    assert sha(source)==data['release_sha256']
    with np.load(source,allow_pickle=False) as saved:
        supine = set(np.flatnonzero(saved['poses']=='supine').tolist())
    changes = scores['paired_comparison']['original-reward']
    changed = supine & set(changes['lost_worlds'] + changes['gained_worlds'])
    forward = set(plan['seven_command']['groups']['forward'])
    world = min((changed & forward) or changed or supine)
    supine_indices = sorted(supine)
    start, end = scores['release_fork_control'], scores['original_roll_entry_control']
    frame_times = (start,141,550,end)
    capture = ('释放', '奖励窗结束', '起身计分', '滚翻前准入')
    common = read_trace(folder/'shared',data['prefix']['chunks'],1,start,('qpos',),sha)['qpos'][-1,world]
    w,x,y,z = common[3:7]
    heading = np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    model = Scene(config(plan).scene,device='cpu').compile()
    work = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .64,float(heading+90.),-12.
    width,height = 300,214
    image = Image.new('RGB',(1272,933),'#eef4f8')
    draw = ImageDraw.Draw(image)
    title, row_font, label = (ImageFont.truetype(str(find_font()),size) for size in (30,21,17))
    draw.text((18,12),'仰卧起身与滚翻前站稳',font=title,fill='#142f45')
    draw.text((18,55),f'实际仰卧场景{world} · 同一释放快照 · 保存的仿真状态回放',font=label,fill='#415e70')
    frames, tilted, metrics = [], {}, {}
    with mujoco.Renderer(model,height=height,width=width) as viewer:
        for row,role in enumerate(ROLES):
            branch = data['branches'][role]
            poses = read_trace(folder/role,branch['chunks'],start+1,end,('qpos',),sha)['qpos']
            controls = (124,141,550,end)
            sampled = poses[np.asarray(controls)-start-1][:,supine_indices]
            tilted[role] = attitude(sampled[...,3:7])[...,2]
            metrics[role] = [dict(control=control,median_tilt_deg=float(np.median(tilt)),
                                  within_15_degrees=int((tilt<=15.).sum()))
                             for control,tilt in zip(controls,tilted[role],strict=True)]
            top = 89+283*row
            draw.text((18,top),LABELS[role],font=row_font,fill='#142f45')
            for column,(control,name) in enumerate(zip(frame_times,capture,strict=True)):
                pose = common if control==start else poses[control-start-1,world]
                work.qpos[:] = pose
                mujoco.mj_forward(model,work)
                np.testing.assert_array_equal(work.qpos,pose)
                camera.lookat[:] = work.qpos[:3]+[0.,0.,.02]
                viewer.update_scene(work,camera=camera)
                left = 18+312*column
                image.paste(Image.fromarray(viewer.render().copy()),(left,top+29))
                tilt = float(attitude(pose[3:7])[2])
                draw.text((left,top+243),f'{control*.02:.2f}秒 · {name} · 倾角{tilt:.1f}°',font=label,fill='#142f45')
                frames.append(dict(role=role,control=control,world=world,tilt_deg=tilt,
                                   qpos_sha256=sha256(pose.tobytes()).hexdigest()))
    result = REPORT/'comparison-keyframes.jpg'
    evidence = REPORT/'comparison-keyframes.json'
    if result.exists():
        previous = read(evidence)
        assert previous['source_collection_sha256'] == state['collection_sha256']
        assert previous['image_sha256'] == sha(result)
    image.save(result,quality=94,subsampling=0)
    delta = tilted[ROLES[-1]]-tilted[ROLES[1]]
    paired = [dict(control=control,candidate_minus_original_reward_median_deg=float(np.median(row)),
                   candidate_more_upright=int((row<0).sum()))
              for control,row in zip(controls,delta,strict=True)]
    write(evidence,dict(analysis_sha256=state['analysis_sha256'],image_sha256=sha(result),
        selection='Smallest actual-supine difference from learned control in forward group, then any group, then smallest actual-supine world.',
        source_collection_sha256=state['collection_sha256'],new_physical_steps=0,frames=frames,
        all_actual_supine_worlds=len(supine_indices),tilt_samples_from_same_release=metrics,
        paired_tilt_candidate_against_original_reward=paired))
    print('UPRIGHT_BAND_COMPARISON_KEYFRAMES',world,sha(result),flush=True)


if __name__=='__main__':
    main()
