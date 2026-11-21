"""Explain the paired scores from saved 20 ms states and render two matched scenes."""

from hashlib import sha256
from pathlib import Path

from mjlab.scene import Scene
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import evaluate_nominal_low_friction_support_ppo as comparison
from plot_forward_support_continuation import find_font
from roll_metrics import rotation
from verify_nominal_skill_composition import read_trace


ROOT, REPORT, PROJECT = comparison.ROOT, comparison.REPORT, comparison.PROJECT
read, write, sha = comparison.read, comparison.write, comparison.sha
ROLES, LABELS = comparison.ROLES, comparison.LABELS
FIELDS = ('qpos','quaternion','roll_phase','head_contact','foot_contact')
COUNTS = (100, 124, 150, 200, 550, 600)


def angle(quaternion):
    return np.rad2deg(np.arccos(np.clip(rotation(quaternion)[...,2,2],-1.,1.)))


def row_at(trace, control, world):
    head = bool(trace['head_contact'][control-1,world])
    feet = int(trace['foot_contact'][control-1,world].sum())
    return dict(tilt_deg=float(angle(trace['quaternion'][control-1,world])),head_touch=head,feet_touch=feet)


def render(plan, trace, witnesses, roles):
    model = Scene(comparison.measured.friction.configured(plan).scene,device='cpu').compile()
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance,camera.elevation = .66,-12.
    width,height = 250,174
    img = Image.new('RGB',(1754,763),'#edf3f7')
    draw = ImageDraw.Draw(img)
    medium,large,small = [ImageFont.truetype(str(find_font()),size) for size in (20,34,16)]
    dark,sub = '#183148','#465b6d'
    draw.text((24,14),'低摩擦仰卧起身关键帧',font=large,fill=dark)
    draw.text((24,58),'同一CUDA释放快照，保存姿态回放',font=small,fill=sub)
    controls = (100,150,600)
    x0,gap = 207,257
    index=[]
    with mujoco.Renderer(model,height=height,width=width) as renderer:
        for example,case in enumerate(witnesses):
            world=case['world']
            label='候选新增准入' if case['kind']=='gained' else '候选转为失分'
            group_x=x0+example*3*gap
            draw.text((group_x,81),f'场景{world}  {label}',font=medium,fill=dark)
            heading=trace[roles[0]]['qpos'][99,world,3:7]
            w,x,y,z=heading
            camera.azimuth=np.rad2deg(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))+90.
            for row,role in enumerate(roles):
                top=117+211*row
                if example==0:
                    draw.text((24,top+47),LABELS[role],font=medium,fill=dark)
                for column,k in enumerate(controls):
                    left=group_x+column*gap
                    pose=trace[role]['qpos'][k-1,world]
                    data.qpos[:]=pose
                    mujoco.mj_forward(model,data)
                    np.testing.assert_array_equal(data.qpos,pose)
                    camera.lookat[:]=pose[:3]+[0.,0.,.02]
                    renderer.update_scene(data,camera=camera)
                    img.paste(Image.fromarray(renderer.render().copy()),(left,top))
                    state=row_at(trace[role],k,world)
                    head='头接地' if state['head_touch'] else '头离地'
                    draw.text((left+5,top+height+5),
                         f'{k*.02:.2f}秒  {state["tilt_deg"]:.1f}°  {head}  {state["feet_touch"]}足接地',font=small,fill=dark)
                    index.append(dict(role=role,world=world,control=k,
                                      qpos_sha256=sha256(pose.tobytes()).hexdigest(),**state))
    path=REPORT/'support-credit-keyframes.jpg'
    assert not path.exists()
    img.save(path,quality=94,subsampling=0)
    return dict(path=comparison.rel(path),sha256=sha(path),frames=index,new_physical_steps=0)


def main():
    stage=read(PROJECT)['goal_execution'][comparison.KEY]
    assert stage['status']=='reviewed'
    scores=read(ROOT/stage['result'],stage['result_sha256'])
    plan=read(ROOT/stage['protocol'],stage['protocol_sha256'])
    collection=read(ROOT/stage['collection'],stage['collection_sha256'])
    folder=ROOT/plan['record']
    assert scores['collection_sha256']==stage['collection_sha256']
    path=folder/'release-poses.npz'
    assert sha(path)==collection['release_sha256']
    with np.load(path,allow_pickle=False) as saved:
        poses=saved['poses']
    supine=poses=='supine'
    assert int(supine.sum())==532 and (scores['release_fork_control'],scores['original_roll_entry_control'])==(100,600)
    shared=read_trace(folder/'shared',collection['prefix']['chunks'],1,100,FIELDS,sha)
    trace={}
    for role in ROLES:
        suffix=read_trace(folder/role,collection['branches'][role]['chunks'],101,600,FIELDS,sha)
        trace[role]={field:np.concatenate((shared[field],suffix[field])) for field in FIELDS}

    def report_section(role):
        item=trace[role]
        entered=item['roll_phase'][-1]==comparison.collector.ROLL
        assert int(entered.sum())==scores['roles'][role]['roll_entry']
        tilt=angle(item['quaternion'])
        head,feet=item['head_contact'],item['foot_contact'].all(axis=-1)
        after=(~entered)&supine
        return entered,dict(sampled_controls=[dict(control=k, actual_supine_count=int(supine.sum()),
                    median_tilt_deg=float(np.median(tilt[k-1,supine])),inside_15_deg=int((tilt[k-1,supine]<=15.).sum()),
                    dual_feet_head_clear=int((feet[k-1]&~head[k-1]&supine).sum())) for k in COUNTS],
                late_actual_supine_failures=int(after.sum()),
                sampled_tilt_above_15_all_last_25=int(((tilt[-25:]>15.).all(axis=0)&after).sum()),
                sampled_head_and_two_feet_all_last_25=int(((head[-25:]&feet[-25:]).all(axis=0)&after).sum()))

    rolled, summary = {}, {}
    for role in ROLES:
        rolled[role],summary[role]=report_section(role)
    witnesses=[]
    parent,candidate=ROLES[0],ROLES[-1]
    for mark in ('gained','lost'):
        values=scores['paired_comparison'][parent][mark+'_worlds']
        world=min(index for index in values if supine[index])
        witnesses.append(dict(kind=mark,world=world,
            assigned_command=next(name for name,ids in plan['seven_command']['groups'].items() if world in ids),
            states={role:[dict(control=k,**row_at(trace[role],k,world)) for k in COUNTS] for role in ROLES},
            entry={role:bool(rolled[role][world]) for role in ROLES}))
    result=REPORT/'support-credit-motion-review.json'
    assert not result.exists()
    media=render(plan,trace,witnesses,ROLES)
    write(result,dict(cohort='actual supine from the single common release',source_collection_sha256=stage['collection_sha256'],
        source_score_sha256=stage['result_sha256'],summary=summary,witnesses=witnesses,
        image=media,source_script_sha256=sha(Path(__file__)),new_physical_steps=0,new_learning_updates=0))
    print('SUPPORT_PPO_MOTION_REVIEW',sha(result),[(case['kind'],case['world']) for case in witnesses],flush=True)


if __name__=='__main__':
    main()
