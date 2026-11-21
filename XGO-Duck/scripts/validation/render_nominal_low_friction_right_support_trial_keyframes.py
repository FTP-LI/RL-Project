"""Render matched measured states of the first triggered common miss."""

from hashlib import sha256

from mjlab.scene import Scene
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plot_forward_support_continuation import find_font
from render_nominal_supine_foot_authority import state
from review_nominal_low_friction_first_support_effort import tilt
from roll_stand_walk_sequence import ROLL
from run_nominal_low_friction_right_support_trial import KEY,LABELS,PROJECT,REPORT,ROLES,ROOT,parent,read,rel,sha,write


def main():
    node = read(PROJECT)['goal_execution'][KEY]
    run,plan,result = (read(ROOT/node[field],node[field+'_sha256']) for field in ('collection','protocol','result'))
    folder = (ROOT/node['collection']).parent
    assert result['status']=='verified-right-hip-knee-position-trial'
    start,end = (plan['right_support_trial'][name] for name in ('decision_control','end_control'))
    eligible = result['triggered_world_ids']
    terminal = {}
    for role in ROLES[:2]:
        row = run['branches'][role]
        part, = (p for p in row['chunks'] if p['first']<=end<=p['last'])
        path = folder/role/part['file']
        assert sha(path)==part['sha256']
        with np.load(path,allow_pickle=False) as f:
            terminal[role] = f['roll_phase'][end-part['first']]
    overlap = [i for i in eligible if all(values[i]!=ROLL for values in terminal.values())]
    chosen = min(overlap or eligible)
    first,original = state(folder/'shared',run['prefix'],start,chosen)
    w,x,y,z = first['qpos'][3:7]
    heading = float(np.rad2deg(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))
    cfg = parent.config(dict(plan,composition={**plan['composition'],'foot_friction':1.}))
    plant = Scene(cfg.scene,device='cpu').compile()
    data = mujoco.MjData(plant)
    camera = mujoco.MjvCamera()
    camera.azimuth,camera.elevation,camera.distance = heading+90,-16.,.60
    width,height = 308,211
    image = Image.new('RGB',(1328,922),'#edf3f7')
    draw = ImageDraw.Draw(image)
    large,section,small = (ImageFont.truetype(str(find_font()),size) for size in (34,23,17))
    draw.text((22,16),'低摩擦右髋膝起身对照',font=large,fill='#152d40')
    draw.text((22,63),f'场景{chosen} · 同一状态分支 · 保存姿态回放',font=small,fill='#415b6f')
    receipts = []
    with mujoco.Renderer(plant,height=height,width=width) as viewer:
        for row,role in enumerate(ROLES):
            top = 99+270*row
            draw.text((22,top),LABELS[role],font=section,fill='#152d40')
            for col,k in enumerate((start,150,400,end)):
                sample,origin = (first,original) if k==start else state(folder/role,run['branches'][role],k,chosen)
                q = sample['qpos']
                data.qpos[:] = q
                mujoco.mj_forward(plant,data)
                assert data.time==0.
                np.testing.assert_array_equal(data.qpos,q)
                camera.lookat[:] = q[:3]+[0.,0.,.015]
                viewer.update_scene(data,camera=camera)
                left = 22+322*col
                image.paste(Image.fromarray(viewer.render().copy()),(left,top+33))
                angle = float(tilt(q[3:7]))
                msg = '进入滚翻' if sample['roll_phase']==ROLL else '等待准入'
                tail = f' · {msg}' if k==end else ''
                draw.text((left,top+249),f'{k*.02:.2f}秒 · 倾角{angle:.1f}°{tail}',font=small,fill='#152d40')
                receipts.append(dict(role=role,control=k,source=origin,tilt_deg=angle,
                    quaternion_and_pose_sha256=sha256(q.tobytes()).hexdigest(),roll_phase=int(sample['roll_phase'])))
    path = REPORT/'keyframes.jpg'
    assert not path.exists()
    image.save(path,quality=94,subsampling=0)
    write(REPORT/'keyframe-source.json',dict(image=rel(path),image_sha256=sha(path),result_sha256=node['result_sha256'],
        scene=chosen,selection='Lowest triggered scene missing roll entry in both references; fallback lowest triggered scene.',
        new_physics_steps=0,frames=receipts))
    print('RIGHT_SUPPORT_KEYFRAMES',sha(path),chosen,len(receipts),flush=True)


if __name__=='__main__':
    main()
