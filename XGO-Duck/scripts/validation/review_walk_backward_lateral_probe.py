"""Compare the signed physical response and render one saved same-release example."""
from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

import collect_walk_backward_onset as onset
from collect_walk_backward_lateral_probe import OUT, PLAN, ROLES, SEEDS, SIGNS
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_common_walk_forward_phase import arrays


OFFSETS = (1,4,7,20,30)
NAMES = {'backward':'原策略','reference-repeat':'原策略复测','body-left':'左向侧移','body-right':'右向侧移'}


def data_for(row):
    record = ROOT/row['record']
    prefix = record/'trace-shared-prefix.npz'
    assert sha(prefix)==row['shared_prefix_sha256']
    baseline = arrays(prefix)
    data = {}
    for role in ROLES:
        path = record/f'trace-{role}.npz'
        assert sha(path)==row['trace_sha256'][role]
        data[role] = arrays(path)
    return record,baseline,data


def response(row,record,data):
    ids = np.asarray(row['triggered'],dtype=int)
    assert ids.size
    primary,repeat = (data[role] for role in ROLES[:2])
    primary_receipt = arrays(record/'impulse-backward.npz')
    q = primary_receipt['qpos'][ids,3:7].astype(float)
    w,x,y,z = q.T
    yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    lateral = np.stack((-np.sin(yaw),np.cos(yaw)),axis=-1)
    expected_xy = .5*(primary['xy'][:,ids].astype(float)+repeat['xy'][:,ids])
    repeat_difference = abs(((repeat['xy'][:,ids]-primary['xy'][:,ids])*lateral).sum(axis=-1))
    common_stalls = np.isin(ids,row['common_near_static'])
    if common_stalls.any():
        cohort_mask = common_stalls
    else:
        cohort_mask = np.ones(len(ids),dtype=bool)
    output = {}
    for role,sign in SIGNS.items():
        trace = data[role]
        signed_xy = ((trace['xy'][:,ids]-expected_xy)*lateral).sum(axis=-1)*sign
        signed_body_speed = (trace['velocity'][:,ids,1]-.5*(primary['velocity'][:,ids,1]+repeat['velocity'][:,ids,1]))*sign
        output[role] = {str(step*20):dict(
            median_directed_lateral_mm=float(1000*np.median(signed_xy[step-1,cohort_mask])),
            median_directed_lateral_speed_m_s=float(np.median(signed_body_speed[step-1,cohort_mask])),
            positive_lateral_delta_worlds=int((signed_xy[step-1,cohort_mask]>0).sum()),
            median_two_reference_difference_mm=float(1000*np.median(repeat_difference[step-1,cohort_mask])))
            for step in OFFSETS}
    return dict(seed=row['seed'],scope='common-stalls' if common_stalls.any() else 'all-triggered',
                worlds=ids[cohort_mask].tolist(),response=output)


def plate(spec,row,baseline,data):
    from mjlab.scene import Scene

    env = onset.environment(spec,row['seed'])
    cfg,_ = onset.config(env)
    model = Scene(cfg.scene,device='cpu').compile()
    with np.load(OUT/'cpu-scene-fields.npz') as expected:
        compare_arrays(model_arrays(model),{name:expected[name] for name in expected.files})
    ids = [i for i in range(model.nbody) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '').endswith('trunk_base')]
    assert len(ids)==1
    world = row['common_near_static'][0]
    model.vis.global_.offwidth = max(model.vis.global_.offwidth,640)
    model.vis.global_.offheight = max(model.vis.global_.offheight,480)
    data_cpu = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance,camera.azimuth,camera.elevation = 1.1,90,-20
    frozen = baseline['qpos'][-1,world]
    data_cpu.qpos[:] = frozen
    mujoco.mj_forward(model,data_cpu)
    camera.lookat[:] = data_cpu.xpos[ids[0]]
    large = ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',28)
    font = ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    board = Image.new('RGB',(1480,1605),'#edf2f5')
    ImageDraw.Draw(board).text((10,8),'后退双向侧移',font=large,fill='#193147')
    ImageDraw.Draw(board).text((10,48),f'种子 {row["seed"]} · 场景 {world} · 两路原策略共同停滞 · 保存状态回放',
                              font=font,fill='#193147')
    sampled = []
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        for i,role in enumerate(ROLES):
            branch = data[role]
            for j,offset in enumerate((0,7,50)):
                if offset==0:
                    pose = frozen
                    speed = -baseline['velocity'][-1,world,0]
                else:
                    pose = branch['qpos'][offset-1,world]
                    speed = -branch['velocity'][offset-1,world,0]
                data_cpu.qpos[:] = pose
                mujoco.mj_forward(model,data_cpu)
                np.testing.assert_array_equal(data_cpu.qpos,pose)
                renderer.update_scene(data_cpu,camera=camera)
                frame = Image.fromarray(renderer.render().copy()).resize((480,360),Image.Resampling.LANCZOS)
                x,y = 10+j*490,78+i*380
                board.paste(frame,(x,y))
                draw = ImageDraw.Draw(board)
                draw.rectangle((x,y,x+480,y+34),fill='#132431')
                draw.text((x+7,y+4),f'{NAMES[role]} · {2.+offset*.02:.2f} s · 后退速度 {speed:+.3f} m/s',
                          font=font,fill='white')
                sampled.append(dict(role=role,world=world,control=100+offset,
                   source_qpos_sha256=sha256(pose.tobytes()).hexdigest()))
    out = OUT/'matched-stall-frames.jpg'
    board.save(out,quality=92)
    return dict(path=out.relative_to(ROOT).as_posix(),sha256=sha(out),selected=sampled,
                seed=row['seed'],world=world,physical_steps=0,original_camera=False)


def main():
    combined = OUT/'analysis.json'
    proof = read(combined)
    assert proof['protocol_sha256']==sha(PLAN) and not proof['heldout_used']
    spec = read(PLAN)
    cohorts = []
    example = None
    for seed in SEEDS:
        path = OUT/f'seed{seed}-analysis.json'
        assert sha(path)==proof['sources'][path.name]
        row = read(path)
        record,baseline,data = data_for(row)
        cohorts.append(response(row,record,data))
        if example is None and row['common_near_static']:
            example = plate(spec,row,baseline,data)
    write(OUT/'response-review.json',dict(status='passed-saved-response-review',analysis_sha256=sha(combined),
          cohorts=cohorts,example=example,script_sha256=sha(Path(__file__)),physical_substeps=0,
          learning_updates=0,heldout_used=False))
    print('BACKWARD_LATERAL_RESPONSE',[(c['seed'],c['scope'],c['response']) for c in cohorts],flush=True)


if __name__=='__main__':
    main()
