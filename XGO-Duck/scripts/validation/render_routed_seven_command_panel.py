"""Compose seven command rows from saved physical states without a rollout."""

from hashlib import sha256

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mjlab.scene import Scene

import run_routed_seven_command as study
from common_scene_contract import compare_arrays, model_arrays


LABELS = dict(stand='站立', forward='前进', backward='后退', **{
    'turn-left':'左转', 'turn-right':'右转', 'strafe-left':'左平移', 'strafe-right':'右平移'})


def recorded_poses(record, state, steps, worlds):
    poses, hashes = {}, {}
    with np.load(record/'initial.npz', allow_pickle=False) as data:
        poses[0] = data['qpos'][worlds]
    hashes['initial.npz'] = study.sha(record/'initial.npz')
    for chunk in state['record_chunks']:
        selected = [step for step in steps if chunk['first'] <= step <= chunk['last']]
        if not selected:
            continue
        path = record/chunk['file']
        assert study.sha(path) == chunk['sha256']
        hashes[path.name] = chunk['sha256']
        with np.load(path, allow_pickle=False) as data:
            controls, q = data['control'], data['qpos']
            for step in selected:
                row = step-chunk['first']
                assert controls[row] == step
                poses[step] = q[row, worlds]
    assert set(poses) == set(steps)
    return poses, hashes


def main():
    spec = study.read(study.PLAN)
    record = study.ROOT/spec['record']
    state = study.read(record/'evaluation.json')
    analysis_path = study.OUT/'analysis.json'
    analysis = study.read(analysis_path)
    assert analysis['protocol_sha256'] == state['protocol_sha256'] == study.sha(study.PLAN)
    assert state['status'] == 'completed' and analysis['physical_resets'] == 1
    target, receipt = study.OUT/'command-keyframes.jpg', study.OUT/'command-keyframes.json'
    groups = spec['seven_command']['groups']
    worlds = [min(set(ids)-set(analysis['score']['commands'][name]['missed_handoff'])) for name, ids in groups.items()]
    steps = (0, 100, 300, 600, 650, 800, 1600)
    times = ('初态', '跌落结束', '起身', '指令接入前', '接入后 1 秒', '接入后 4 秒', '结束')
    poses, input_hashes = recorded_poses(record, state, steps, worlds)
    cfg, _, _ = study.original.prepare(spec['environment'])
    model = Scene(cfg.env.scene, device='cpu').compile()
    study.base.exact_scene(model)
    scene_path = record/'scene-static-fields.npz'
    with np.load(scene_path, allow_pickle=False) as expected:
        compare_arrays(model_arrays(model), dict(expected))
    input_hashes[scene_path.name] = study.sha(scene_path)
    data = mujoco.MjData(model)
    root_body, = [idx for idx in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, idx) or '').endswith('trunk_base')]
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .8, 135, -15
    small, large = (ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc', size) for size in (18, 27))
    width, height, top, left, gap = 244, 183, 112, 170, 10
    plate = Image.new('RGB', (left+len(steps)*(width+gap)+10, top+len(worlds)*(height+gap)+10), '#f4f7fa')
    draw = ImageDraw.Draw(plate)
    draw.text((18, 13), '起身接七指令', font=large, fill='#19344a')
    draw.text((18, 53), '保存状态复盘 · 每条指令取已接入的最小场景编号 · 时间为仿真秒数', font=small, fill='#19344a')
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    rows = {}
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for col, (step, label) in enumerate(zip(steps, times, strict=True)):
            draw.text((left+col*(width+gap), 83), f'{step*.02:g} s  {label}', font=small, fill='#19344a')
        for row, (name, world) in enumerate(zip(groups, worlds, strict=True)):
            outcome = analysis['score']['commands'][name]
            entered = world not in outcome['missed_handoff']
            complete = world in outcome['complete_worlds']
            label = '完整通过' if complete else '未达门槛' if entered else '未接入'
            y = top+row*(height+gap)
            draw.text((18, y+16), LABELS[name], font=large, fill='#19344a')
            draw.text((18, y+58), f'场景 {world}\n{label}', font=small, fill='#19344a', spacing=7)
            checksums = []
            for col, step in enumerate(steps):
                q = poses[step][row]
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                np.testing.assert_array_equal(data.qpos, q)
                camera.lookat[:] = data.xpos[root_body]
                renderer.update_scene(data, camera=camera)
                plate.paste(Image.fromarray(renderer.render().copy()), (left+col*(width+gap), y))
                checksums.append(sha256(q.tobytes()).hexdigest())
            rows[name] = dict(world=world, entered=entered, complete=complete, qpos_sha256=checksums)
    temporary = target.with_suffix('.tmp')
    plate.save(temporary, format='JPEG', quality=92)
    temporary.replace(target)
    study.write(receipt, dict(status='verified-physical-state-panel', source_evaluation=spec['record'],
        protocol_sha256=study.sha(study.PLAN), analysis_sha256=study.sha(analysis_path),
        original_camera=False, physical_substeps=0, frame_count=len(steps)*len(worlds), control_steps=steps,
        selection='Minimum world index among reached command assignments, independent of final score',
        rows=rows, source_sha256=input_hashes, image_sha256=study.sha(target)))
    print('ROUTED_SEVEN_COMMAND_PANEL', len(steps)*len(worlds), target, flush=True)


if __name__ == '__main__':
    main()
