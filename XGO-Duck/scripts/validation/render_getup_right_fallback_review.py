"""Render selected right-side cases from saved states without stepping physics."""

from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from mjlab.scene import Scene

from common_scene_contract import compare_arrays, model_arrays
from getup_shared_release import prepare
from record_io import write
from review_getup_right_fallback_cross_seed import OUT, ROOT, read, sha


def main():
    analysis = OUT / 'analysis.json'
    proof = read(analysis)['seeds']['1164']
    record = ROOT / proof['record']
    protocol = record / 'protocol.json'
    assert sha(protocol) == proof['protocol_sha256']
    cases = {name: min((row for row in proof['rows'] if row['cohort'] == name), key=lambda row: row['world'])
             for name in ('all-pass', 'all-fail', 'split')}
    split = cases['split']['passed']
    good, bad = (next(role for role, outcome in split.items() if outcome is desired) for desired in (True, False))
    examples = ((cases['all-pass'], 'reference-a', '共同成功'), (cases['all-fail'], 'reference-a', '共同失败'),
                (cases['split'], good, '分歧场景 · 成功'), (cases['split'], bad, '分歧场景 · 失败'))
    cfg, _, _ = prepare(read(protocol)['environment'])
    model = Scene(cfg.env.scene, device='cpu').compile()
    with np.load(record / 'scene-static-fields.npz', allow_pickle=False) as saved:
        compare_arrays(model_arrays(model), {key: saved[key] for key in saved.files})
    root = model.body('robot/trunk_base').id
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .78, 135, -15
    canvas = Image.new('RGB', (1510, 1106), 'white')
    draw = ImageDraw.Draw(canvas)
    title, label, small = (ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc', size) for size in (26, 18, 15))
    color = '#173950'
    draw.text((12, 6), '右侧起身保存状态回放', font=title, fill=color)
    draw.text((12, 42), '同权分支 · 实际右侧释放 · 各组取最小场景编号', font=label, fill=color)
    snapshots, sources = [], {}
    times = (2., 2.38, 2.6, 3., 11.)
    names = ('qpos', 'qvel', 'native_head_leg_contact')
    prefix_path = record / 'trace-shared-prefix.npz'
    with np.load(prefix_path, allow_pickle=False) as archive:
        prefix = {key: archive[key][-1] for key in names}
    sources[prefix_path.name] = sha(prefix_path)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for row, (item, role, caption) in enumerate(examples):
            world = item['world']
            source = record / f'trace-{role}.npz'
            assert sha(source) == proof['trace_sha256'][role]
            sources[source.name] = proof['trace_sha256'][role]
            with np.load(source, allow_pickle=False) as saved:
                trace = {key: saved[key][[round(s/.02)-101 for s in times[1:]], world] for key in names}
            y = 76 + 256*row
            draw.text((12, y), f'{caption} · 场景 {world} · {role}', font=label, fill=color)
            for col, seconds in enumerate(times):
                fields = {key: prefix[key][world] if col == 0 else trace[key][col-1] for key in names}
                pose = fields['qpos']
                data.qpos[:], data.qvel[:] = pose, fields['qvel']
                mujoco.mj_forward(model, data)
                np.testing.assert_array_equal(data.qpos, pose)
                camera.lookat[:] = data.xpos[root]
                renderer.update_scene(data, camera=camera)
                tile = Image.fromarray(renderer.render().copy()).resize((286, 214), Image.Resampling.LANCZOS)
                x = 12 + 300*col
                canvas.paste(tile, (x, y+22))
                tilt = float(np.rad2deg(np.arccos(np.clip(1 - 2 * np.sum(pose[4:6].astype(float)**2), -1, 1))))
                contact = bool(fields['native_head_leg_contact'])
                draw.text((x, y+238), f'{seconds:g} s · 倾角 {tilt:.1f}° · '+('头腿接触' if contact else '头腿分离'),
                          font=small, fill=color)
                snapshots.append(dict(world=world, role=role, seconds=seconds, tilt_deg=tilt, contact=contact,
                                      qpos_sha256=sha256(pose.tobytes()).hexdigest()))
    output = OUT / 'right-side-keyframes.jpg'
    assert not output.exists() and not torch.cuda.is_initialized()
    canvas.save(output, quality=93)
    write(OUT / 'media-review.json', dict(status='rendered-from-saved-states', source_sha256=sources,
        analysis_sha256=sha(analysis), script_sha256=sha(Path(__file__)), image_sha256=sha(output),
        snapshots=snapshots, original_camera=False, physical_substeps=0, learning_updates=0))
    print('RIGHT_FALLBACK_KEYFRAMES', [(item['world'], role) for item, role, _ in examples], len(snapshots), flush=True)


if __name__ == '__main__':
    main()
