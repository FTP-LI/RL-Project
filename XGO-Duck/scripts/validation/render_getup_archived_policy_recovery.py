"""Show one shared reference failure recovered by both archived policies."""

from hashlib import sha256
from pathlib import Path

import mujoco
from mjlab.scene import Scene
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import run_getup_archived_policy_qualified as trial


def main():
    result = trial.OUT / 'analysis.json'
    proof = trial.read(result)
    spec = trial.read(trial.PLAN)
    assert proof['protocol_sha256'] == trial.sha(trial.PLAN) and not proof['heldout_used']
    record = trial.ROOT / proof['record']
    roles = spec['branches']
    recoveries = [set(proof['comparisons'][role][ref]['by_pose']['supine']['wins'])
                  for role in roles[2:] for ref in roles[:2]]
    common = sorted(set.intersection(*recoveries))
    world = common[0]
    assert all(proof['arms'][role]['timely']['episodes'][world]['release_pose'] == 'supine' for role in roles)
    prefix = record / 'trace-shared-prefix.npz'
    state = trial.read(record / 'evaluation.json')
    assert state['actor_weights_unchanged'] and state['protocol_sha256'] == proof['protocol_sha256']
    cfg, _, _ = trial.base.shared.prepare(spec['environment'])
    model = Scene(cfg.env.scene, device='cpu').compile()
    trial.base.exact_scene(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
    data = mujoco.MjData(model)
    root = model.body('robot/trunk_base').id
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .8, 135, -15
    font_path = '/mnt/c/Windows/Fonts/msyh.ttc'
    title, label, small = (ImageFont.truetype(font_path, size) for size in (25, 19, 16))
    canvas = Image.new('RGB', (1024, 1295), 'white')
    draw = ImageDraw.Draw(canvas)
    ink = '#15334b'
    draw.text((14, 8), '同场景仰卧起身', font=title, fill=ink)
    draw.text((14, 46), f'保存状态回放 · 场景 {world} · 两路参照失败、两份候选通过的最小编号', font=label, fill=ink)
    field_names = ('qpos', 'qvel', 'quaternion', 'foot_contact', 'native_head_leg_contact')
    names = ('参照 A', '参照 B', '原奖励续训', '侧卧到仰卧回放')
    times = (2., 3.2, 11.)
    samples = []
    with np.load(prefix, allow_pickle=False) as arrays:
        shared = {name: arrays[name][-1, world] for name in field_names}
    with np.load(record / 'release-reference-a.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(shared['qpos'], saved['qpos'][world])
        np.testing.assert_array_equal(shared['qvel'], saved['qvel'][world])
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for row, (role, name) in enumerate(zip(roles, names, strict=True)):
            branch = record / f'trace-{role}.npz'
            assert trial.sha(branch) == proof['arms'][role]['trace_sha256']
            y = 81 + 300 * row
            draw.text((14, y), name + (' · 完整持稳' if row >= 2 else ' · 未持稳'), font=label, fill=ink)
            with np.load(branch, allow_pickle=False) as arrays:
                steps = np.rint(np.asarray(times[1:]) / .02).astype(int)
                indices = steps - spec['prefix_controls'] - 1
                later = {key: arrays[key][indices, world] for key in field_names}
            for col, seconds in enumerate(times):
                x = 14 + 335 * col
                frame = shared if col == 0 else {key: later[key][col - 1] for key in field_names}
                data.qpos[:] = frame['qpos']
                data.qvel[:] = frame['qvel']
                mujoco.mj_forward(model, data)
                np.testing.assert_array_equal(data.qpos, frame['qpos'])
                camera.lookat[:] = data.xpos[root]
                renderer.update_scene(data, camera=camera)
                pixels = Image.fromarray(renderer.render().copy()).resize((324, 243), Image.Resampling.LANCZOS)
                canvas.paste(pixels, (x, y + 25))
                quat = frame['quaternion']
                tilt = float(np.rad2deg(np.arccos(np.clip(1 - 2 * (quat[1]**2 + quat[2]**2), -1, 1))))
                feet, head = int(frame['foot_contact'].sum()), bool(frame['native_head_leg_contact'])
                draw.text((x, y + 272), f'{seconds:.2f} s · 倾角 {tilt:.1f}° · 双脚触地 {feet}/2', font=small, fill=ink)
                samples.append(dict(role=role, world=world, seconds=seconds, tilt_deg=tilt, foot_contacts=feet,
                    head_leg_contact=head, qpos_sha256=sha256(frame['qpos'].tobytes()).hexdigest()))
    target = trial.OUT / 'paired-recovery-keyframes.jpg'
    assert not target.exists()
    canvas.save(target, quality=93)
    trial.write(target.with_suffix('.json'), dict(status='rendered-from-saved-states', world=world,
        joint_improvement_worlds=common, image_sha256=trial.sha(target), scoring_sha256=trial.sha(result),
        prefix_sha256=trial.sha(prefix), script_sha256=trial.sha(Path(__file__)), samples=samples,
        physical_steps=0, learning_updates=0, original_camera=False))
    print('ARCHIVED_POLICY_RECOVERY_KEYFRAMES', world, len(common), len(samples), flush=True)


if __name__ == '__main__':
    main()
