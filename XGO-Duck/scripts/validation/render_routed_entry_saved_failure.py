"""Reconstruct one paired failure from saved physical states without stepping."""

from hashlib import sha256

import mujoco
from mjlab.scene import Scene
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import run_routed_forward_entry_ramp as study
from standup_metrics import projected_gravity


def tilt(qpos):
    gravity = projected_gravity(qpos[..., 3:7])
    return np.rad2deg(np.arccos(np.clip(-gravity[..., 2], -1., 1.)))


def main():
    out, root, read, sha = study.OUT, study.ROOT, study.read, study.sha
    spec, analysis = read(study.PLAN), read(out / 'analysis.json')
    assert analysis['protocol_sha256'] == sha(study.PLAN) and analysis['passed_diagnostic_gate']
    record = root / spec['record']
    state = read(record / 'evaluation.json')
    assert sha(record / 'evaluation.json') == analysis['run_sha256']
    with np.load(record / 'release-poses.npz', allow_pickle=False) as saved:
        poses = saved['poses']
    with np.load(record / 'fork-input.npz', allow_pickle=False) as saved:
        q0 = saved['qpos']
    gate = spec['pair']['diagnostic_gate']
    traces, severe = {}, {}
    for role in study.ROLES:
        path = record / role / 'trace.npz'
        assert sha(path) == state['branches'][role]['trace_sha256']
        with np.load(path, allow_pickle=False) as saved:
            q, v, entered = saved['qpos'], saved['qvel'], saved['mode'][0] == 2
        bad = (tilt(q) > gate['severe_tilt_deg']) | (q[..., 2] < gate['severe_height_min_m'])
        severe[role] = np.lib.stride_tricks.sliding_window_view(bad, gate['sustained_failure_controls'], axis=0).all(axis=-1) & entered
        traces[role] = q, v
    a, b = (severe[role] for role in study.ROLES)
    selected = np.flatnonzero(a.any(axis=0) & ~b.any(axis=0) & (poses == 'supine'))
    assert len(selected) == analysis['branches']['step-command']['severe_by_release']['supine']
    first = a[:, selected].argmax(axis=0) + 1
    median = float(np.median(first))
    world = int(selected[np.argmin(abs(first - median))])
    onset = int(a[:, world].argmax() + 1)
    frames = (0, onset, onset + gate['sustained_failure_controls'] - 1, 100)
    assert 0 < frames[1] < frames[2] <= frames[3]
    cfg, _, _ = study.parent.prepare(spec['environment'])
    model = Scene(cfg.env.scene, device='cpu').compile()
    study.base.exact_scene(model)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .72, 135, -15
    trunk = model.body('robot/trunk_base').id
    image = Image.new('RGB', (1292, 595), '#f9fbfe')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc', 18)
    draw.text((14, 9), f'起步失稳对照 · 仰卧释放场景 {world} · 保存状态重建', font=font, fill='#193147')
    proof = []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for row, role in enumerate(study.ROLES):
            q, v = traces[role]
            for col, k in enumerate(frames):
                source = q0[world] if k == 0 else q[k-1, world]
                data.qpos[:] = source
                data.qvel[:] = 0. if k == 0 else v[k-1, world]
                mujoco.mj_forward(model, data)
                np.testing.assert_array_equal(data.qpos.astype(source.dtype), source)
                camera.lookat[:] = data.xpos[trunk]
                renderer.update_scene(data, camera=camera)
                tile = Image.fromarray(renderer.render().copy()).resize((304, 228), Image.Resampling.LANCZOS)
                x, y = 14 + col*319, 40 + row*275
                image.paste(tile, (x, y))
                label = ('立即指令', '半秒渐升')[row]
                angle = float(tilt(source))
                height = float(source[2])
                draw.text((x, y+229), f'{label} · 起步后 {k/50:.2f} 秒', font=font, fill='#193147')
                draw.text((x, y+251), f'倾角 {angle:.1f}° · 高度 {height*1000:.0f} mm', font=font, fill='#193147')
                proof.append(dict(role=role, forward_control=k, tilt_deg=angle, height_m=height,
                    qpos_sha256=sha256(source.tobytes()).hexdigest()))
    target = out / 'saved-failure-keyframes.jpg'
    assert not target.exists()
    image.save(target, quality=94)
    study.write(out / 'saved-failure-media.json', dict(status='reconstructed-from-verified-saved-states',
        source_analysis_sha256=sha(out / 'analysis.json'), script_sha256=sha(root / 'scripts/validation/render_routed_entry_saved_failure.py'),
        selected_world=world, release_pose='supine', selection='Nearest onset to the median among rescued supine worlds, smallest index on ties',
        control_severe_start_seconds=onset/50, camera_frames=proof, image_sha256=sha(target), physical_substeps=0))
    print('ROUTED_ENTRY_SAVED_FAILURE', world, onset / 50, len(proof), flush=True)


if __name__ == '__main__':
    main()
