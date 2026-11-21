"""Replay the preselected actual-right release from verified sampled states."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import imageio_ffmpeg
import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mjlab.scene import Scene

import run_getup_right_robustness_development as study
from common_scene_contract import compare_arrays, model_arrays


def world_state(path, world, keys):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key][:, world].copy() for key in keys}


def render(plan, out, *, reserved_holdout=False):
    read, sha, write = study.read, study.sha, study.write
    score_path = out / 'analysis.json'
    release = read(out / 'release-verification.json')
    assert release['status'] == 'passed-independent-release-match' and release['analysis_sha256'] == sha(score_path)
    score, spec = read(score_path), read(plan)
    assert score['protocol_sha256'] == sha(plan)
    assert score['heldout_used'] == release['heldout_used'] == reserved_holdout
    if reserved_holdout:
        decision = read(out / 'decision.json')
        assert decision['analysis_sha256'] == sha(score_path)
        assert decision['data_scope'] == 'reserved-holdout' and not decision['holdout_eligible_for_tuning']
    record, world = study.ROOT / score['record'], release['selected_world']
    media = out / 'actual-right-replay'
    assert not media.exists()
    cfg, _, _ = study.base.shared.prepare(spec['environment'])
    model = Scene(cfg.env.scene, device='cpu').compile()
    with np.load(record / 'scene-static-fields.npz', allow_pickle=False) as arrays:
        compare_arrays(model_arrays(model), {key: arrays[key] for key in arrays.files})
    root = model.body('robot/trunk_base').id
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
    data, camera = mujoco.MjData(model), mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .8, 135, -15
    keys = ('qpos', 'qvel')
    initial_path, prefix_path = record / 'initial.npz', record / 'trace-shared-prefix.npz'
    with np.load(initial_path, allow_pickle=False) as arrays:
        initial = {key: arrays[key][world:world+1] for key in keys}
        requested = str(arrays['requested_pose'][world])
    prefix = world_state(prefix_path, world, keys)
    with np.load(record / 'release-reference-a.npz', allow_pickle=False) as arrays:
        for key in keys:
            np.testing.assert_array_equal(prefix[key][-1], arrays[key][world])
    steps = np.arange(0, spec['environment']['maximum_control_steps'] + 1, 2)
    selected_steps = (100, 160, 400, 550)
    font = '/mnt/c/Windows/Fonts/msyh.ttc'
    title, label = (ImageFont.truetype(font, size) for size in (26, 18))
    canvas = Image.new('RGB', (1304, 930), 'white')
    draw = ImageDraw.Draw(canvas)
    ink = '#173950'
    draw.text((12, 7), '右侧起身', font=title, fill=ink)
    draw.text((12, 43), f'场景 {world} · 实际右侧释放 · 保存状态回放', font=label, fill=ink)
    media.mkdir()
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    items = []
    titles = ('参照 A', '参照 B', '右侧专用策略')
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for row, (role, label_text) in enumerate(zip(spec['branches'], titles, strict=True)):
            arm = score['arms'][role]
            assert arm['timely']['episodes'][world]['release_pose'] == 'right-side'
            path = record / f'trace-{role}.npz'
            assert sha(path) == arm['trace_sha256']
            branch = world_state(path, world, keys)
            states = {key: np.concatenate((initial[key], prefix[key], branch[key])) for key in keys}
            assert len(states['qpos']) == len(states['qvel']) == steps[-1] + 1
            output = media / f'{role}.mp4'
            frames, stills = [], []
            y = 77 + 284 * row
            draw.text((12, y), label_text, font=label, fill=ink)
            with mediapy.VideoWriter(str(output), shape=(480, 640), fps=25, qp=24) as video:
                for step in steps:
                    pose = states['qpos'][step]
                    data.qpos[:], data.qvel[:] = pose, states['qvel'][step]
                    mujoco.mj_forward(model, data)
                    np.testing.assert_array_equal(data.qpos, pose)
                    camera.lookat[:] = data.xpos[root]
                    renderer.update_scene(data, camera=camera)
                    frame = Image.fromarray(renderer.render().copy())
                    stamp = step * spec['environment']['criteria']['step_dt']
                    digest = sha256(pose.tobytes()).hexdigest()
                    frames.append(dict(step=int(step), seconds=stamp, qpos_sha256=digest))
                    if step in selected_steps:
                        x = 12 + 322 * selected_steps.index(step)
                        canvas.paste(frame.resize((312, 234), Image.Resampling.LANCZOS), (x, y + 24))
                        tilt = np.rad2deg(np.arccos(np.clip(1 - 2 * np.sum(pose[4:6].astype(float)**2), -1, 1)))
                        draw.text((x, y + 259), f'{stamp:g} s · 倾角 {tilt:.1f}°', font=label, fill=ink)
                        stills.append(dict(step=int(step), tilt_deg=float(tilt), qpos_sha256=digest))
                    overlay = ImageDraw.Draw(frame)
                    overlay.rectangle((0, 0, 640, 46), fill=(17, 30, 41))
                    overlay.text((8, 6), f'Saved simulation states | env {world} | t={stamp:.2f}s', fill='white')
                    overlay.text((8, 26), f'{role} | actual right side at t=2s', fill='white')
                    video.add_image(np.asarray(frame))
            decoded = imageio_ffmpeg.read_frames(str(output), pix_fmt='rgb24')
            metadata = next(decoded)
            assert metadata['fps'] == 25 and metadata['size'] == (640, 480)
            assert sum(1 for _ in decoded) == len(frames) == 276
            items.append(dict(role=role, path=study.base.rel(output), sha256=sha(output),
                trace_sha256=arm['trace_sha256'], frames=frames, stills=stills))
    image = media / 'right-side-keyframes.jpg'
    canvas.save(image, quality=94)
    write(media / 'manifest.json', dict(status='verified-actual-right-saved-state-replays', world=world,
        requested_pose=requested, release_pose='right-side', source='actual sampled GPU states', physics_substeps=0,
        heldout_used=reserved_holdout, holdout_eligible_for_tuning=False,
        initial_sha256=sha(initial_path), prefix_sha256=sha(prefix_path), script_sha256=sha(Path(__file__)),
        analysis_sha256=sha(score_path), image_sha256=sha(image),
        camera=dict(distance=.8, azimuth=135, elevation=-15), clips=items))
    print('RIGHT_ROBUSTNESS_MEDIA', world, len(items), len(steps), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--reserved-holdout', action='store_true')
    if parser.parse_args().reserved_holdout:
        import holdout_getup_right_robustness as holdout
        render(holdout.PLAN, holdout.OUT, reserved_holdout=True)
    else:
        render(study.PLAN, study.OUT)
