"""Recount the matched outcomes and visualize saved actual-right states.

Two contrasting worlds are selected by an explicit rule: the lowest numbered
actual-right world where the candidate beats both references, and the lowest
numbered world where it loses to both. These examples are not typical-case
estimates. Rendering uses recorded qpos/qvel and mj_forward, never a physics
step; the replays are clearly distinguished from the original camera videos.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
sys.path.insert(0, str(ROOT / 'scripts/validation'))

import imageio_ffmpeg
import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mjlab.scene import Scene

import getup_shared_release as shared
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from s1_reference_inputs import read, sha

OUT = ROOT / 'records/reports/getup-monitored-pair-action-s3267'
PLAN = ROOT / 'configs/experiments/getup_monitored_pair_action_s3267.json'
ROLES = ('parent', 'control', 'candidate')


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    score = read(OUT / 'analysis.json')
    spec = read(PLAN)
    if score['status'] != 'passed-independent-verification' or score['protocol_sha256'] != sha(PLAN):
        raise ValueError('Independent trajectory verification is required')
    record = ROOT / score['record']
    outcomes, release = {}, None
    for role in ROLES:
        arm = score['arms'][role]
        episodes = arm['timing']['episodes']
        classes = [row['release_pose'] for row in episodes]
        if release is None:
            release = classes
        if release != classes:
            raise ValueError('Branches do not share release classes')
        outcomes[role] = [row['hold_start_seconds'] is not None and stable and row['release_pose'] in shared.POSES
                          for row, stable in zip(episodes, arm['end_hold']['stable_by_environment'], strict=True)]
        if sum(outcomes[role]) != arm['combined_successes']:
            raise ValueError('Independent outcome recount differs: ' + role)
        if sha(record / f'trace-{role}.npz') != arm['trace_sha256']:
            raise ValueError('Saved trajectory changed: ' + role)
    paired = {}
    for role in ROLES[:-1]:
        wins = [i for i, (ref, cand) in enumerate(zip(outcomes[role], outcomes['candidate'], strict=True)) if cand and not ref]
        losses = [i for i, (ref, cand) in enumerate(zip(outcomes[role], outcomes['candidate'], strict=True)) if ref and not cand]
        expected = score['comparisons'][role]['all']
        if wins != expected['wins'] or losses != expected['losses']:
            raise ValueError('Paired recount differs: ' + role)
        n = len(wins) + len(losses)
        p_exact = min(1., 2 * sum(math.comb(n, i) for i in range(min(len(wins), len(losses)) + 1)) / 2**n) if n else 1.
        paired[role] = dict(wins=wins, losses=losses, net=len(wins)-len(losses),
                            two_sided_exact_binomial_p=p_exact)
    candidate_wins = set(paired['parent']['wins']) & set(paired['control']['wins'])
    candidate_losses = set(paired['parent']['losses']) & set(paired['control']['losses'])
    wins = sorted(i for i in candidate_wins if release[i] == 'right-side')
    losses = sorted(i for i in candidate_losses if release[i] == 'right-side')
    if not wins or not losses:
        raise ValueError('No contrasting shared actual-right examples')
    cases = [wins[0], losses[0]]
    telemetry = read(record / 'gpu-telemetry.json')
    if telemetry['failure'] is not None or not telemetry['samples']:
        raise ValueError('GPU telemetry is incomplete')
    samples = telemetry['samples']
    summary = dict(status='recounted-from-independent-episode-data', analysis_sha256=sha(OUT / 'analysis.json'),
        protocol_sha256=sha(PLAN), record=score['record'], outcomes={r: sum(outcomes[r]) for r in ROLES},
        motion_gates={r: score['arms'][r]['all_motion_gates_passed'] for r in ROLES},
        actual_right={r: score['arms'][r]['timing']['actual']['right-side'] for r in ROLES},
        paired=paired, chosen_worlds=dict(candidate_beats_both=cases[0], candidate_loses_to_both=cases[1]),
        selection_rule='Lowest numbered actual-right world in each stated paired outcome category.',
        original_video_actual_release={f"{v['role']}:{v['requested_pose']}": v['actual_release_pose'] for v in score['videos']},
        gpu=dict(samples=len(samples), median_utilization_percent=statistics.median(s['gpu_percent'] for s in samples),
                 peak_used_mib=max(s['used_mib'] for s in samples),
                 scope='Whole evaluation telemetry, including initialization and artifact saving.'),
        additional_physics_substeps=0, additional_learning_updates=0)
    target = OUT / 'actual-right-review'
    if target.exists():
        raise FileExistsError('Preserve existing visual review')
    target.mkdir()
    write(target / 'paired-summary.json', summary)

    cfg, _, _ = shared.prepare(spec['environment'])
    model = Scene(cfg.env.scene, device='cpu').compile()
    with np.load(record / 'scene-static-fields.npz', allow_pickle=False) as fields:
        compare_arrays(model_arrays(model), {key: fields[key] for key in fields.files})
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
    root = model.body('robot/trunk_base').id
    data, camera = mujoco.MjData(model), mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .8, 135, -15
    title, label, small = (ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc', size) for size in (25, 18, 15))
    canvas = Image.new('RGB', (1300, 1800), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 6), '实际右侧起身', font=title, fill='#173950')
    draw.text((12, 40), '保存状态回放：对照两个相反结果；示例不代表总体成功率', font=label, fill='#173950')
    with np.load(record / 'initial.npz', allow_pickle=False) as a:
        initial = {key: a[key][cases].copy() for key in ('qpos', 'qvel')}
    with np.load(record / 'trace-shared-prefix.npz', allow_pickle=False) as a:
        prefix = {key: a[key][:, cases].copy() for key in ('qpos', 'qvel')}
    with np.load(record / 'release-parent.npz', allow_pickle=False) as a:
        for key in ('qpos', 'qvel'):
            np.testing.assert_array_equal(prefix[key][-1], a[key][cases])
    branches = {}
    for role in ROLES:
        with np.load(record / f'trace-{role}.npz', allow_pickle=False) as a:
            branches[role] = {key: a[key][:, cases].copy() for key in ('qpos', 'qvel')}
    def state(role, case_index):
        return {key: np.concatenate((initial[key][case_index:case_index+1], prefix[key][:, case_index],
                                     branches[role][key][:, case_index])) for key in ('qpos', 'qvel')}
    def render(renderer, pose, velocity):
        data.qpos[:], data.qvel[:] = pose, velocity
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(data.qpos, pose)
        camera.lookat[:] = data.xpos[root]
        renderer.update_scene(data, camera=camera)
        return Image.fromarray(renderer.render().copy())
    times = (2., 3.2, 8., 11.)
    labels = {'parent': '父模型', 'control': '同步组', 'candidate': '监控候选组'}
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    stills, videos = [], []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for case_index, world in enumerate(cases):
            for role_index, role in enumerate(ROLES):
                row = case_index * 3 + role_index
                y = 77 + row * 282
                outcome = '持稳通过' if outcomes[role][world] else '未持稳'
                draw.text((12, y), f'场景 {world} · {labels[role]} · {outcome}', font=label, fill='#173950')
                saved = state(role, case_index)
                if len(saved['qpos']) != 551 or len(saved['qvel']) != 551:
                    raise ValueError('Unexpected full-episode length')
                for col, seconds in enumerate(times):
                    step = round(seconds / .02)
                    pose = saved['qpos'][step]
                    image = render(renderer, pose, saved['qvel'][step])
                    x = 12 + 322 * col
                    canvas.paste(image.resize((312, 234), Image.Resampling.LANCZOS), (x, y + 24))
                    tilt = float(np.rad2deg(np.arccos(np.clip(1 - 2 * np.sum(pose[4:6].astype(float)**2), -1, 1))))
                    draw.text((x, y + 260), f'{seconds:g} s · 倾角 {tilt:.1f}°', font=small, fill='#173950')
                    stills.append(dict(world=world, role=role, seconds=seconds, tilt_deg=tilt,
                                       qpos_sha256=hashlib.sha256(pose.tobytes()).hexdigest()))
            saved = state('candidate', case_index)
            output = target / f'candidate-right-env{world}-replay.mp4'
            frame_hashes = []
            with mediapy.VideoWriter(str(output), shape=(480, 640), fps=25, qp=24) as video:
                for step in range(0, 551, 2):
                    pose = saved['qpos'][step]
                    frame = render(renderer, pose, saved['qvel'][step])
                    overlay = ImageDraw.Draw(frame)
                    overlay.rectangle((0, 0, 640, 48), fill=(17, 30, 41))
                    overlay.text((8, 3), f'保存状态回放 · 候选组 · 实际右侧 · 环境 {world}', font=small, fill='white')
                    overlay.text((8, 25), f't={step*.02:.2f} s · 无额外物理步', font=small, fill='white')
                    video.add_image(np.asarray(frame))
                    frame_hashes.append(hashlib.sha256(pose.tobytes()).hexdigest())
            decoded = imageio_ffmpeg.read_frames(str(output), pix_fmt='rgb24')
            metadata = next(decoded)
            if metadata['fps'] != 25 or metadata['size'] != (640, 480) or sum(1 for _ in decoded) != 276:
                raise ValueError('Full replay decode failed')
            videos.append(dict(world=world, path=relative(output), sha256=sha(output), frames=276, fps=25,
                               qpos_sha256_by_frame=frame_hashes))
    image = target / 'actual-right-keyframes.jpg'
    canvas.save(image, quality=93)
    write(target / 'manifest.json', dict(status='verified-saved-state-replays', analysis_sha256=sha(OUT / 'analysis.json'),
        summary_sha256=sha(target / 'paired-summary.json'), image=relative(image), image_sha256=sha(image),
        snapshots=stills, videos=videos, script_sha256=sha(Path(__file__)),
        additional_physics_substeps=0, learning_updates=0, interpolation=False,
        original_camera=False, selected_examples_are_not_population_rates=True))
    print('MATCHED_REVIEW_VERIFIED', cases, 'frames', len(stills), 'videos', len(videos), flush=True)


if __name__ == '__main__':
    main()
