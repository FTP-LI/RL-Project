"""Inspect the actual first-trigger cases without starting a new simulation."""
import argparse
from hashlib import sha256
import os
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from standup_metrics import projected_gravity


OUT = ROOT / 'records/reports/getup-knee-support/supply-recovery'
ASSET = ROOT / 'docs/reports/assets/getup-knee-screen-tilt.png'
ROLES = ('reference-a', 'reference-b', 'knee-pulse')


def collect():
    proof = read(OUT / 'analysis.json')
    assert proof['status'] == 'passed-integrity' and proof['sampling_uses_unchanged_supply']
    record = ROOT / proof['record']
    coordinator = read(OUT / 'coordinator.json')
    fork = coordinator['fork_control']
    worlds = proof['first_trigger_worlds']
    keys = ('quaternion', 'angular_velocity', 'joint_positions', 'applied_targets', 'unmodified_targets',
            'native_head_leg_contact', 'foot_contact')
    prefix_path = record / 'trace-shared-prefix.npz'
    assert sha(prefix_path) == proof['shared_prefix_sha256']
    with np.load(prefix_path) as source:
        prefix = {name: source[name][:, worlds] for name in keys}
    angles, outputs, losses = {}, {}, []
    for role in ROLES:
        path = record / f'trace-{role}.npz'
        assert sha(path) == proof['trace_sha256'][role]
        with np.load(path) as source:
            trace = {name: np.concatenate((prefix[name], source[name][:, worlds])) for name in keys}
        gravity = projected_gravity(trace['quaternion'])
        tilt = np.rad2deg(np.arccos((-gravity[..., 2]).clip(-1., 1.)))
        angles[role] = tilt
        rows = []
        for i, world in enumerate(worlds):
            phases = []
            for step in (fork, fork + 2, fork + 6, fork + 12, fork + 25, fork + 50, 400, 550):
                k = step - 1
                knees = coordinator['knee_ids']
                offset = trace['applied_targets'][k, i, knees] - trace['unmodified_targets'][k, i, knees]
                phases.append(dict(time_seconds=step * .02, tilt_deg=float(tilt[k, i]),
                    pitch_rate_rad_s=float(trace['angular_velocity'][k, i, 1]),
                    knee_position_deg=np.rad2deg(trace['joint_positions'][k, i, knees]).tolist(), knee_target_offset_rad=offset.tolist(),
                    head_leg_contact=bool(trace['native_head_leg_contact'][k, i]), both_feet=bool(trace['foot_contact'][k, i].all())))
            rows.append(dict(world=world, phases=phases, final_hold=proof['scores'][role]['final_hold_by_world'][world]))
        outputs[role] = rows
    for i, row in enumerate(proof['early_response']):
        for role in ROLES:
            np.testing.assert_allclose(angles[role][fork + 24, i], row['tilt_at_half_second_deg'][role], atol=1e-9, rtol=0)
    for world in proof['comparison']['concordant_losses']:
        candidate = proof['scores']['knee-pulse']
        losses.append(dict(world=world, actual_release=candidate['rows'][world]['release_pose'],
            candidate_trigger=candidate['trigger_by_world'][world], received_pulse=world in candidate['received_pulse']))
    return proof, fork, worlds, angles, outputs, losses


def plot(fork, worlds, angles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager
    from figure_style import apply_figure_style
    from render_roll_angle_results import checked_save

    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei', sizes=(11, 10, 9))
    matplotlib.rcParams['savefig.bbox'] = None
    indices = np.arange(fork - 10, fork + 41)
    t = (indices - fork + 1) * .02
    show = worlds[:4]
    fig, axes = plt.subplots(1, len(show), squeeze=False, sharey=True, figsize=(4.15 * len(show), 4.1))
    fig.subplots_adjust(left=.12 if len(show) == 1 else .10, right=.97, bottom=.29, top=.85, wspace=.18)
    upper = max(40, int(np.ceil(max(values[indices].max() for values in angles.values()) / 10)) * 10)
    handles = []
    for index, (world, ax) in enumerate(zip(show, axes[0], strict=True)):
        ax.axvspan(0, .24, color='#eef4f8', zorder=0)
        ax.axvline(.50, color='#78919e', linewidth=.8, linestyle=':')
        ax.axhline(15, color='#939da2', linewidth=.8, linestyle=':')
        ax.set(title=f'环境 {world}', xlim=(-.18, .82), ylim=(0, upper), xticks=(0, .24, .50, .80),
               yticks=np.linspace(0, upper, 5), xlabel='距首次触发（秒）')
        for role, label, color, style in (
            ('reference-a', '官方一', '#59636d', '-'), ('reference-b', '官方二', '#2479a8', '--'),
            ('knee-pulse', '伸膝候选', '#a76819', '-')
        ):
            line, = ax.plot(t, angles[role][indices, index], color=color, linestyle=style, label=label, linewidth=1.75)
            if index == 0:
                handles.append(line)
    axes[0, 0].set_ylabel('躯干倾角（°）')
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .025), ncols=3)
    return checked_save(fig, ASSET)


def render_trigger_frames():
    import mujoco
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from mjlab.scene import Scene
    from common_scene_contract import compare_arrays, model_arrays
    from getup_knee_support_supply_recovery import PLAN, shared

    assert os.isatty(1), 'Use the original visible VS Code task terminal'
    proof, fork, worlds, angles, _, _ = collect()
    record = ROOT / proof['record']
    spec = read(PLAN)
    assert sha(PLAN) == proof['protocol_sha256']
    cfg, _, _ = shared.prepare(spec['environment'])
    model = Scene(cfg.env.scene, device='cpu').compile()
    with np.load(record / 'scene-static-fields.npz') as saved:
        compare_arrays(model_arrays(model), {name: saved[name] for name in saved.files})
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation = .54, -15
    root = next(i for i in range(model.nbody) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or '').endswith('trunk_base'))
    world = worlds[0]
    with np.load(record / 'trace-shared-prefix.npz') as shared_trace:
        origin = {key: shared_trace[key][-1, world].copy() for key in ('qpos', 'qvel')}
    w, x, y, z = origin['qpos'][3:7]
    camera.azimuth = float(np.rad2deg(np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))) + 90)
    steps = (fork, fork + 6, fork + 25, 550)
    stages = ('共同状态', '伸膝期间', '触发后 0.50 秒', '终态')
    labels = ('官方一', '官方二', '伸膝候选')
    figure = Image.new('RGB', (1380, 994), 'white')
    draw = ImageDraw.Draw(figure)
    font_path = '/mnt/c/Windows/Fonts/msyh.ttc'
    title, font = (ImageFont.truetype(font_path, size) for size in (27, 19))
    color = '#183a51'
    draw.text((12, 6), '同状态伸膝关键帧', font=title, fill=color)
    draw.text((12, 45), f'实际环境 {world} · 最早触发中编号最小者 · 保存状态回放，零物理步', font=font, fill=color)
    rows = []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for row, (role, label) in enumerate(zip(ROLES, labels, strict=True)):
            yy = 82 + 302 * row
            draw.text((12, yy), label, font=font, fill=color)
            with np.load(record / f'trace-{role}.npz') as src:
                qpos, qvel = src['qpos'][:, world], src['qvel'][:, world]
                for col, (step, stage) in enumerate(zip(steps, stages, strict=True)):
                    i = step - fork - 1
                    pos, vel = (origin['qpos'], origin['qvel']) if i < 0 else (qpos[i], qvel[i])
                    data.qpos[:], data.qvel[:] = pos, vel
                    mujoco.mj_forward(model, data)
                    np.testing.assert_array_equal(data.qpos, pos)
                    camera.lookat[:] = data.xpos[root]
                    renderer.update_scene(data, camera=camera)
                    frame = Image.fromarray(renderer.render().copy()).resize((328, 240), Image.Resampling.LANCZOS)
                    xx = 12 + 342 * col
                    figure.paste(frame, (xx, yy + 32))
                    angle = angles[role][step - 1, 0]
                    draw.text((xx, yy + 277), f'{step*.02:.2f} s · {stage} · {angle:.1f}°', font=font, fill=color)
                    rows.append(dict(role=role, step=step, tilt_deg=float(angle), qpos_sha256=sha256(pos.tobytes()).hexdigest()))
    assert not torch.cuda.is_initialized() and data.time == 0
    assert len({row['qpos_sha256'] for row in rows if row['step'] == fork}) == 1
    target = ASSET.with_name('getup-knee-screen-trigger-frames.jpg')
    figure.save(target, quality=95)
    write(OUT / 'trigger-frames.json', dict(status='rendered-awaiting-visual-inspection', world=world,
        frames=rows, image=target.relative_to(ROOT).as_posix(), image_sha256=sha(target),
        analysis_sha256=sha(OUT / 'analysis.json'), original_camera=False, new_physics_steps=0, learning_updates=0))
    print('KNEE_TRIGGER_FRAMES_READY', world, len(rows), flush=True)


def review_motion():
    proof, fork, worlds, angles, outputs, losses = collect()
    figure = plot(fork, worlds, angles)
    write(OUT / 'motion-review.json', dict(status='rendered-awaiting-visual-inspection', source_sha256=sha(OUT / 'analysis.json'),
        traces=proof['trace_sha256'], exact_trigger_worlds=worlds, fork_control=fork, saved_steps=outputs,
        concordant_losses=losses, figure=figure,
        new_physics_steps=0, learning_updates=0, script_sha256=sha(Path(__file__))))
    print('KNEE_MOTION_REVIEW', worlds, figure, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trigger-frames', action='store_true')
    if parser.parse_args().trigger_frames:
        render_trigger_frames()
    else:
        review_motion()
