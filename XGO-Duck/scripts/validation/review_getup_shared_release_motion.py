"""Explain orientation differences using the two verified saved branches."""
from pathlib import Path
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from record_io import write
from recovery_end_hold import evaluate_end_hold
from review_getup_release_prefix import state_distance
from s1_reference_inputs import read, sha
from verify_getup_shared_release import load_arrays

OUT = ROOT / 'records/reports/getup-shared-release/fixed-terrain'
ASSETS = ROOT / 'docs/reports/assets'
ROLES = ('repeat-a', 'repeat-b')


def orientation_features(q):
    q = q.astype(np.float64)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    forward_x = 1 - 2 * (y*y + z*z)
    forward_y = 2 * (x*y + w*z)
    heading = np.arctan2(forward_y, forward_x)
    valid = np.hypot(forward_x, forward_y) > .1
    tilt = np.rad2deg(np.arccos(np.clip(1 - 2 * (x*x + y*y), -1, 1)))
    return heading, valid, tilt


def measure():
    proof = read(OUT / 'analysis.json')
    assert proof['status'] == 'passed-integrity'
    record = ROOT / proof['record']
    paths = [record / f'trace-{role}.npz' for role in ROLES]
    a, b = (load_arrays(path) for path in paths)
    np.testing.assert_array_equal(a['time'], b['time'])
    _, angle, joints = state_distance(a['qpos'], b['qpos'])
    h0, v0, t0 = orientation_features(a['qpos'][..., 3:7])
    h1, v1, t1 = orientation_features(b['qpos'][..., 3:7])
    heading = abs(np.rad2deg((h0 - h1 + np.pi) % (2*np.pi) - np.pi))
    tilt = abs(t0 - t1)
    assert (v0[-1] & v1[-1]).all()
    valid_heading = np.where(v0 & v1, heading, np.nan)
    max_at_end = int(np.argmax(angle[-1]))
    np.testing.assert_allclose(angle[-1, max_at_end], proof['repeat']['by_control'][-1]['max_orientation_deg'],
                               atol=1e-10, rtol=0)
    gates = [proof['scores'][role]['rows'] for role in ROLES]
    time_delta = [abs(left['hold_start_seconds'] - right['hold_start_seconds'])
                  for left, right in zip(*gates, strict=True)
                  if left['hold_start_seconds'] is not None and right['hold_start_seconds'] is not None]
    assert len(time_delta) == 31
    run = read(record / 'protocol.json')
    pass_a, pass_b = (np.asarray(evaluate_end_hold(trace, run['environment'], rows)['stable_by_environment'])
                      for trace, rows in zip((a, b), gates, strict=True))
    assert np.array_equal(pass_a, pass_b) and pass_a.sum() == 31
    result = dict(status='passed-saved-data-only', trace_sha256={str(x.relative_to(ROOT)): sha(x) for x in paths},
                  independent_analysis_sha256=sha(OUT / 'analysis.json'), learning_updates=0, physical_steps=0,
                  final_largest_rotation_pair=dict(environment=max_at_end, total_orientation_deg=float(angle[-1,max_at_end]),
                    heading_deg=float(heading[-1,max_at_end]), tilt_a_deg=float(t0[-1,max_at_end]),
                    tilt_b_deg=float(t1[-1,max_at_end]), both_final_holds=bool(pass_a[max_at_end] and pass_b[max_at_end])),
                  final_max_abs_tilt_difference_deg=float(tilt[-1].max()),
                  max_joint_angle_difference_deg=float(joints.max()),
                  timely_hold_onset_difference_seconds=dict(pairs=len(time_delta), median=float(np.median(time_delta)),
                                                           maximum=float(max(time_delta))),
                  heading_projection_minimum=.1,
                  note='Heading is horizontal body-forward azimuth; tilt is magnitude relative to gravity.')
    write(OUT / 'motion-review.json', result)
    return result, a['time'], angle.max(axis=1), np.nanmax(valid_heading, axis=1), tilt.max(axis=1)


def plot(times, angle, heading, tilt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager
    from figure_style import apply_figure_style
    from render_roll_angle_results import checked_save

    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei', sizes=(11, 10, 9))
    matplotlib.rcParams['savefig.bbox'] = None
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    fig.subplots_adjust(left=.13, right=.97, bottom=.28, top=.88)
    for label, values, color, style in (
        ('完整姿态', angle, '#223c4b', '-'), ('水平朝向', heading, '#2479a8', '--'),
        ('倾角大小', tilt, '#ae752a', ':')
    ):
        ax.plot(np.r_[2., times], np.r_[0., values], label=label, color=color, linestyle=style, linewidth=1.9)
    top = max(10, 10 * math.ceil(max(angle.max(), np.nanmax(heading), tilt.max()) / 10))
    ax.set(xlim=(2, 11), xticks=(2, 3, 5, 8, 11), ylim=(0, top),
           yticks=np.linspace(0, top, 6), xlabel='动作时间（秒）', ylabel='32对样本中的最大差（°）')
    ax.set_title('姿态差异', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.24), ncols=3)
    return checked_save(fig, ASSETS / 'getup-shared-release-drift.png')


def keyframes():
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    source = read(OUT / 'analysis.json')
    record = ROOT / source['record']
    capture = read(record / 'evaluation.json')
    indices = (0, 50, 75, 275)
    times = ('0秒 · 起点', '2秒 · 共享释放', '3秒 · 释放后', '11秒 · 终态')
    width = 320
    image = Image.new('RGB', (4 * width, 555), 'white')
    draw = ImageDraw.Draw(image)
    font = '/mnt/c/Windows/Fonts/msyh.ttc'
    title, text = (ImageFont.truetype(font, size) for size in (25, 17))
    draw.text((10, 5), '共同释放关键帧', fill='#15334b', font=title)
    outcomes = [source['scores'][role]['rows'][0] for role in ROLES]
    assert all(row['env_index'] == 0 and row['hold_start_seconds'] is None for row in outcomes)
    draw.text((10, 37), '固定官方策略 · 预定相机环境0 · 两次均未及时持稳', fill='#15334b', font=text)
    frames = []
    for row, role in enumerate(ROLES):
        proof = source['videos'][role]
        raw = ROOT / proof['path']
        assert sha(raw) == proof['sha256']
        info = next(c['video'] for c in capture['conditions'] if c['condition'] == role)
        reader = imageio_ffmpeg.read_frames(str(raw), pix_fmt='rgb24')
        meta = next(reader)
        assert meta['size'] == (640, 480) and meta['fps'] == 25
        y = 69 + 242 * row
        for i, data in enumerate(reader):
            if i not in indices:
                continue
            col = indices.index(i)
            pic = Image.frombytes('RGB', meta['size'], data)
            pic = pic.crop((0, 46, 640, 480)).resize((304, 206), Image.Resampling.LANCZOS)
            x = col * width + 8
            draw.text((x, y), f'重复{row + 1} · {times[col]}', fill='#15334b', font=text)
            image.paste(pic, (x, y + 27))
            frames.append(dict(role=role, index=i, seconds=i/25, qpos_sha256=info['qpos_sha256_by_frame'][i],
                               video_sha256=proof['sha256']))
        assert i + 1 == proof['frames']
    assert len(frames) == 8 and all(frames[i]['qpos_sha256'] == frames[i+4]['qpos_sha256'] for i in (0, 1))
    dst = ASSETS / 'getup-shared-release-frames.jpg'
    image.save(dst, quality=94)
    return dict(path=str(dst.relative_to(ROOT)), sha256=sha(dst), frames=frames, video_header_crop_pixels=46)


if __name__ == '__main__':
    result, *series = measure()
    write(OUT / 'figure-review.json', dict(status='rendered-awaiting-visual-review', plot=plot(*series), keyframes=keyframes(),
                                         motion_review_sha256=sha(OUT / 'motion-review.json'),
                                         script_sha256=sha(Path(__file__))))
    print(result)
