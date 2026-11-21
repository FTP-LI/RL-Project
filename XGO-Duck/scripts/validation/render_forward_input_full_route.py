"""Render the scored full-route yaw trajectories and synchronized actual snapshots."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_forward_support_continuation_keyframes import CONTACTS, archived_frames
from plot_forward_support_continuation import ROOT, find_font, read, sha, write


REPORT = ROOT/'records/reports/forward-input-full-route'
COLORS = '#1E5D86', '#A64A10'


def motion(case, role):
    folder = ROOT/'records/evaluations'/case['run_id']
    record = folder/'guarded-evaluation.json'
    assert sha(record) == case['collection_sha256']
    proof = read(record)
    source = folder/'shared/trace.npz'
    branch = folder/'clips'/role/'trace.npz'
    assert sha(source) == proof['prefix_sha256']['trace.npz']
    assert sha(branch) == proof['branches'][role]['trace_sha256']['trace.npz']
    with np.load(source, allow_pickle=False) as common, np.load(branch, allow_pickle=False) as series:
        yaw = np.abs(np.concatenate((common['angular_velocity'][51:, :, 2], series['angular_velocity'][1:, :, 2]))).astype(np.float64)
        feet = np.concatenate((common['foot_contact'], series['foot_contact'][1:]))
    assert yaw.shape == (1000, 128) and feet.shape == (1051, 128, 2)
    curve = np.median(np.cumsum(yaw, axis=0)/np.arange(1, 1001)[:, None], axis=1)
    np.testing.assert_allclose(curve[-1], case['arms'][role]['median_abs_yaw_rad_s'], atol=1e-6, rtol=0)
    return curve, feet, folder


def render_curves(cases, series, label, *, output=None, marker_label='共同快照',
                  heading='20秒前进累计平均绝对偏航（rad/s）'):
    title, body, small = (ImageFont.truetype(str(find_font()), size) for size in (38, 27, 23))
    image = Image.new('RGB', (1660, 1110), '#FFFFFF')
    draw = ImageDraw.Draw(image)
    ink, subtle, grid = '#17344A', '#5C6C77', '#E4EAEE'
    draw.text((82, 25), heading, font=title, fill=ink)
    for x, name, color in ((88, '参照', COLORS[0]), (285, label, COLORS[1]), (735, '20秒上限', subtle), (1040, marker_label, '#9BA4AB')):
        draw.line((x, 104, x+38, 104), fill=color, width=4)
        draw.text((x+50, 86), name, fill=ink, font=body)
    high = max(.25, np.ceil(max(curve.max() for curves in series.values() for curve in curves)/.05)*.05)
    for i, (key, case) in enumerate(cases.items()):
        x, y = 109+811*(i%2), 210+425*(i//2)
        w, h = 635, 308
        draw.text((x, y-52), f'种子{case["seed"]}／脚底摩擦{case["foot_friction"]:.1f}', font=body, fill=ink)

        def row(value):
            return y+h-round(value*h/high)

        for tick in np.arange(0, high+.001, .05):
            yy = row(tick)
            draw.line((x, yy, x+w, yy), fill=grid)
            draw.text((x-13, yy), f'{tick:.2f}', anchor='rm', font=small, fill=subtle)
        for sec in (0, 4, 8, 12, 16, 20):
            col = x+round(w*sec/20)
            draw.text((col, y+h+15), str(sec), anchor='mt', font=small, fill=subtle)
        for col in range(x, x+w, 17):
            draw.line((col, row(.20), min(col+8, x+w), row(.20)), fill=subtle, width=2)
        boundary = x+round(w*.2)
        for yy in range(y, y+h, 17):
            draw.line((boundary, yy, boundary, min(yy+8, y+h)), fill='#9BA4AB', width=2)
        draw.line((x, y, x, y+h, x+w, y+h), fill=subtle, width=2)
        for values, color in zip(series[key], COLORS, strict=True):
            points = list(zip(x+np.arange(1, 1001)*w/1000, y+h-values*h/high))
            draw.line(points, fill=color, width=4)
    draw.text((825, 1067), '前进时间（秒）', anchor='mm', font=body, fill=ink)
    image.save(output or REPORT/'full-forward-yaw.png', dpi=(180, 180))


def render_frames(cases, movements, roles, candidate):
    title, group, label = (ImageFont.truetype(str(find_font()), size) for size in (31, 23, 19))
    low = [(name, row) for name, row in cases.items() if row['foot_friction'] == .8]
    frame = Image.new('RGB', (1159, 1416), '#F3F7F9')
    draw = ImageDraw.Draw(frame)
    ink, subtle = '#17344A', '#5C6C77'
    draw.text((24, 14), '低摩擦前进关键帧', font=title, fill=ink)
    draw.text((24, 58), '环境0实景。时间从站立起点计，标注取同期物理接触。', font=label, fill=subtle)
    controls = 250, 550, 1050
    labels = '共同快照', '稳定前进', '路线终点'
    evidence = {}
    for group_number, (name, case) in enumerate(low):
        top = 98+group_number*650
        draw.text((24, top), f'种子{case["seed"]}／脚底摩擦0.8', font=group, fill=ink)
        evidence[name] = {}
        for row, (role, role_label) in enumerate(zip(roles, ('参照', candidate), strict=True)):
            _, feet, folder = movements[name][role]
            snapshots, frames = archived_frames(folder, role, controls)
            y = top+41+row*306
            for i, (step, caption) in enumerate(zip(controls, labels, strict=True)):
                x = 24+381*i
                frame.paste(snapshots[step].resize((350, 237), Image.Resampling.LANCZOS), (x, y))
                left, right = feet[step, 0]
                support = CONTACTS[int(left)+2*int(right)]
                draw.text((x, y+240), f'{role_label} · {caption} · {step*.02:.2f}秒', font=label, fill=ink)
                draw.text((x, y+261), support, font=label, fill=subtle)
                frames[step].update(time_seconds=step*.02, actual_support=support)
            evidence[name][role] = frames
    frame.save(REPORT/'full-forward-keyframes.jpg', quality=91, subsampling=0)
    return evidence


def main():
    result_file = REPORT/'full-route-comparison.json'
    project = read(ROOT/'project.json')['goal_execution']['forward_input_full_route']
    assert project['status'] == 'reviewed' and sha(result_file) == project['result_sha256']
    result = read(result_file)
    cases, roles = result['cases'], ('reference-a', result['candidate'])
    name = f'模型输入{result["internal_input_m_s"]:.3f}m/s'
    data = {key:{role:motion(row, role) for role in roles} for key, row in cases.items()}
    render_curves(cases, {key:[data[key][role][0] for role in roles] for key in cases}, name)
    frames = render_frames(cases, data, roles, name)
    write(REPORT/'full-forward-media.json', dict(source_sha256=sha(result_file), snapshots=frames,
          graphs_sha256=sha(REPORT/'full-forward-yaw.png'), keyframes_sha256=sha(REPORT/'full-forward-keyframes.jpg'),
          environment_index=0, additional_physical_steps=0))
    print('FORWARD_FULL_MEDIA_VERIFIED', len(cases), sum(len(arm) for case in frames.values() for arm in case.values()), flush=True)


if __name__ == '__main__':
    main()
