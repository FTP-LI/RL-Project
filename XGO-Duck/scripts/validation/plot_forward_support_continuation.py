"""Plot cumulative yaw and measure when the paired trajectories separate."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/training'))
from record_io import write


REPORT = ROOT/'records/reports/forward-right-support-continuation'
ROLES = ('onset-only', 'right-support-continuation')
COLORS = ('#1E5D86', '#A64A10')
LEGEND = ('仅首段脉冲', '持续修正', '20秒验收上限', '共同快照')


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def find_font():
    choices = ('C:/Windows/Fonts/msyh.ttc', '/mnt/c/Windows/Fonts/msyh.ttc',
               '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    return next((Path(name) for name in choices if Path(name).is_file()), None)


def read_motion(path, digest, window):
    if sha(path) != digest:
        raise ValueError(f'Trace digest changed: {path}')
    with np.load(path, allow_pickle=False) as data:
        return data['angular_velocity'][window, :, 2].astype(np.float64), data['foot_contact'][window]


def support_statistics(signed, feet):
    supports = feet[..., 0].astype(np.int8) + 2*feet[..., 1].astype(np.int8)
    ages = np.ones_like(supports, dtype=np.int32)
    for control in range(1, len(supports)):
        ages[control] = np.where(supports[control] == supports[control-1], ages[control-1]+1, 1)
    masks = {'no_support': supports == 0, 'double_support': supports == 3}
    for side, value in (('left', 1), ('right', 2)):
        for period, selected in (('early', ages <= 3), ('middle', (ages >= 4) & (ages <= 8)), ('late', ages >= 9)):
            masks[f'{side}_{period}'] = (supports == value) & selected
    parts = {}
    for phase, mask in masks.items():
        parts[phase] = dict(occupancy_fraction=float(mask.mean()),
                            conditional_mean_absolute_yaw_rad_s=float(np.abs(signed[mask]).mean()) if mask.any() else None,
                            pooled_absolute_yaw_contribution_rad_s=float(np.where(mask, np.abs(signed), 0.).mean()),
                            pooled_signed_yaw_contribution_rad_s=float(np.where(mask, signed, 0.).mean()))
    np.testing.assert_allclose(sum(row['pooled_absolute_yaw_contribution_rad_s'] for row in parts.values()),
                               np.abs(signed).mean(), atol=1e-10, rtol=0)
    return dict(median_world_time_mean_abs_yaw_rad_s=float(np.median(np.abs(signed).mean(axis=0))),
                median_world_abs_time_mean_signed_yaw_rad_s=float(np.median(np.abs(signed.mean(axis=0)))),
                fraction_positive=float((signed > 0).mean()), phase_contributions=parts)


def before_first_extension(folder, receipt):
    branches = [folder/'clips'/role for role in ROLES]
    expected = receipt['branches'][ROLES[1]]['trace_sha256']['commands.npz']
    command = branches[1]/'commands.npz'
    assert sha(command) == expected
    with np.load(command, allow_pickle=False) as data:
        offset = int(np.flatnonzero(data['right_extension_applied'].any(axis=1))[0])
    fields = ('qpos', 'qvel')
    before = []
    for branch in branches:
        with np.load(branch/'trace.npz', allow_pickle=False) as data:
            state = {key: data[key][:offset+1].copy() for key in fields}
            state['yaw'] = data['angular_velocity'][:offset+1, :, 2].copy()
            before.append(state)
    return dict(first_extension_zero_based_control=receipt['shared_controls']+offset,
                maximum_absolute_yaw_delta_rad_s=float(np.abs(before[1]['yaw']-before[0]['yaw']).max()),
                maximum_absolute_state_delta={key: float(np.abs(before[1][key]-before[0][key]).max()) for key in fields})


def measures(case):
    folder = ROOT/'records/evaluations'/case['run_id']
    receipt = read(folder/'guarded-evaluation.json')
    if sha(folder/'guarded-evaluation.json') != case['collection_sha256']:
        raise ValueError('Collection receipt changed')
    common = read_motion(folder/'shared/trace.npz', receipt['prefix_sha256']['trace.npz'], slice(51, None))
    arms = []
    mechanics = {}
    for role in ROLES:
        meta = receipt['branches'][role]
        suffix = read_motion(folder/'clips'/role/'trace.npz', meta['trace_sha256']['trace.npz'], slice(1, None))
        signed, feet = (np.concatenate((before, after)) for before, after in zip(common, suffix, strict=True))
        samples = np.abs(signed)
        assert samples.shape == (1000, 128)
        mean = samples.mean(axis=0)
        np.testing.assert_allclose(np.median(mean), case['arms'][role]['median_abs_yaw_rad_s'], atol=1e-6, rtol=0)
        arms.append(samples)
        mechanics[role] = support_statistics(signed, feet)
    difference = arms[1]-arms[0]
    np.testing.assert_allclose(np.median(difference.mean(axis=0)), case['paired_yaw_median_change_rad_s'], atol=1e-6, rtol=0)
    curves = [np.median(np.cumsum(samples, axis=0)/np.arange(1, 1001)[:, None], axis=1) for samples in arms]
    windows = {}
    for start, end in ((0, 2), (2, 6), (6, 12), (12, 20)):
        means = difference[50*start:50*end].mean(axis=0)
        windows[f'{start}-{end}'] = dict(paired_worlds=len(means), paired_median_delta_rad_s=float(np.median(means)),
                                       worlds_with_lower_yaw=int((means < 0).sum()))
    assert np.array_equal(arms[0][:100], arms[1][:100])
    return curves, dict(paired_windows=windows, support=mechanics, before_first_extension=before_first_extension(folder, receipt))


def render_matplotlib(cases, curves):
    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import font_manager, pyplot as plt
    from matplotlib.lines import Line2D
    from figure_style import apply_figure_style
    from plot_forward_hip_yaw_response import check_layout

    font = find_font()
    if font:
        font_manager.fontManager.addfont(font)
    apply_figure_style(font=font_manager.FontProperties(fname=font).get_name() if font else None,
                       sizes=(11, 10, 9))
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.55), sharex=True, sharey=True)
    fig.subplots_adjust(left=.13, right=.98, bottom=.113, top=.83, wspace=.18, hspace=.34)
    timeline = np.arange(1, 1001)*.02
    lines = ('-', '--')
    for ax, (key, case) in zip(axes.flat, cases.items(), strict=True):
        for curve, color, line in zip(curves[key], COLORS, lines, strict=True):
            ax.plot(timeline, curve, color=color, linestyle=line, linewidth=1.65)
        ax.axhline(.20, color='#78838A', linestyle=':', linewidth=.85)
        ax.axvline(2., color='#989EA3', linestyle='-.', linewidth=.85)
        ax.set_title(f'种子{case["seed"]}／摩擦{case["foot_friction"]:.1f}', loc='left', pad=6)
        ax.set_xlim(0., 20.3)
        ax.set_xticks((0, 5, 10, 15, 20))
    fig.supxlabel('前进时间（秒）', y=.035)
    fig.supylabel('累计平均绝对偏航速度（rad/s）', x=.027)
    legend = [Line2D([], [], color=color, linestyle=line, label=label) for color, line, label in (
        (COLORS[0], lines[0], LEGEND[0]), (COLORS[1], lines[1], LEGEND[1]),
        ('#78838A', ':', LEGEND[2]), ('#989EA3', '-.', LEGEND[3]))]
    fig.legend(handles=legend, loc='upper center', ncol=4, bbox_to_anchor=(.53, .977))
    check_layout(fig)
    fig.savefig(REPORT/'cumulative-forward-yaw.png')
    plt.close(fig)


def render_pillow(cases, curves):
    from PIL import Image, ImageDraw, ImageFont

    font = find_font()
    if not font:
        raise FileNotFoundError('No Chinese font found for the Windows report plot')
    title, label, tick = (ImageFont.truetype(str(font), size) for size in (37, 28, 23))
    figure = Image.new('RGB', (1650, 1160), 'white')
    draw = ImageDraw.Draw(figure)
    ink, grey, grid = '#17344A', '#586976', '#E6EBEF'
    draw.text((92, 27), '累计平均绝对偏航速度（rad/s）', font=title, fill=ink)
    keys = (*COLORS, '#687884', '#9BA4AB')
    for index, (text, color) in enumerate(zip(LEGEND, keys, strict=True)):
        x = (92, 420, 720, 1130)[index]
        draw.line((x, 112, x+46, 112), fill=color, width=4)
        draw.text((x+62, 95), text, font=label, fill=ink)
    curve_top = max(float(y.max()) for arms in curves.values() for y in arms)
    top = max(.25, np.ceil(curve_top/.05)*.05)
    for index, (key, case) in enumerate(cases.items()):
        x, y = 109+810*(index%2), 227+443*(index//2)
        width, height = 620, 324
        draw.text((x, y-52), f'种子{case["seed"]}／脚底摩擦系数{case["foot_friction"]:.1f}', font=label, fill=ink)
        for value in np.arange(0, top+.001, .05):
            row = y+height-round(height*value/top)
            draw.line((x, row, x+width, row), fill=grid, width=1)
            draw.text((x-13, row), f'{value:.2f}', anchor='rm', font=tick, fill=grey)
        for t in (0, 5, 10, 15, 20):
            col = x+round(width*t/20)
            draw.line((col, y+height, col, y+height+8), fill=grey, width=2)
            draw.text((col, y+height+18), str(t), anchor='mt', font=tick, fill=grey)
        for col in range(x, x+width, 17):
            bound = y+height-round(height*.20/top)
            draw.line((col, bound, min(col+8, x+width), bound), fill=keys[2], width=2)
        boundary = x+round(width*2/20)
        for row in range(y, y+height, 17):
            draw.line((boundary, row, boundary, min(row+9, y+height)), fill=keys[3], width=2)
        draw.line((x, y, x, y+height, x+width, y+height), fill=grey, width=2)
        for order, (curve, color) in enumerate(zip(curves[key], COLORS, strict=True)):
            points = list(zip(x+np.arange(1, 1001)*width/1000, y+height-curve*height/top))
            if order == 0:
                draw.line(points, fill=color, width=4)
            else:
                for start in range(0, len(points), 24):
                    draw.line(points[start:start+15], fill=color, width=4)
    draw.text((825, 1126), '前进时间（秒）', anchor='mm', font=label, fill=ink)
    figure.save(REPORT/'cumulative-forward-yaw.png', dpi=(180, 180))


def main():
    comparison = REPORT/'closed-loop-comparison.json'
    expected = read(ROOT/'project.json')['goal_execution']['forward_right_support_continuation']['result_sha256']
    assert sha(comparison) == expected
    cases = read(comparison)['cases']
    curves, windows = {}, {}
    for key, case in cases.items():
        curves[key], windows[key] = measures(case)
    try:
        import matplotlib
    except ModuleNotFoundError:
        render_pillow(cases, curves)
    else:
        render_matplotlib(cases, curves)
    write(REPORT/'paired-yaw-windows.json', dict(
        measurement='Median across paired worlds of the difference of time-mean absolute yaw in each window',
        support_basis='Each arm is partitioned by its actual end-of-control foot contacts; signed and absolute terms are pooled over all worlds and controls',
        seconds_after_forward_entry=True, source_sha256=expected, cases=windows))
    print('FORWARD_SUPPORT_WINDOWS_VERIFIED', len(cases), flush=True)


if __name__ == '__main__':
    main()
