"""Plot the signed pulse response and numerical repeat envelope on low friction."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'records/reports/forward-hip-yaw-pulse'


def check_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [(t, t.get_window_extent(renderer)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    spines = [(s, s.get_window_extent(renderer)) for ax in fig.axes for s in ax.spines.values() if s.get_visible()]
    own_ticks = {ax: set(ax.get_xticklabels()+ax.get_yticklabels()) for ax in fig.axes}
    outside = [label.get_text() for label, box in labels
               if not (fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1))]
    assert not outside, outside
    collisions = [(a.get_text(), b.get_text()) for idx, (a, first) in enumerate(labels) for b, second in labels[idx+1:] if first.overlaps(second)]
    collisions += [(label.get_text(), 'spine') for label, text in labels for spine, line in spines
                   if text.overlaps(line) and label not in own_ticks[spine.axes]]
    assert not collisions, collisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style-kernel', required=True, type=Path)
    parser.add_argument('--qa-dir', type=Path)
    args = parser.parse_args()
    loader = importlib.util.spec_from_file_location('pulse_figure_style', args.style_kernel)
    style = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(style)
    style.apply_figure_style(sizes=(11, 9, 8))
    projects = json.loads((ROOT/'project.json').read_text(encoding='utf-8'))
    evidence = projects['goal_execution']['forward_hip_yaw_pulse']
    raw = (ROOT/evidence['result']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == evidence['result_sha256']
    report = json.loads(raw)
    cases = [row for row in report['cases'].values() if row['foot_friction'] == .8]
    assert [row['seed'] for row in cases] == [1247, 1248]
    time = np.arange(1, 9)*20
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.65), sharex=True)
    fig.subplots_adjust(left=.127, right=.985, bottom=.185, top=.77, wspace=.42)
    colors, markers, styles = ('#1E5D86', '#AE4D10'), ('o', 'd'), ('-', '--')
    for side, ax in zip(('left', 'right'), axes, strict=True):
        items = [row['sides'][side] for row in cases]
        pairs = [item['complete'] for item in items]
        floor = np.max([c['zero_repeat_median_abs_difference_by_control_rad_s'] for c in pairs], axis=0)
        ax.fill_between(time, -floor, floor, color='#D4D9DD', linewidth=.7, edgecolor='#7B848B')
        ax.axhline(0, color='#596571', linewidth=.65, zorder=0)
        for pair, color, marker, line in zip(pairs, colors, markers, styles, strict=True):
            ax.plot(time, pair['odd_signed_yaw_by_control_rad_s'], marker=marker, markersize=3.1, linewidth=1.5,
                    linestyle=line, color=color)
        ax.set_title(f'{side.title()} foot contact', pad=8)
        ax.set_xticks((20, 60, 100, 140, 160))
        ax.margins(x=.06, y=.15)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    fig.supxlabel('Time after the target pulse (ms)', y=.045)
    fig.supylabel('Signed yaw half-difference (rad/s)', x=.025)
    keys = [Line2D([], [], color=color, marker=marker, linestyle=line, label=f'Seed {row["seed"]}')
            for row, color, marker, line in zip(cases, colors, markers, styles, strict=True)]
    keys.append(Patch(facecolor='#D4D9DD', edgecolor='#7B848B', label='Zero-repeat median absolute difference'))
    fig.legend(handles=keys, loc='upper center', ncol=3, bbox_to_anchor=(.55, .99))
    output = REPORT/'low-friction-pulse-response.png'
    fig.savefig(output)
    check_layout(fig)
    if args.qa_dir:
        args.qa_dir.mkdir(exist_ok=True)
        with Image.open(output) as png:
            for name, box in style.panel_crops(fig).items():
                png.crop(box).save(args.qa_dir/f'panel-{name}.png')
    plt.close(fig)
    print(f'PULSE_FIGURE_VERIFIED {output}', flush=True)


if __name__ == '__main__':
    main()
