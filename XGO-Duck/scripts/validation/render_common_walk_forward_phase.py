"""Show paired forward yaw errors and the frozen improvement threshold."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
import numpy as np

from s1_reference_inputs import ROOT, read, sha
from figure_style import apply_figure_style
from record_io import write
from render_roll_angle_results import checked_save


OUT = ROOT / 'records/reports/common-walk-forward-phase'
PLAN = ROOT / 'configs/experiments/common_walk_forward_phase_screen.json'
ASSET = ROOT / 'docs/reports/assets/walk-forward-phase-yaw.png'
COLORS = {'reference-a': '#50697e', 'reference-b': '#6a9cac', 'phase-yaw': '#ad612a'}
NAMES = {'reference-a': '官方一', 'reference-b': '官方二', 'phase-yaw': '相位候选'}


def main():
    path = OUT / 'analysis.json'
    result, plan = read(path), read(PLAN)
    assert result['status'] == 'passed-integrity' and result['protocol_sha256'] == sha(PLAN)
    series = {name: np.array([row['yaw_error'] for row in result['scores'][name]['forward']['episode_scores']])
              for name in COLORS}
    order = np.argsort((series['reference-a'] + series['reference-b']) / 2)
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei', sizes=(11, 10, 9))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={'width_ratios': [1.4, 1]})
    fig.subplots_adjust(left=.096, right=.97, bottom=.18, top=.79, wspace=.37)
    worlds = np.arange(1, 33)
    ax = axes[0]
    for role, color in COLORS.items():
        ax.plot(worlds, series[role][order], 'o-', label=NAMES[role], color=color, markersize=3, linewidth=.9)
    ceiling = plan['environment']['criteria']['yaw_error_max']
    ax.axhline(ceiling, color='#7d8891', linestyle='--', linewidth=.85)
    ax.set(xlim=(0, 33), xticks=(1, 8, 16, 24, 32), xlabel='按官方平均误差排列的配对环境',
           ylabel='整段偏航绝对误差（rad/s）')
    ax.set_title('逐环境偏航')
    ax = axes[1]
    comparisons = result['comparison']['comparisons']
    for y, role in enumerate(('reference-b', 'reference-a')):
        row = comparisons[role]
        avg = row['mean_yaw_gain_rad_s']
        low, high = row['paired_bootstrap95_rad_s']
        ax.errorbar(avg, y, xerr=[[avg-low], [high-avg]], fmt='o', color=COLORS[role], capsize=5)
    ax.axvline(0, color='#7d8891', linewidth=.85)
    gate = plan['gate']['minimum_yaw_gain_rad_s']
    ax.axvline(gate, color='#ad612a', linestyle='--', linewidth=.85)
    span = [0., gate, *(x for item in comparisons.values() for x in item['paired_bootstrap95_rad_s'])]
    lower, upper = min(span), max(span)
    padding = .16 * (upper - lower)
    ax.set(yticks=(0, 1), yticklabels=('对官方二', '对官方一'), ylim=(-.65, 1.65),
           xlim=(lower-padding, upper+padding), xticks=np.linspace(lower, upper, 6),
           xlabel='官方误差减候选误差（rad/s）')
    ax.set_title('平均收益与 95% 区间')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, .975), ncols=3)
    image = checked_save(fig, ASSET)
    write(OUT / 'figure-review.json', dict(status='rendered-awaiting-visual-review', figure=image,
          source_sha256=sha(path), script_sha256=sha(Path(__file__)), new_physical_steps=0, learning_updates=0))


if __name__ == '__main__':
    main()
