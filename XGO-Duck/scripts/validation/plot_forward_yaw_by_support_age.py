"""Plot the measured low-friction yaw gap by contact phase and phase age."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import font_manager, pyplot as plt
import numpy as np

from plot_forward_hip_yaw_response import check_layout


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'records/reports/forward-support-joint-pulse'
PHASES = {
    'left_controls_1_to_3': '左脚单独支撑第1至3步',
    'left_controls_4_to_8': '左脚单独支撑第4至8步',
    'left_controls_9_and_later': '左脚单独支撑第9步起',
    'right_controls_1_to_3': '右脚单独支撑第1至3步',
    'right_controls_4_to_8': '右脚单独支撑第4至8步',
    'right_controls_9_and_later': '右脚单独支撑第9步起',
    'double_support': '双脚支撑',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style-kernel', required=True, type=Path)
    parser.add_argument('--font', required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('figure_style', args.style_kernel)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    font_manager.fontManager.addfont(args.font)
    family = font_manager.FontProperties(fname=args.font).get_name()
    style.apply_figure_style(font=family, sizes=(11, 10, 9))
    data = json.loads((REPORT/'archived-yaw-by-support-age.json').read_bytes())
    assert data['samples'] == 512000 and all(row['samples'] == 128000 for row in data['cases'].values())
    groups = data['pairs']
    assert set(groups) == {'1247', '1248'}
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    fig.subplots_adjust(left=.31, right=.96, bottom=.17, top=.84)
    positions = np.arange(len(PHASES))
    for seed, color, marker, offset in (('1247', '#1E5D86', 'o', -.13), ('1248', '#A64A10', 'D', .13)):
        values = groups[seed]['contribution_by_contact_and_age_rad_s']
        assert values['no_support'] == 0
        np.testing.assert_allclose(sum(values.values()), groups[seed]['pooled_mean_abs_yaw_gap_rad_s'], rtol=0, atol=1e-12)
        ax.scatter([1000*values[name] for name in PHASES], positions+offset,
                   s=30, color=color, marker=marker, label=f'种子{seed}', zorder=3)
    ax.axvline(0, color='#72808B', linewidth=.8, zorder=1)
    ax.set_yticks(positions, PHASES.values())
    ax.set_ylim(len(PHASES)-.5, -.5)
    ax.set_xlabel('低摩擦组与标称组绝对偏航差的分项贡献（毫弧度/秒）', labelpad=8)
    ax.set_xticks([-2, 0, 5, 10, 15])
    ax.set_xlim(-2.8, 17.3)
    ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02), ncols=2, borderaxespad=0)
    # Validate the final physical placement before publishing the artifact.
    check_layout(fig)
    target = REPORT/'yaw-gap-by-support-age.png'
    fig.savefig(target)
    plt.close(fig)
    print('SUPPORT_AGE_FIGURE_VERIFIED', target, flush=True)


if __name__ == '__main__':
    main()
