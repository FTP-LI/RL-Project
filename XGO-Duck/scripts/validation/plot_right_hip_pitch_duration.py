"""Plot the causal yaw response of the two right-support duration arms."""

import argparse
import importlib.util
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import font_manager, pyplot as plt
import numpy as np

from check_right_hip_pitch_duration_window import KEY, REPORT
from plot_forward_hip_yaw_response import check_layout
from run_roll_stand_forward_diagnostic import P, ROOT, read, sha


def extract(case):
    folder = (ROOT/case['video']['path']).parent
    forks = folder/'pulse-forks.json'
    assert sha(forks) == read(folder/'guarded-evaluation.json')['forks_sha256']
    saved = read(forks)['right']
    selected = saved['selected_worlds']
    assert len(selected) == case['sides']['right']['worlds']
    series = {}
    for role, digest in saved['branches'].items():
        source = folder/'right'/f'{role}.npz'
        assert sha(source) == digest
        with np.load(source, allow_pickle=False) as segment:
            series[role] = np.abs(segment['state_angular_velocity'][1:, selected, 2].astype(np.float64))
    a, b = series['zero-a'], series['zero-b']
    zero = (a+b)*.5
    differences = [series[arm]-zero for arm in ('one-extra-control', 'continue-through-eight')]
    for key, count in (('primary', 8), ('complete', 16)):
        stored = case['sides']['right'][key]
        values = stored['median_change_in_mean_abs_yaw_rad_s']
        np.testing.assert_allclose([np.median(group[:count].mean(axis=0)) for group in differences], list(values.values()), atol=1e-14, rtol=0)
        np.testing.assert_allclose(np.quantile(np.abs((a[:count]-b[:count]).mean(axis=0)), .95), stored['zero_pair_gap_p95_rad_s'], atol=1e-14, rtol=0)
    return [np.median(group, axis=1) for group in differences], np.quantile(np.abs(a-b), .95, axis=1)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--style-kernel', required=True, type=Path)
    cli.add_argument('--font', required=True, type=Path)
    args = cli.parse_args()
    spec = importlib.util.spec_from_file_location('figure_style', args.style_kernel)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    font_manager.fontManager.addfont(args.font)
    style.apply_figure_style(font=font_manager.FontProperties(fname=args.font).get_name(), sizes=(11, 10, 9))
    state = read(P)['goal_execution'][KEY]
    source = ROOT/state['result']
    assert state['status'] == 'reviewed' and sha(source) == state['result_sha256']
    cases = read(source)['cases']
    rows = [cases[f's{seed}-friction-low'] for seed in (1247, 1248)]
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8.6, 3.8))
    fig.subplots_adjust(left=.14, right=.98, bottom=.183, top=.73, wspace=.14)
    time = np.arange(1, 17)*20
    for ax, case in zip(axes, rows, strict=True):
        median, noise = extract(case)
        ax.fill_between(time, -noise, noise, color='#D4D9DD', edgecolor='#7B848B', linewidth=.65, label='逐步双零差第95百分位')
        ax.axhline(0, color='#5D6872', linewidth=.65)
        ax.axvline(160, color='#8A949B', linewidth=.75, linestyle=':', label='160毫秒主评分点')
        for values, color, line, label in zip(median, ('#1E5D86', '#A64A10'), ('-', '--'), ('追加一拍', '持续修正'), strict=True):
            ax.plot(time, values, linewidth=1.5, color=color, linestyle=line, label=label)
        ax.set_title(f'种子{case["seed"]}，{case["sides"]["right"]["worlds"]}个场景', loc='left', pad=6)
        ax.set_xticks((20, 100, 160, 240, 320))
        ax.margins(x=.06, y=.08)
    fig.supxlabel('从分支开始的时间（毫秒）', y=.051)
    fig.supylabel('绝对偏航变化（rad/s）', x=.036)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(.55, .986))
    check_layout(fig)
    target = REPORT/'matched-right-support-yaw-response.png'
    fig.savefig(target)
    plt.close(fig)
    print('RIGHT_SUPPORT_DURATION_FIGURE_VERIFIED', target, flush=True)


if __name__ == '__main__':
    main()
