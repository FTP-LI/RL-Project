"""Plot verified cumulative yaw over the complete forward window under low friction."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import font_manager, pyplot as plt
import numpy as np

from plot_forward_hip_yaw_response import check_layout


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'records/reports/forward-phase-hip-pitch'


def cumulative_yaw(run):
    folder = ROOT/'records/evaluations'/run
    raw = json.loads((folder/'raw-score.json').read_bytes())
    manifest = json.loads((folder/'guarded-evaluation.json').read_bytes())
    trace = folder/'trace.npz'
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == manifest['trace_sha256']
    selected = np.asarray(raw['forward_at']) == 50
    assert selected.any() and set(raw['forward_at']) <= {-1, 50}
    with np.load(trace, allow_pickle=False) as state:
        part = slice(51, 1051)
        yaw = np.abs(state['angular_velocity'][part, selected, 2].astype(np.float64))
        active = state['active'][part, selected]
    assert yaw.shape == (1000, int(selected.sum()))
    total = np.cumsum(np.where(active, yaw, 0.), axis=0)
    count = np.cumsum(active, axis=0)
    mean = np.divide(total, count, where=count > 0, out=np.full_like(total, np.nan))
    return np.nanmedian(mean, axis=1), int(selected.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style-kernel', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('figure_style', args.style_kernel)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    font_manager.fontManager.addfont(args.font)
    style.apply_figure_style(font=font_manager.FontProperties(fname=args.font).get_name(), sizes=(11, 10, 9))
    project = json.loads((ROOT/'project.json').read_bytes())
    evidence = project['goal_execution']['forward_phase_hip_pitch']
    content = (ROOT/evidence['result']).read_bytes()
    assert hashlib.sha256(content).hexdigest() == evidence['result_sha256']
    cases = json.loads(content)['cases']
    groups = [cases[f's{seed}-friction-low'] for seed in (1247, 1248)]
    assert all(case['foot_friction'] == .8 for case in groups)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.85), sharex=True, sharey=True)
    fig.subplots_adjust(left=.13, right=.98, bottom=.175, top=.73, wspace=.14)
    time = np.arange(1, 1001)*.02
    for ax, case in zip(axes, groups, strict=True):
        for metric, record, color, line, label in (
                ('official', 'official_run_id', '#1E5D86', '-', '官方策略'),
                ('timed_hip_pitch', 'candidate_run_id', '#A64A10', '--', '定时髋俯仰')):
            curve, count = cumulative_yaw(case[record])
            assert count == case[metric]['entered']
            np.testing.assert_allclose(curve[-1], case[metric]['median_abs_yaw_rad_s'], atol=1e-6, rtol=0)
            ax.plot(time, curve, color=color, linestyle=line, linewidth=1.65, label=label)
        ax.axhline(.20, color='#78838A', linestyle=':', linewidth=.85, label='20秒验收上限')
        ax.axvline(2., color='#989EA3', linestyle='-.', linewidth=.8, label='允许脉冲触发')
        ax.set_title(f'种子{case["seed"]}', loc='left', pad=6)
        ax.set_xlim(0., 20.3)
        ax.set_xticks((0, 5, 10, 15, 20))
    fig.supxlabel('进入前进后的时间（秒）', y=.046)
    fig.supylabel('平均绝对偏航速度（rad/s）', x=.034)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(.54, .986))
    check_layout(fig)
    out = REPORT/'low-friction-forward-cumulative-yaw.png'
    fig.savefig(out)
    plt.close(fig)
    print('FORWARD_CUMULATIVE_YAW_FIGURE_VERIFIED', out, flush=True)


if __name__ == '__main__':
    main()
