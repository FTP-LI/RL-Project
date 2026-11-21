"""Plot solved normal loads from the verified failure intervals."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
import numpy as np

from figure_style import apply_figure_style
from render_roll_angle_results import checked_save
from review_roll_contact_clock_boundaries import OUT
from s1_reference_inputs import ROOT, read, sha
from record_io import write


def main():
    boundary_path = OUT/'boundary-review.json'
    review = read(boundary_path)
    assert review['status'] == 'passed-saved-data-only' and not review['heldout_used']
    assert sha(OUT/'analysis.json') == review['clock_analysis_sha256']
    rows = [row for row in review['rows'] if not row['success']]
    values = np.concatenate([np.column_stack((row['force_at_each_control_offset_n'], row['endpoint_force_n']))
                             for row in rows], axis=0)
    samples = review['coarse_gaps']['failure']
    assert values.shape == (samples['control_intervals'], 6) and len(rows) == samples['worlds']
    q1, median, q3 = np.quantile(values, [.25, .5, .75], axis=0)
    np.testing.assert_allclose(median[:5], samples['median_force_by_offset_n'], atol=0., rtol=0.)
    ms = np.arange(6)*4
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei', sizes=(11, 10, 9))
    matplotlib.rcParams['savefig.bbox'] = None
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fig.subplots_adjust(left=.15, right=.94, bottom=.19, top=.87)
    ax.axvspan(0, 20, color='#dbe7eb', alpha=.48)
    ax.fill_between(ms, q1, q3, color='#78a6b5', alpha=.35, label='四分位范围')
    ax.plot(ms, median, 'o-', color='#16627e', linewidth=1.8, markersize=5, label='法向力中位数')
    ax.axvline(20, color='#845334', linewidth=1., linestyle='--', label='下一步求解')
    peak = float(q3.max())
    tick_interval = max(1., np.ceil(peak/5))
    ticks = np.arange(0, peak+tick_interval, tick_interval)
    ax.set(xlabel='相对控制步开始（ms）', ylabel='地面法向力（N）', xlim=(-1., 21.),
           ylim=(-.35, ticks[-1]+.4), xticks=ms, yticks=ticks)
    ax.set_title(f'原信号标为离地的区间（{len(rows)} 个失败场景）', loc='left')
    ax.legend(loc='lower left')
    figure = checked_save(fig, ROOT/'docs/reports/assets/roll-contact-clock-force.png')
    write(OUT/'force-figure.json', dict(status='rendered', figure=figure,
        source_sha256=sha(boundary_path), script_sha256=sha(Path(__file__)),
        failed_worlds=len(rows), intervals=len(values), relative_ms=ms.tolist(),
        median_n=median.tolist(), quartiles_n=[q1.tolist(), q3.tolist()],
        final_value_is_next_control_solve_at_same_pose=True, new_physical_steps=0))


if __name__ == '__main__':
    main()
