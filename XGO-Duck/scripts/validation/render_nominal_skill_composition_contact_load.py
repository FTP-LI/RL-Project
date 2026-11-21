"""Plot solved loads in the nominal roll's missed scoring intervals."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
import numpy as np

from figure_style import apply_figure_style
from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha


def main():
    state = read(ROOT / 'project.json')['goal_execution']['nominal_skill_composition_contact_load']
    review_file = ROOT / state['contact_load_review']
    review = read(review_file, state['contact_load_review_sha256'])
    cohort = review['groups']['insufficient_supported_rotation']
    cases = cohort['cases']
    assert len(cases) == cohort['intervals'] and cases
    folder = (ROOT / state['collection']).parent
    source = folder / 'contact-clock.npz'
    assert sha(source) == review['contact_clock_sha256']
    first, last = review['capture_control_boundaries']
    rows = np.asarray([row['control'] - first - 1 for row in cases])
    worlds = np.asarray([row['world'] for row in cases])
    with np.load(source, allow_pickle=False) as stored:
        loads = stored['normal_force_n'][:, :, 0].reshape(last - first, 5, review['worlds'])
    values = np.column_stack((loads[rows, :, worlds], loads[rows + 1, 0, worlds]))
    assert np.count_nonzero(values[:, -1] > review['load_threshold_n']) == cohort['next_substep_loaded']
    assert np.count_nonzero(values[:, -2] > review['load_threshold_n']) == cohort['final_substep_loaded']
    weight = review['nominal_weight_n']
    fractions = values / weight
    low, high = np.quantile(fractions, (.25, .75), axis=0)
    median = np.median(fractions, axis=0)
    np.testing.assert_array_equal(median[:5], cohort['median_load_by_physics_substep_body_weights'])
    times = np.arange(6) * review['physics_period_s'] * 1000.
    threshold = review['load_threshold_n'] / weight

    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei', sizes=(12, 11, 10))
    matplotlib.rcParams['savefig.bbox'] = None
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    fig.subplots_adjust(left=.15, right=.95, bottom=.20, top=.84)
    ax.fill_between(times, low, high, color='#C6DCE8', label='第25至75百分位')
    ax.plot(times, median, 'o-', color='#155A7B', linewidth=2, label='整机法向载荷中位数')
    ax.axhline(threshold, color='#95612E', linestyle='--', linewidth=1, label='2%承重阈值')
    ax.set(xlim=(-1., 21.), xticks=times, ylabel='法向载荷占标称重量',
        xlabel='距该评分区间首个求解的时间（ms）')
    ax.set_ylim(bottom=-.035 * max(high.max(), threshold))
    ax.yaxis.set_major_formatter(PercentFormatter(1., decimals=0))
    ax.set_title(f'滚翻支撑失分场景的{len(cases)}个评分区间', loc='left', pad=10)
    ax.legend(loc='best')
    out = review_file.with_name('contact-load-window.png')
    assert not out.exists()
    fig.savefig(out, dpi=300, facecolor='white')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    ticks = set(ax.get_xticklabels() + ax.get_yticklabels())
    assert all(fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1) for _, box in texts)
    assert not [(a.get_text(), b.get_text()) for i, (a, x) in enumerate(texts)
                for b, y in texts[i + 1:] if x.overlaps(y)]
    assert not [label.get_text() for label, box in texts for spine in ax.spines.values()
                if spine.get_visible() and label not in ticks and box.overlaps(spine.get_window_extent(renderer))]
    plt.close(fig)
    write(out.with_suffix('.json'), dict(review_sha256=sha(review_file), figure_sha256=sha(out),
        role='insufficient_supported_rotation', worlds=cohort['worlds_with_gaps'], intervals=len(cases),
        solver_start_times_ms=times.tolist(), median_load_fraction=median.tolist(), lower_quartile=low.tolist(),
        upper_quartile=high.tolist(), threshold_fraction=threshold))
    print('NOMINAL_CONTACT_LOAD_WINDOW_RENDERED', sha(out), flush=True)


if __name__ == '__main__':
    main()
