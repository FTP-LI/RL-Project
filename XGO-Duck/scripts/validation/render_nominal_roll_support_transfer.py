"""Render the first recorded head-to-head unload in the lowest numbered affected scene."""

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
    state = read(ROOT / 'project.json')['goal_execution']['nominal_roll_support_transfer']
    source = ROOT / state['support_transfer_comparison']
    comparison = read(source, state['support_transfer_comparison_sha256'])
    cases = comparison['groups']['1257']['rows']['supine_supported_angle_failure']['unloads']
    case = min((row for row in cases if row['before'] == row['after'] == 'head'),
        key=lambda row: (row['world'], row['start']))
    raw = (ROOT / state['collection']).parent
    clock = raw / 'contact-clock.npz'
    metadata = read(raw / 'contact-clock.json')
    assert sha(clock) == comparison['groups']['1257']['contact_clock_sha256'] == metadata['sha256']
    with np.load(clock, allow_pickle=False) as saved:
        take = (saved['substep'] >= case['start'] - 5) & (saved['substep'] <= case['end'] + 5)
        time_ms = (saved['substep'][take] - case['start']) * 4
        load = saved['normal_force_n'][take, case['world']] / metadata['nominal_weight_n']
    assert np.count_nonzero(time_ms == 0) == 1
    assert not (load[(time_ms >= 0) & (time_ms < case['gap_ms']), 0] > .02).any()
    assert (load[time_ms == case['gap_ms'], 1] > .02).all()

    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei', sizes=(12, 11, 10))
    matplotlib.rcParams['savefig.bbox'] = None
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    fig.subplots_adjust(left=.15, right=.95, bottom=.18, top=.71)
    ax.plot(time_ms, load[:, 1], 'o-', color='#155A7B', label='头部法向载荷')
    ax.plot(time_ms, load[:, 2:].sum(axis=1), 's--', color='#687682', label='两脚法向载荷合计')
    ax.axhline(.02, color='#94623F', linestyle=':', label='2%承重阈值')
    ax.axvspan(0, case['gap_ms'], color='#E9DCCB', alpha=.55, label='所选卸载时段')
    ceiling = np.ceil((max(load[:, 1].max(), load[:, 2:].sum(axis=1).max()) + .05) / .2) * .2
    ax.set(xlim=(time_ms.min() - 1.6, time_ms.max() + 1.6), xticks=np.unique([*time_ms[::2], 0, case['gap_ms']]),
        ylim=(-.05, ceiling + .015), yticks=np.arange(0, ceiling + .01, .2),
        xlabel='距首次卸载求解的时间（ms）', ylabel='法向载荷占标称重量')
    ax.yaxis.set_major_formatter(PercentFormatter(1., decimals=0))
    fig.text(.15, .92, f'场景{case["world"]}的头部再接触', fontsize=12)
    ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02), ncol=2, borderaxespad=0.)
    out = source.with_name('head-recontact-example.png')
    receipt = out.with_suffix('.json')
    if receipt.exists():
        assert read(receipt)['comparison_sha256'] == sha(source)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    ticks = set(ax.get_xticklabels() + ax.get_yticklabels())
    assert not [x.get_text() for x, a in texts
                if not (fig.bbox.contains(a.x0, a.y0) and fig.bbox.contains(a.x1, a.y1))]
    assert not [(x.get_text(), y.get_text()) for i, (x, a) in enumerate(texts) for y, b in texts[i + 1:] if a.overlaps(b)]
    assert not [x.get_text() for x, a in texts for spine in ax.spines.values()
                if spine.get_visible() and x not in ticks and a.overlaps(spine.get_window_extent(renderer))]
    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    write(receipt, dict(comparison_sha256=sha(source), contact_clock_sha256=sha(clock),
        figure_sha256=sha(out), scene=case['world'], exact_selection='Lowest world ID, then earliest head recontact',
        first_unloaded_substep=case['start'], recontact_substep=case['end'], unloading_ms=case['gap_ms'],
        relative_solve_times_ms=time_ms.tolist(), normal_load_fraction=load.tolist()))
    print('NOMINAL_HEAD_RECONTACT_FIGURE', sha(out), case['world'], case['gap_ms'])


if __name__ == '__main__':
    main()
