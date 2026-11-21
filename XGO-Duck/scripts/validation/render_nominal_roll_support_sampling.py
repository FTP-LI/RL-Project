"""Plot medians from the same-trajectory roll contact sampling audit."""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

from figure_style import apply_figure_style
from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha


KEY = 'nominal_roll_support_sampling_audit'
LABELS = {
    'official-recording-1256': '官方记录种子1256',
    'official-recording-1257': '官方记录种子1257',
    'original-reference-first': '同快照官方参照一',
    'original-reference-second': '同快照官方参照二',
    'one-sided-descent-limit': '颈部目标下调候选',
}
PASS, SHORT = 'original-passed', 'original-supported-rotation-shortfall'


def main():
    state = read(ROOT / 'project.json')['goal_execution'][KEY]
    source = ROOT / state['evidence']
    data = read(source, state['evidence_sha256'])
    runs = data['force_aligned_runs']
    assert tuple(runs) == tuple(LABELS)
    font = 'C:/Windows/Fonts/msyh.ttc' if os.name == 'nt' else '/mnt/c/Windows/Fonts/msyh.ttc'
    fontManager.addfont(font)
    apply_figure_style(font='Microsoft YaHei', sizes=(11, 10, 9))
    matplotlib.rcParams['savefig.bbox'] = None
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10.6, 4.15))
    fig.subplots_adjust(left=.255, right=.965, top=.71, bottom=.18, wspace=.17)
    labels = []
    values = []
    for number, (name, contents) in enumerate(runs.items()):
        a, b = (contents['bands'][x] for x in (PASS, SHORT))
        labels.append(f'{LABELS[name]}\n{a["worlds"]}／{b["worlds"]}个场景')
        values.append((a, b))
    colors = ('#166583', '#AA5632')
    markers = ('o', 'D')
    fields = ('median_missed_positive_rotation_20ms_deg', 'median_physical_unloaded_positive_rotation_4ms_deg')
    titles = ('20ms评分边界漏计', '4ms物理步实际无载荷')
    for i, (ax, field) in enumerate(zip(axes, fields, strict=True)):
        for k, pair in enumerate(values):
            x, y = [row[field] for row in pair]
            ax.plot([x, y], [k, k], color='#B7C2C8', linewidth=1.4, zorder=2)
            for number, value in enumerate((x, y)):
                # A small vertical offset keeps coincident 4 ms medians visible.
                ax.scatter(value, k + (-.11, .11)[number], s=36, color=colors[number],
                    marker=markers[number], zorder=3, linewidths=.5, edgecolor='white')
        ax.set(xlim=(0., 40.), xticks=(0, 10, 20, 30, 40), ylim=(4.55, -.55),
            xlabel='正向转角中位数（°）', title=titles[i])
        ax.grid(axis='x', color='#E2E8EB', linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].tick_params(axis='y', length=0, pad=8)
    axes[1].tick_params(axis='y', length=0, labelleft=False)
    fig.text(.255, .94, '仰卧起身后滚翻的支撑取样', fontsize=12, color='#142F45')
    handles = [Line2D([], [], linestyle='', marker=m, markerfacecolor=c, markeredgecolor='white', color=c, label=label)
        for m, c, label in zip(markers, colors, ('原滚翻通过', '有效支撑转角不足'), strict=True)]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.244, .898), ncol=2)
    fig.text(.255, .05, '滚翻后0.54–0.92秒 · 左侧为通过／失分场景数', color='#486172', fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text = [(t.get_text(), t.get_window_extent(renderer)) for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip()]
    assert not [name for name, a in text if not (fig.bbox.contains(a.x0, a.y0) and fig.bbox.contains(a.x1, a.y1))]
    assert not [(a, b) for i, (a, ax) in enumerate(text) for b, bx in text[i + 1:] if ax.overlaps(bx)]
    target = source.with_name('support-sampling-comparison.png')
    fig.savefig(target, dpi=300, facecolor='white')
    plt.close(fig)
    write(target.with_suffix('.json'), dict(evidence_sha256=sha(source), figure_sha256=sha(target),
        window=data['window']['seconds_after_roll_entry'], rows=list(runs), columns=list(fields),
        cohort_names=[PASS, SHORT]))
    print('NOMINAL_SUPPORT_SAMPLING_FIGURE', sha(target), flush=True)


if __name__ == '__main__':
    main()
