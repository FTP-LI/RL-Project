"""Plot the audited world-level medians and quartiles around dual-foot support."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

from figure_style import apply_figure_style
from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha


OUT = ROOT/'records/reports/nominal-skill-composition/supine-early-support'
GROUPS = (('original_roll_admitted','进入滚翻','#40758a','o','-'),
          ('original_roll_missed','未获滚翻准入','#a83e27','s','--'))
METRICS = (('tilt_deg','机身倾角（°）'),('front_margin_mm_on_agreed_dual_foot','重心至前缘余量（mm）'))


def main():
    source = OUT/'analysis.json'
    state = read(ROOT/'project.json')['goal_execution']['nominal_supine_early_support']
    summary = read(source,state['analysis_sha256'])
    font = Path('/mnt/c/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    apply_figure_style(font=font_manager.FontProperties(fname=str(font)).get_name(),sizes=(11,10,9))
    fig, axes = plt.subplots(2,2,figsize=(8.8,5.4),sharex=True,sharey='row',layout='constrained')
    for column,case in enumerate(summary['cohorts']):
        axes[0,column].set_title(f'种子{case["seed"]}    通过{case["groups"][GROUPS[0][0]]["worlds"]}例，未准入{case["groups"][GROUPS[1][0]]["worlds"]}例')
        for row,(field,label) in enumerate(METRICS):
            ax = axes[row,column]
            for key,name,color,marker,style in GROUPS:
                timeline = case['groups'][key]['timeline']
                offsets = sorted(map(int,timeline))
                stats = [timeline[str(t)][field] for t in offsets]
                x = np.asarray(offsets)*20
                y = [np.nan if t['median'] is None else t['median'] for t in stats]
                lo,hi = ([np.nan if t['quartiles'] is None else t['quartiles'][i] for t in stats] for i in (0,1))
                ax.plot(x,y,label=name,color=color,marker=marker,linestyle=style,linewidth=1.7,markersize=4)
                ax.fill_between(x,lo,hi,color=color,alpha=.14,linewidth=0)
            ax.axvline(0,color='#888888',linewidth=.8,linestyle=':',zorder=0)
            if row:
                ax.axhline(0,color='#888888',linewidth=.8,zorder=0)
                ax.set_xlabel('相对双足接地确认时间（ms）')
            if not column:
                ax.set_ylabel(label)
            ax.margins(x=.04,y=.12)
    axes[0,0].legend(loc='center right')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.flat:
        bounds = ax.get_tightbbox(renderer)
        assert bounds.x0>=0 and bounds.y0>=0 and bounds.x1<=fig.bbox.width and bounds.y1<=fig.bbox.height
    output = OUT/'aligned-support.png'
    fig.savefig(output)
    plt.close(fig)
    write(OUT/'aligned-support-figure.json',dict(data_sha256=sha(source),figure_sha256=sha(output),
        source_script_sha256=sha(Path(__file__)),grouping='original roll admission by pose and seed',
        lines='world-level medians',bands='interquartile range',margin_inclusion='dual-foot at boundary with matching native and reconstructed contact'))
    print('SUPINE_EARLY_SUPPORT_FIGURE',sha(output),flush=True)


if __name__=='__main__':
    main()
