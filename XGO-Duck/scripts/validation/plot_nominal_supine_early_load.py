"""Plot first foot contact, continuous support and solved loads by seed."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from figure_style import apply_figure_style
from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha


OUT = ROOT/'records/reports/nominal-skill-composition/early-getup-load'
GROUPS = (('admitted','进入滚翻','#326d87','-'), ('missed','停在起身准入','#a3472a','--'))


def main():
    node = read(ROOT/'project.json')['goal_execution']['nominal_supine_early_load']
    source = ROOT/node['analysis']
    result = read(source,node['analysis_sha256'])
    draft = read(ROOT/node['design'],node['design_sha256'])
    cjk = Path('/mnt/c/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(cjk))
    family = font_manager.FontProperties(fname=str(cjk)).get_name()
    apply_figure_style(font=family,sizes=(11,10,9))
    fig, axes = plt.subplots(2,2,figsize=(9.3,5.7),sharex=True,sharey='row',layout='constrained')
    for i, seed in enumerate(result['seed_results']):
        chronology = seed['chronology']
        time = np.asarray(chronology['time_ms'])
        events, loads = axes[:,i]
        good, bad = [seed['cohorts'][name]['worlds'] for name in ('admitted','missed')]
        events.set_title(f'种子{seed["seed"]}　准入{good}例　未准入{bad}例')
        events.step(time,chronology['first_touch_fraction'],where='post',color='#64696e',linestyle=':',label='首次足地接触')
        for key,name,color,style in GROUPS:
            curve = chronology['cohort_curves'].get(key)
            if curve is None:
                continue
            events.step(time,curve['sustained_bilateral_fraction'],where='post',color=color,linestyle=style,
                        linewidth=1.65,label='持续双足承重 · '+name)
            lower,median,upper = curve['foot_load_quartiles']
            loads.plot(time[1:],median,color=color,linestyle=style,linewidth=1.45)
            loads.fill_between(time[1:],lower,upper,color=color,alpha=.13,linewidth=0)
        loads.axhline(draft['bilateral_sum_load_fraction'],color='#777777',linestyle=':',linewidth=.8)
        for ax in (events,loads):
            ax.axvline((131-draft['release_control'])*draft['controller_period_s']*1000,color='#777777',
                       linestyle=(0,(4,4)),linewidth=.7,zorder=0)
            ax.margins(x=.025,y=.1)
        events.yaxis.set_major_formatter(PercentFormatter(1))
        loads.set_xlabel('释放后时间（ms）')
    axes[0,0].set_ylabel('已达到事件的场景比例')
    axes[1,0].set_ylabel('双足合计法向力（机重倍数）')
    axes[0,0].set_ylim(-.03,1.05)
    axes[0,1].legend(loc='lower right',fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.flat:
        box = ax.get_tightbbox(renderer)
        assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
    target = OUT/'support-load-chronology.png'
    fig.savefig(target,dpi=300)
    plt.close(fig)
    receipt = OUT/'support-load-figure.json'
    write(receipt,dict(analysis_sha256=sha(source),image_sha256=sha(target),script_sha256=sha(Path(__file__)),
                       source_seeds=[item['seed'] for item in result['seed_results']],
                       upper='First foot contact: all actual supine releases. Bilateral confirmation: original-admission groups.',
                       lower='Both foot normal loads in nominal body weights. Lines: medians. Bands: spatial cohort interquartile range.',
                       vertical_marker_control=131,new_physical_steps=0))
    print('SUPINE_EARLY_LOAD_FIGURE',sha(target),flush=True)


if __name__ == '__main__':
    main()
