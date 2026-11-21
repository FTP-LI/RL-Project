"""Render the short torque replay and the next training signal from verified data."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
from PIL import Image

from figure_style import apply_figure_style
from plot_forward_support_continuation import find_font
from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, read, sha, write


REPORT=ROOT/'records/reports/nominal-skill-composition/low-friction-early-transfer-authority'
BLUE='#27618E'
GRAY='#5D6871'


def check_layout(fig):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=[(item,item.get_window_extent(renderer)) for item in fig.findobj(Text)
           if item.get_visible() and item.get_text().strip()]
    assert all(fig.bbox.contains(box.x0,box.y0) and fig.bbox.contains(box.x1,box.y1) for _,box in texts)
    assert not [(a.get_text(),b.get_text()) for i,(a,x) in enumerate(texts) for b,y in texts[i+1:] if x.overlaps(y)]
    for ax in fig.axes:
        ticks=set(ax.get_xticklabels(which='both')+ax.get_yticklabels(which='both'))
        for spine in ax.spines.values():
            if spine.get_visible():
                border=spine.get_window_extent(renderer)
                assert not [t.get_text() for t,box in texts if t not in ticks and box.overlaps(border)]


def render(qa):
    facts=read(PROJECT)['goal_execution']
    keys=('nominal_low_friction_early_hip_roll_replay','nominal_low_friction_loaded_upright_progress',
          'nominal_low_friction_loaded_upright_device')
    nodes=[facts[key] for key in keys]
    replay,progress,device=[read(ROOT/n['result'],n['result_sha256']) for n in nodes]
    assert not replay['eligible_for_development_followup'] and progress['recommended_reward_window_ms']=='0-80'
    assert device['positive_pairs']==12 and device['positive_focus_pairs']==4 and device['eligible_for_live_sampler']
    trials=sorted((row for row in replay['results'].values() if row['is_focus']),key=lambda row:row['world'])
    assert len(trials)==4 and all(row['advance_ms'] is not None for row in trials)
    slices=sorted(zip(progress['pairs'],progress['comparisons']['0-80']['by_measure']['supported_upright_progress_deg']['differences'],
                      strict=True),key=lambda item:item[1])
    assert len(slices)==13 and sum(value>0 for _,value in slices)==12
    median=progress['comparisons']['0-80']['by_measure']['supported_upright_progress_deg']['paired_median']
    assert median==np.median([value for _,value in slices])

    font=find_font()
    assert font is not None
    font_manager.fontManager.addfont(str(font))
    apply_figure_style(font=font_manager.FontProperties(fname=str(font)).get_name(),sizes=(13,11,10))
    fig,(a,b)=plt.subplots(1,2,figsize=(10.5,4.65),width_ratios=(.9,1.2),layout='constrained')
    a.set_title('左髋横滚脉冲短回放',loc='left')
    a.set_xlabel('右脚持续承重提前量（ms）')
    a.set_ylabel('目标场景编号')
    a.set_yticks(np.arange(len(trials)),[str(row['world']) for row in trials])
    a.set_ylim(len(trials)-.45,-1.35)
    a.set_xlim(-.75,8.95)
    a.set_xticks([0,2,4,6,8])
    a.axvline(8,color=GRAY,linewidth=1.,linestyle=(0,(4,4)),zorder=1)
    for i,row in enumerate(trials):
        a.scatter(row['advance_ms'],i,facecolors=BLUE if row['reference_alignment'] else 'white',
                  edgecolors=BLUE,marker='o',s=48,zorder=2)
    a.text(8.1,2.45,'8ms门槛',rotation=90,color=GRAY,ha='left',va='center',fontsize=10)
    a.legend(handles=[Line2D([],[],marker='o',linestyle='',markerfacecolor='white',markeredgecolor=BLUE,
                            label='参照与归档时点相差超过20ms')],loc='upper left')

    b.set_title('右脚持续承重后的80ms',loc='left')
    b.set_xlabel('承重转正量配对差（°）')
    b.set_ylabel('失分场景编号')
    b.set_yticks(np.arange(len(slices)),[str(row['missed_id']) for row,_ in slices])
    b.set_ylim(len(slices)-.3,-2.4)
    values=[delta for _,delta in slices]
    b.set_xlim(min(-1.,min(values)-.5),max(6.6,max(values)+.65))
    b.set_xticks([-2,0,2,4,6])
    b.axvline(0.,color=GRAY,linewidth=.8,zorder=0)
    for i,(row,value) in enumerate(slices):
        focal=row['focus']
        shade=BLUE if focal else GRAY
        b.hlines(i,min(value,0.),max(value,0.),color=shade,linewidth=1.,alpha=.7,zorder=1)
        b.scatter(value,i,facecolors=shade if focal else 'white',edgecolors=shade,marker='D' if focal else 'o',s=28,zorder=2)
    b.text(.06,.95,f'13对场景的中位差{median:.2f}°',transform=b.transAxes,va='top',ha='left',fontsize=11)
    b.legend(handles=[Line2D([],[],marker='D',linestyle='',color=BLUE,label='4个目标左转场景')],loc='upper right')
    for ax,letter in zip((a,b),'ab',strict=True):
        ax.text(-.10,1.02,letter,transform=ax.transAxes,va='bottom',ha='left',fontweight='bold',fontsize=14)
    target=REPORT/'early-recovery-comparison.png'
    fig.savefig(target,dpi=300,bbox_inches=fig.bbox_inches)
    check_layout(fig)
    qa.mkdir(parents=True,exist_ok=True)
    with Image.open(target) as canvas:
        scale=300/fig.dpi
        for name,axis in zip('ab',(a,b),strict=True):
            box=axis.get_tightbbox(fig.canvas.get_renderer())
            region=(max(0,round(box.x0*scale)-9),max(0,canvas.height-round(box.y1*scale)-9),
                    min(canvas.width,round(box.x1*scale)+9),min(canvas.height,canvas.height-round(box.y0*scale)+9))
            panel=canvas.crop(region)
            panel.thumbnail((850,850))
            panel.save(qa/f'xgo-early-recovery-{name}.png')
    plt.close(fig)
    write(REPORT/'early-recovery-figure-source.json',dict(
        sources={n['result']:n['result_sha256'] for n in nodes},image_sha256=sha(target),script_sha256=sha(Path(__file__)),
        layout_collision_check=True,actual_replay_focus_scene_ids=[r['world'] for r in trials],matched_pairs=len(slices)))
    print('EARLY_RECOVERY_COMPARISON_FIGURE',sha(target),flush=True)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('qa_directory',type=Path)
    render(parser.parse_args().qa_directory)
