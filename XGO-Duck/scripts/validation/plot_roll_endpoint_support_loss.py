"""Plot the verified paired airborne-angle differences by roll phase."""

import argparse
import importlib.util
from pathlib import Path
import warnings

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import font_manager, pyplot as plt
import numpy as np
from PIL import Image

from locate_roll_endpoint_support_loss import OUT, read, sha
from plot_forward_support_continuation import find_font


PHASES = ('0-to-90', '90-to-180', '180-to-270', '270-to-330')
GROUPS = (
    ('first-original-reward-reference', '候选减第一次参照', '#176884', 'o', -.24),
    ('second-original-reward-reference', '候选减第二次参照', '#5A557F', '^', .0),
    ('reference-repeat', '第二次参照减第一次参照', '#666970', 's', .24),
)


def inspect_layout(fig, target):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [(item, item.get_window_extent(renderer)) for item in fig.findobj(mpl.text.Text)
              if item.get_text().strip() and item.get_visible()]
    spines = [(item, item.get_window_extent(renderer)) for ax in fig.axes for item in ax.spines.values() if item.get_visible()]
    tick_labels = {ax: set(ax.get_xticklabels(which='both')+ax.get_yticklabels(which='both')) for ax in fig.axes}
    collisions = [(a.get_text(), b.get_text()) for index,(a,box) in enumerate(labels) for b,other in labels[index+1:]
                  if box.overlaps(other)]
    collisions.extend((text.get_text(), spine.spine_type) for text,box in labels for spine,edge in spines
                      if text not in tick_labels[spine.axes] and box.overlaps(edge))
    if collisions or any(not fig.bbox.contains(box.x0,box.y0) or not fig.bbox.contains(box.x1,box.y1) for _,box in labels):
        raise ValueError(f'Clipped or overlapping figure text: {target}: {collisions}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style-kernel',type=Path,required=True)
    parser.add_argument('--qa-dir',type=Path,required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('figure_style',args.style_kernel)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    font = find_font()
    if font is None:
        raise ValueError('Chinese font is required to render the phase labels')
    font_manager.fontManager.addfont(str(font))
    style.apply_figure_style(font=font_manager.FontProperties(fname=font).get_name(),sizes=(11,10,9))

    result = read(OUT)
    fig, axes = plt.subplots(1,2,sharey=True,figsize=(8.8,3.55))
    fig.subplots_adjust(left=.11,right=.98,top=.77,bottom=.24,wspace=.06)
    bounds, handles = [], []
    for (case,seed), axis in zip(result['seeds'].items(),axes,strict=True):
        focus = seed['cohorts']['matched_regressions']
        count = len(focus['worlds'])
        axis.set_title(f'种子{seed["seed"]}（{count}个配对场景）')
        axis.set_xlim(-.5,3.5)
        axis.set_xticks(range(4),('0–90°','90–180°','180–270°','270–330°'))
        axis.set_xlabel('几何前翻角区间')
        axis.yaxis.grid(True,color='#E1E6EC',linewidth=.6)
        axis.set_axisbelow(True)
        axis.axhline(0,color='#8D9299',linewidth=.7)
        for name,label,color,marker,offset in GROUPS:
            source = (focus['second_minus_first_reference'] if name=='reference-repeat'
                      else focus['candidate_minus_each_reference'][name])
            rows = [source['first_330']['bands'][phase]['air_loss_deg'] for phase in PHASES]
            if any(row['count']!=count for row in rows):
                raise ValueError(f'Plot scope changed for {case}/{name}')
            low, median, high = (np.array([row[key] for row in rows]) for key in ('q25','median','q75'))
            mark = axis.errorbar(np.arange(4)+offset,median,yerr=(median-low, high-median),fmt=marker,
                                label=label,color=color,elinewidth=1.35,capsize=2.2,markersize=4.4)
            if axis is axes[0]:
                handles.append(mark)
            bounds.extend((float(low.min()),float(high.max())))
    axes[0].set_ylabel('端点离地扣除角度差（°）')
    axes[0].set_ylim(min(bounds)-3,max(bounds)+3)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.97),ncol=3)
    fig.text(.54,.075,'圆点、三角和方点为场景配对中位数，误差线为第25–75百分位',ha='center',fontsize=10,color='#425265')
    target = OUT.with_name('support-loss-phases.png')
    with warnings.catch_warnings():
        warnings.filterwarnings('error',message='.*Glyph.*missing from font.*')
        inspect_layout(fig,target)
        fig.savefig(target)
        args.qa_dir.mkdir(parents=True,exist_ok=True)
        with Image.open(target) as image:
            for seed,box in zip(result['seeds'],style.panel_crops(fig).values(),strict=True):
                image.crop(box).save(args.qa_dir/f'{seed}.png')
    print('ROLL_SUPPORT_PHASE_FIGURE',sha(OUT),sha(target),flush=True)


if __name__ == '__main__':
    main()
