"""Plot saved training timings; this script never starts training.

The vertical bands are actual video windows, not fitted overhead estimates.
An event timestamp gap includes logging and saves approximately; the first
iteration has no preceding event and is omitted from the throughput panel.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path.home()/'XGO-Duck/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
import numpy as np
from figure_style import apply_figure_style


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis',type=Path)
    args=parser.parse_args()
    root=args.analysis.resolve()
    report=json.loads((root/'analysis.json').read_text())
    assert report['status']=='passed'
    with (root/'iterations.csv').open() as f: rows=list(csv.DictReader(f))
    x=np.array([int(r['iteration']) for r in rows])
    col=np.array([float(r['collection_seconds']) for r in rows])
    learn=np.array([float(r['learning_seconds']) for r in rows])
    gap=np.array([float(r['event_gap_seconds']) if r['event_gap_seconds'] else np.nan for r in rows])
    apply_figure_style(sizes=(11,10,9),font='DejaVu Sans')
    fig,axes=plt.subplots(2,1,figsize=(9,6.6),sharex=True)
    fig.subplots_adjust(left=.13,right=.97,top=.93,bottom=.16,hspace=.34)
    axes[0].plot(x,col,color='#0072B2',label='Collection',lw=1.2)
    axes[0].plot(x,learn,color='#D55E00',label='Learning',lw=1.2)
    axes[0].plot(x,gap,color='#343E4C',label='Event gap',lw=.8,ls='--')
    axes[0].set_title('Iteration timing',pad=10)
    axes[0].set_ylabel('Seconds / iteration')
    axes[0].legend(loc='upper right',ncol=3)
    axes[1].plot(x,report['transitions_per_iteration']/gap/1000,color='#0072B2',lw=1.2)
    axes[1].set_title('Observed throughput',pad=10)
    axes[1].set_ylabel('Thousand transitions / s')
    axes[1].set_xlabel('Completed iteration')
    for ax in axes:
        for begin in report['video']['trigger_steps']:
            start=begin/report['num_steps_per_env']-.5
            end=(begin+report['video']['length_frames'])/report['num_steps_per_env']-.5
            ax.axvspan(start,end,color='#E69F00',alpha=.14,zorder=0)
        ax.axvline(report['warmup_iterations']-.5,color='#8B9299',lw=.9,ls=':')
        ax.set_xlim(-2,max(x)+2)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y',alpha=.15)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper'))
    axes[0].set_ylim(top=max(np.nanmax(gap),col.max())*1.25)
    axes[1].set_xticks([0,50,100,150,200,250])
    fig.text(.13,.045,'Shaded: recorded video. Dotted: 20-iteration warm-up boundary. No smoothing.',fontsize=10)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
           if t.get_visible() and t.get_text().strip()]
    collisions=[]
    for i,(a,ab) in enumerate(texts):
        assert fig.bbox.contains(ab.x0,ab.y0) and fig.bbox.contains(ab.x1,ab.y1),a.get_text()
        for b,bb in texts[i+1:]:
            if ab.overlaps(bb): collisions.append([a.get_text(),b.get_text()])
    for ax in axes:
        own=set(ax.get_xticklabels(which='both')+ax.get_yticklabels(which='both'))
        for spine in ax.spines.values():
            if spine.get_visible():
                box=spine.get_window_extent(renderer)
                for t,tbox in texts:
                    if t not in own and tbox.overlaps(box): collisions.append([t.get_text(),'spine'])
    assert not collisions,collisions
    out=root/'training-efficiency.png'
    fig.savefig(out,dpi=300,bbox_inches=None,facecolor='white')
    # The two excerpts overlap so both retain their complete titles and axes.
    im=Image.open(out)
    im.crop((0,0,im.width,round(im.height*.48))).save(root/'timing-panel.png')
    im.crop((0,round(im.height*.44),im.width,im.height)).save(root/'throughput-panel.png')
    qa=dict(status='passed',text_collisions=collisions,visual_inspection='pending',
            figure_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
            data_sha256=hashlib.sha256((root/'iterations.csv').read_bytes()).hexdigest())
    (root/'plot-check.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(out)


if __name__=='__main__': main()
