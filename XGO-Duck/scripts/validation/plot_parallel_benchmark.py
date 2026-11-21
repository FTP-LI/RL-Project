"""Plot verified scaling throughput and all prespecified update times.

Two ordered passes use one fixed seed; they are not independent gait trials.
Keep all 20 measured updates per process, including slow outliers.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics

os.environ.setdefault('MPLCONFIGDIR',str(Path.home()/'XGO-Duck/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from figure_style import apply_figure_style


def main(folder):
    record=folder/'verification.json'
    checked=json.loads(record.read_text())
    assert checked['status']=='passed'
    for name,expected in checked['input_sha256'].items():
        if name.endswith('iteration-timing.jsonl'):
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected
    apply_figure_style(sizes=(11,10,9),font='DejaVu Sans')
    fig,axes=plt.subplots(2,1,figsize=(9,7.7),sharex=True)
    fig.subplots_adjust(left=.13,right=.96,top=.94,bottom=.15,hspace=.33)
    all_times=[]
    all_throughputs=[]
    for pass_index,(color,label) in enumerate((('#0072B2','Forward order'),('#D55E00','Reverse order'))):
        rows=sorted((r for r in checked['trials'] if r['pass_index']==pass_index),key=lambda r:r['num_envs'])
        values=[r['transitions_per_second']/1000 for r in rows]
        all_throughputs.extend(values)
        axes[0].plot(range(3),values,'o-',color=color,label=label,lw=1.5,ms=5)
        for position,row in enumerate(rows):
            trial=folder/f'trial-{row["trial_index"]+1:02d}-n{row["num_envs"]}'
            times=[json.loads(line)['wall_seconds'] for line in (trial/'iteration-timing.jsonl').read_text().splitlines()][20:]
            assert len(times)==20
            all_times.extend(times)
            center=position+(-.09 if pass_index==0 else .09)
            xs=[center+(i%5-2)*.012 for i in range(len(times))]
            axes[1].scatter(xs,times,color=color,s=17,alpha=.6)
            axes[1].plot([center-.065,center+.065],[statistics.median(times)]*2,color=color,lw=2.5)
    axes[0].set_title('Sampling throughput',pad=10)
    axes[0].set_ylabel('Transitions/s (thousands)')
    upper=math.ceil(max(all_throughputs)/5)*5
    axes[0].set_ylim(0,upper*1.15)
    axes[0].set_yticks(range(0,upper+1,5))
    axes[0].legend(loc='upper left',ncol=2)
    axes[1].set_title('Update time',pad=10)
    axes[1].set_ylabel('Elapsed time (s)')
    upper=math.ceil(max(all_times))
    axes[1].set_ylim(0,upper*1.1)
    axes[1].set_yticks(range(0,upper+1,max(1,math.ceil(upper/5))))
    axes[1].set_xticks([0,1,2],['1024','2048','4096'])
    axes[1].set_xlim(-.3,2.3)
    axes[1].set_xlabel('Parallel environments')
    for ax in axes:
        ax.grid(axis='y',alpha=.16)
    fig.text(.13,.065,'One host and seed; 2 ordered runs per size, 20 measured updates per run.',fontsize=10)
    fig.text(.13,.035,'Lower panel: all update times and medians. Diagnostic weights excluded from gait selection.',fontsize=10)
    output=folder/'figures'
    output.mkdir(exist_ok=True)
    png=output/'parallel-throughput.png'
    with plt.rc_context({'savefig.bbox':None}):
        fig.savefig(png,dpi=300,bbox_inches=None,facecolor='white')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
           if t.get_visible() and t.get_text().strip()]
    collisions=[]
    for i,(a,box) in enumerate(texts):
        assert fig.bbox.contains(box.x0,box.y0) and fig.bbox.contains(box.x1,box.y1),a.get_text()
        collisions.extend([a.get_text(),b.get_text()] for b,other in texts[i+1:] if box.overlaps(other))
    for ax in axes:
        own=set(ax.get_xticklabels(which='both')+ax.get_yticklabels(which='both'))
        for spine in ax.spines.values():
            if spine.get_visible():
                collisions.extend([text.get_text(),'spine'] for text,box in texts if text not in own and box.overlaps(spine.get_window_extent(renderer)))
    assert not collisions,collisions
    image=Image.open(png)
    sx,sy=image.width/fig.bbox.width,image.height/fig.bbox.height
    for ax,name in zip(axes,('throughput','update-time')):
        box=ax.get_tightbbox(renderer)
        image.crop((max(0,int((box.x0-4)*sx)),max(0,int(image.height-(box.y1+4)*sy)),
                    min(image.width,math.ceil((box.x1+4)*sx)),min(image.height,math.ceil(image.height-box.y0*sy)))).save(output/(name+'.png'))
    for src in (Path(__file__),Path(__file__).with_name('figure_style.py')):
        shutil.copy2(src,output/src.name)
    result=dict(status='passed',visual_inspection='pending',collisions=collisions,
                plotted_runs=6,plotted_update_times=len(all_times),
                verification_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
                figure_sha256=hashlib.sha256(png.read_bytes()).hexdigest())
    (output/'plot-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    main(parser.parse_args().folder.resolve())
