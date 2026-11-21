"""Plot saved profiling measurements, retaining all five captured updates.

The ordered pair measures instrumentation overhead descriptively. CPU scope
times include waiting; the lower panel shows non-nested entry scopes only.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics

os.environ.setdefault('MPLCONFIGDIR', str(Path.home()/'XGO-Duck/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from figure_style import apply_figure_style


def main(pair):
    analysis_path = pair/'analysis/analysis.json'
    analysis = json.loads(analysis_path.read_text())
    assert analysis['status'] == 'passed'
    assert hashlib.sha256((pair/'verification.json').read_bytes()).hexdigest() == analysis['verification_sha256']
    output = pair/'analysis'
    apply_figure_style(sizes=(11,10,9), font='DejaVu Sans')
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.3), gridspec_kw={'height_ratios':[1,2]})
    fig.subplots_adjust(left=.27, right=.96, top=.94, bottom=.13, hspace=.60)
    captured_values=[]
    for index, (mode, label, color) in enumerate((('reference','Ordinary run','#0072B2'),
                                                ('traced','Profiler run','#D55E00'))):
        selected = json.loads((pair/mode/'iteration-timing.json').read_text())[22:27]
        values = [r['wall_seconds'] for r in selected]
        captured_values.extend(values)
        axes[0].scatter(values, [index+(i-2)*.055 for i in range(5)], color=color, s=28)
        axes[0].plot([statistics.median(values)]*2,[index-.21,index+.21],color=color,lw=2)
    axes[0].set_yticks([0,1],['Ordinary run','Profiler run'])
    axes[0].set_ylim(1.45,-.45)
    upper=math.ceil(max(captured_values))
    axes[0].set_xlim(0,upper*1.05)
    axes[0].set_xticks(range(upper+1))
    axes[0].set_xlabel('Elapsed time per update (s)')
    axes[0].set_title('Captured updates', pad=12)
    labels = [('environment_step','Environment'), ('ppo_update','PPO update'),
              ('policy_act','Policy action'), ('nan_check','Finite-value check'),
              ('rollout_storage','Rollout storage'), ('step_logging','Step logging'),
              ('returns','Returns'), ('iteration_logging','Update logging')]
    lookup = {r['name']:r for r in analysis['scopes']}
    values = [lookup['xgo/'+key]['cpu_total_us']/1e6/5 for key,_ in labels]
    for index,value in enumerate(values):
        axes[1].plot([0,value],[index,index],color='#7aa4b6',lw=1)
    axes[1].scatter(values,range(len(values)),color='#0072B2',s=26,zorder=3)
    axes[1].set_yticks(range(len(values)),[label for _,label in labels])
    axes[1].set_ylim(len(values)-.5,-.5)
    upper=math.ceil(max(values))
    axes[1].set_xlim(-upper*.025,upper*1.05)
    axes[1].set_xticks(range(upper+1))
    axes[1].set_xlabel('Inclusive CPU scope time per update (s)')
    axes[1].set_title('Host-side timing',pad=12)
    for ax in axes:
        ax.grid(axis='x',alpha=.15)
    fig.text(.07,.055,'5 updates per run (5022–5026); dots: measurements; short ticks: medians.',fontsize=10)
    fig.text(.07,.025,'One ordered pair. CPU scopes include waiting; no controlled speedup claim.',fontsize=10)
    png=output/'runtime-profile.png'
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
                collisions.extend([t.get_text(),'spine'] for t,b in texts
                                  if t not in own and b.overlaps(spine.get_window_extent(renderer)))
    assert not collisions,collisions
    image=Image.open(png)
    sx,sy=image.width/fig.bbox.width,image.height/fig.bbox.height
    for ax,name in zip(axes,('updates','cpu-scopes')):
        box=ax.get_tightbbox(renderer)
        # Proportional padding on a tall panel can include a clipped footnote.
        # Keep the lower crop at its own labels' boundary, above figure text.
        image.crop((max(0,int((box.x0-4)*sx)),max(0,int(image.height-(box.y1+4)*sy)),
                    min(image.width,math.ceil((box.x1+4)*sx)),
                    min(image.height,math.ceil(image.height-box.y0*sy)))).save(output/(name+'.png'))
    for src in (Path(__file__),Path(__file__).with_name('figure_style.py')):
        shutil.copy2(src,output/src.name)
    check=dict(status='passed',collisions=collisions,updates_per_mode=5,
               visual_inspection='pending',analysis_sha256=hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
               figure_sha256=hashlib.sha256(png.read_bytes()).hexdigest())
    (output/'plot-check.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps(check,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair',type=Path)
    main(parser.parse_args().pair.resolve())
