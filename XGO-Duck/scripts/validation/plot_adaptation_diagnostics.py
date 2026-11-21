"""Plot all first-episode tracking errors from a verified diagnostic batch.

Small dots are the 32 episodes of one policy/seed, not independent training
seeds. Large markers are equal-episode means. No smoothing, resampling or
selected successful episode is used. This script does not run simulation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path.home()/'XGO-Duck/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from figure_style import apply_figure_style


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    directory=args.batch.resolve()
    verified=json.loads((directory/'comparison.json').read_text())
    batch=json.loads((directory/'batch.json').read_text())
    assert verified['status']=='passed' and batch['mode']=='comparison'
    specs=[('reference_repeat','Reference','#555555','o'),
           ('runtime_home','Runtime home','#0072B2','s'),
           ('runtime_filter','Runtime filter','#D55E00','^'),
           ('actuator_kp6','Actuator P = 6','#009E73','D')]
    conditions=['stop','forward','backward','turn_left','turn_right']
    labels=['Stop','Forward','Backward','Left turn','Right turn']
    runs={r['variant']:root/r['path'] for r in batch['runs']}
    apply_figure_style(sizes=(11,10,9),font='DejaVu Sans')
    fig,axes=plt.subplots(2,1,figsize=(9.2,7.1),sharex=True)
    fig.subplots_adjust(left=.12,right=.97,bottom=.15,top=.86,hspace=.30)
    metrics=[('linear_xy_mae_m_s','Planar error','Error (m/s)',.08),
             ('yaw_mae_rad_s','Yaw error','Error (rad/s)',.20)]
    handles=[]; output=directory/'figures'; output.mkdir(exist_ok=True)
    for index,((ax,(key,title,ylabel,threshold))) in enumerate(zip(axes,metrics)):
        for offset,(variant,label,color,marker) in zip(np.linspace(-.27,.27,4),specs):
            report=json.loads((runs[variant]/'evaluation.json').read_text())
            means=[]
            for i,condition in enumerate(conditions):
                item=next(r for r in report['conditions'] if r['condition']==condition)
                values=np.asarray([r[key] for r in item['episodes']])
                np.testing.assert_allclose(values.mean(),item[key],atol=1e-7)
                ax.scatter(i+offset+np.linspace(-.035,.035,len(values)),values,
                           s=7,color=color,alpha=.25,linewidths=0,zorder=2)
                means.append(values.mean())
            handle=ax.scatter(np.arange(5)+offset,means,s=38,marker=marker,
                              facecolor='white',edgecolor=color,linewidth=1.6,label=label,zorder=4)
            if index==0: handles.append(handle)
        ax.axhline(threshold,color='#64748b',linestyle='--',linewidth=1,zorder=1)
        ax.set_title(title,pad=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y',alpha=.15)
        ax.margins(y=.15)
        ax.set_xlim(-.6,4.6)
    axes[-1].set_xticks(np.arange(5),labels)
    axes[-1].set_xlabel('Command condition')
    fig.legend(handles,[s[1] for s in specs],loc='upper center',bbox_to_anchor=(.54,.99),ncol=2,frameon=False)
    fig.text(.12,.045,'32 first episodes per command; seed 1001. Small: episodes; large: mean; dashed: threshold.',fontsize=9)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
           if t.get_visible() and t.get_text().strip()]
    collisions=[]
    for i,(a,ab) in enumerate(texts):
        assert fig.bbox.contains(ab.x0,ab.y0) and fig.bbox.contains(ab.x1,ab.y1),a.get_text()
        for b,bb in texts[i+1:]:
            if ab.overlaps(bb): collisions.append([a.get_text(),b.get_text()])
    assert not collisions,collisions
    figure=output/'adaptation-comparison.png'
    fig.savefig(figure,dpi=300,facecolor='white')
    check=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
               figure_sha256=hashlib.sha256(figure.read_bytes()).hexdigest(),
               comparison_sha256=hashlib.sha256((directory/'comparison.json').read_bytes()).hexdigest(),
               text_collisions=collisions,visual_inspection='pending')
    (output/'plot-check.json').write_text(json.dumps(check,indent=2)+'\n')
    print(figure)


if __name__=='__main__': main()
