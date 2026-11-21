"""Plot the six verified throughput trials and actual observer state rates.

Two ordered passes are shown separately, without invented confidence bounds.
Display caps and the prespecified 8 Hz criterion are not measured frame rates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    root=p.parse_args().folder.resolve();verification=root/'verification.json'
    v=json.loads(verification.read_text());assert v['status']=='passed'
    assert hashlib.sha256((root/'batch.json').read_bytes()).hexdigest()==v['batch_sha256']
    os.environ.setdefault('MPLCONFIGDIR',str(Path.home()/'XGO-Duck/cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from figure_style import apply_figure_style
    apply_figure_style(sizes=(11,10,9),font='DejaVu Sans')
    names=['default4','gpu4','gpu15'];labels=['Default\n4 Hz cap','D3D12\n4 Hz cap','D3D12\n15 Hz cap']
    fig,axes=plt.subplots(1,2,figsize=(9,4.6))
    fig.subplots_adjust(left=.10,right=.98,bottom=.24,top=.78,wspace=.40)
    for repeat,color,marker,offset in [(0,'#0072B2','o',-.08),(1,'#D55E00','s',.08)]:
        rows={r['case']:r for r in v['trials'] if r['pass_index']==repeat}
        axes[0].scatter(np.arange(3)+offset,[rows[n]['transitions_per_second']/1000 for n in names],
            color=color,marker=marker,s=48,label=f'Pass {repeat+1}',zorder=3)
        axes[1].scatter(np.arange(3)+offset,[rows[n]['actual_state_samples_per_second'] for n in names],
            color=color,marker=marker,s=48,zorder=3)
    for i,cap in enumerate((4,4,15)):
        axes[1].plot([i-.2,i+.2],[cap,cap],color='#6B7280',linewidth=1.2)
    axes[1].axhline(8,color='#6B7280',linestyle='--',linewidth=.9)
    axes[0].set_title('Training throughput',pad=12)
    axes[1].set_title('State display rate',pad=12)
    axes[0].set_ylabel('Transitions/s (thousands)')
    axes[1].set_ylabel('New state samples/s')
    axes[0].set_ylim(0,max(r['transitions_per_second']/1000 for r in v['trials'])*1.2)
    axes[1].set_ylim(0,17)
    for ax in axes:
        ax.set_xticks(range(3),labels);ax.set_xlim(-.5,2.5);ax.grid(axis='y',alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.98),ncol=2)
    fig.text(.10,.075,'20 measured updates per trial; 4096 environments. All timing spikes retained.',fontsize=9)
    fig.text(.10,.025,'Grey marks: configured caps. Dashed: 8 Hz criterion. These are not measured display refresh rates.',fontsize=9)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    collisions=[]
    for i,(a,ab) in enumerate(texts):
        assert fig.bbox.contains(ab.x0,ab.y0) and fig.bbox.contains(ab.x1,ab.y1),a.get_text()
        for b,bb in texts[i+1:]:
            if ab.overlaps(bb):collisions.append([a.get_text(),b.get_text()])
    assert not collisions,collisions
    out=root/'figures';out.mkdir(exist_ok=True);png=out/'observer-performance.png'
    with plt.rc_context({'savefig.bbox':None}):fig.savefig(png,dpi=300,bbox_inches=None,facecolor='white')
    img=Image.open(png);fig.canvas.draw();renderer=fig.canvas.get_renderer();scale=img.width/fig.bbox.width
    for ax,name in zip(axes,['throughput-panel.png','state-rate-panel.png']):
        box=ax.get_tightbbox(renderer).expanded(1.04,1.08)
        img.crop((max(0,int(box.x0*scale)),max(0,int(img.height-box.y1*scale)),min(img.width,int(np.ceil(box.x1*scale))),min(img.height,int(np.ceil(img.height-box.y0*scale))))).save(out/name)
    plt.close(fig)
    check=dict(status='passed',source_verification_sha256=hashlib.sha256(verification.read_bytes()).hexdigest(),
        png_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),text_collisions=collisions,
        all_six_trials_shown=True,visual_inspection='pending')
    (out/'plot-check.json').write_text(json.dumps(check,indent=2)+'\n')
    print(png)


if __name__=='__main__':main()
