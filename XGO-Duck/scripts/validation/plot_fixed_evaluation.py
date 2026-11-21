"""Plot measured command response from a verified fixed-evaluation trace.

Thin traces contain only first episodes. Bold means use the still-observed
first episodes at each time; survival counts in the adjacent report are needed
to interpret attrition. No smoothing or synthetic trajectories are added.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path.home()/'XGO-Duck/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
from figure_style import apply_figure_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    report = json.loads((root/'evaluation.json').read_text())
    verification = json.loads((root/'verification.json').read_text())
    assert report['status'] == 'completed' and verification['status'] == 'passed'
    output = root/'figures'
    output.mkdir(exist_ok=True)
    apply_figure_style(sizes=(11,10,9), font='DejaVu Sans')
    fig, axes = plt.subplots(2,1,figsize=(9,6.6),sharex=True)
    fig.subplots_adjust(left=0.12,right=0.97,bottom=0.16,top=0.94,hspace=0.30)
    pairs = [(('forward','Forward'),('backward','Backward')),
             (('turn_left','Left turn'),('turn_right','Right turn'))]
    counts = {}
    for row, (ax,pair) in enumerate(zip(axes,pairs)):
        for (condition,label),color in zip(pair,('#0072B2','#D55E00')):
            data = np.load(root/f'{condition}-trace.npz')
            mask = data['active']
            measured = data['velocity'][:,:,0] if row == 0 else data['yaw_rate']
            measured = np.where(mask, measured, np.nan)
            t = (np.arange(len(measured))+1)*report['step_dt']
            ax.plot(t, measured, color=color, alpha=0.13, linewidth=0.5)
            ax.plot(t, np.nanmean(measured,axis=1), color=color, linewidth=1.8,label=label)
            result = next(v for v in report['conditions'] if v['condition']==condition)
            target = result['command'][0 if row == 0 else 2]
            ax.axhline(target,color=color,linestyle='--',linewidth=1.1)
            counts[condition] = dict(initial=int(mask[0].sum()), last=int(mask[-1].sum()))
        ax.set_title('Linear response' if row == 0 else 'Turning response',pad=10)
        ax.set_ylabel('Body vx (m/s)' if row == 0 else 'Yaw rate (rad/s)')
        ax.legend(loc='upper right',ncol=2)
        ax.margins(x=0.015,y=0.15)
        ax.grid(axis='y',alpha=0.15)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
    axes[-1].set_xlabel('Simulation time (s)')
    axes[-1].set_xticks(np.arange(0,report['episode_seconds']+0.001,5))
    fig.text(0.12,0.05,f'{report["num_envs"]} first episodes per command; seed {report["seed"]}. Thin: episodes; bold: mean; dashed: command.',fontsize=10)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
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
                for text,tbox in texts:
                    if text not in own and tbox.overlaps(box): collisions.append([text.get_text(),'spine'])
    assert not collisions,collisions
    png=output/'command-response.png'
    # Override the style's tight-save default so the saved pixel origin agrees
    # with the figure geometry used by the bounds check and panel crops.
    with plt.rc_context({'savefig.bbox': None}):
        fig.savefig(png,dpi=300,bbox_inches=None,facecolor='white')
    # Save each panel with axes/labels for visual inspection, not merely its data rectangle.
    image=Image.open(png)
    # savefig temporarily changes DPI; refresh the screen-DPI renderer before
    # converting artist bounds to the saved image's pixel coordinates.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Crop from each axes' actual text extent. A fixed half-height split can
    # cut the lower panel title, hiding a defect during the visual QA pass.
    scale = image.width / fig.bbox.width
    for ax, name in zip(axes, ('linear-panel.png', 'turning-panel.png')):
        box = ax.get_tightbbox(renderer).expanded(1.02, 1.08)
        pixels = (max(0, int(box.x0*scale)), max(0, int(image.height-box.y1*scale)),
                  min(image.width, int(np.ceil(box.x1*scale))),
                  min(image.height, int(np.ceil(image.height-box.y0*scale))))
        image.crop(pixels).save(output/name)
    (output/'plot-check.json').write_text(json.dumps(dict(status='passed',
        text_collisions=collisions, counts=counts, source_verification=verification['run_id'],
        visual_inspection='pending'),indent=2)+'\n')
    print(png)


if __name__ == '__main__':
    main()
