"""Plot the aligned force, tilt and neck effort measured after release."""

from argparse import ArgumentParser
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

from plot_forward_support_continuation import find_font
import review_nominal_low_friction_forward_support as review


def collisions(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [(t,t.get_window_extent(renderer)) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    spines = [(s,s.get_window_extent(renderer)) for ax in fig.axes for s in ax.spines.values() if s.get_visible()]
    own_ticks = {ax:set(ax.get_xticklabels()+ax.get_yticklabels()) for ax in fig.axes}
    hits = [(a.get_text(),b.get_text()) for i,(a,ba) in enumerate(labels) for b,bb in labels[i+1:] if ba.overlaps(bb)]
    hits += [(t.get_text(),'spine') for t,bt in labels for s,bs in spines if bt.overlaps(bs) and t not in own_ticks[s.axes]]
    hits += [(t.get_text(),'figure edge') for t,box in labels if not
             (fig.bbox.contains(box.x0,box.y0) and fig.bbox.contains(box.x1,box.y1))]
    assert not hits,hits


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('--style-kernel',type=Path,required=True)
    args = cli.parse_args()
    kernel = spec_from_file_location('xgo_figure_style',args.style_kernel)
    style = module_from_spec(kernel)
    kernel.loader.exec_module(style)
    font = find_font()
    assert font is not None
    font_manager.fontManager.addfont(str(font))
    style.apply_figure_style(font=font_manager.FontProperties(fname=font).get_name(),sizes=(12,10,9))
    root,report = review.ROOT,review.REPORT
    project = review.read(review.experiment.PROJECT)
    stage = project['goal_execution'][review.experiment.KEY]
    score = review.read(root/stage['result'],stage['result_sha256'])
    plan = review.read(review.experiment.PLAN,stage['protocol_sha256'])
    role = review.ROLES[-1]
    receipt = score['roles'][role]
    path = root/receipt['source']
    assert review.sha(path)==receipt['source_sha256']
    meta = review.read(root/receipt['metadata'],receipt['metadata_sha256'])
    scope = plan['forward_support']
    worlds = scope['selected_worlds']
    with np.load(path,allow_pickle=False) as data:
        time=(data['substep']-5*review.START)*scope['physics_dt']
        part=(time>=.45)&(time<=1.16)
        angle=review.tilt(data['orientation_after'][part])
        force=data['normal_force_n'][part]/meta['nominal_weight_n']
        neck=data['applied_torque_nm'][part,:,meta['joint_order'].index('neck_pitch')]
    time = time[part]
    blue,gray,orange,tint = '#065a83','#6d747b','#a64b19','#cfe8ef'
    fig,axs = plt.subplots(3,2,figsize=(10.4,7.3),sharex='col',sharey='row')
    fig.subplots_adjust(left=.115,right=.965,top=.81,bottom=.10,wspace=.11,hspace=.42)
    fig.suptitle('低摩擦起身的承重与转正',y=.97)
    for col,world in enumerate(scope['focus_worlds']):
        match=scope['matched_archive_worlds'][str(world)]
        i,j=worlds.index(world),worlds.index(match)
        a,b,c=axs[:,col]
        a.set_title(f'场景{world}与配对场景{match}',pad=8)
        a.plot(time,angle[:,j],color=gray,ls='--',lw=1.4)
        a.plot(time,angle[:,i],color=blue,lw=1.7)
        a.axhline(15,color='#7b7b7b',ls=':',lw=.8)
        b.plot(time,force[:,i,2:].sum(axis=1),color=blue,lw=1.45)
        b.plot(time,force[:,i,1],color=orange,ls='-.',lw=1.25)
        c.plot(time,neck[:,j],color=gray,ls='--',lw=1.4)
        c.plot(time,neck[:,i],color=blue,lw=1.7)
        timing=score['target_readback'][str(world)]['groups'][role]
        left,right=(timing[key]*.001 for key in ('first_bilateral_without_head_load_ms','next_sustained_head_load_ms'))
        for axis in axs[:,col]:
            axis.axvspan(left,right,color=tint,zorder=0)
            axis.set_xlim(.45,1.16)
            axis.set_xticks([.5,.65,.8,.95,1.1])
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4,prune='both'))
        c.set_xlabel('释放后时间（秒）')
    axs[0,0].set_ylabel('机身倾角（°）')
    axs[1,0].set_ylabel('载荷（机重倍数）')
    axs[2,0].set_ylabel('颈部扭矩（N·m）')
    top = [Line2D([],[],color=blue,lw=1.7,label='目标场景／双足载荷'),
           Line2D([],[],color=gray,ls='--',label='配对场景'),
           Line2D([],[],color=orange,ls='-.',label='目标场景头部载荷'),
           Line2D([],[],color='#7b7b7b',ls=':',label='15°倾角限值'),
           Patch(color=tint,label='头部卸载确认至再承重')]
    fig.legend(handles=top,loc='upper center',bbox_to_anchor=(.54,.925),ncol=5,columnspacing=1.1)
    output=report/'forward-support-timing.png'
    fig.savefig(output,dpi=300,facecolor='white')
    collisions(fig)
    qa=root/'workspace/plot-qa/low-friction-forward-support'
    qa.mkdir(parents=True,exist_ok=True)
    with Image.open(output) as full:
        for group,box in style.panel_crops(fig).items():
            full.crop(box).save(qa/f'{group}.png')
    review.write(report/'figure-source.json',dict(path=review.rel(output),sha256=review.sha(output),
        archived_input=receipt['source'],archived_input_sha256=receipt['source_sha256'],role=role,
        worlds=scope['focus_worlds'],matched=scope['matched_archive_worlds'],time_window_seconds=[.45,1.16],
        text_collisions=0,new_physical_steps=0))
    plt.close(fig)
    print('FORWARD_SUPPORT_FIGURE',review.sha(output),flush=True)


if __name__=='__main__':
    main()
