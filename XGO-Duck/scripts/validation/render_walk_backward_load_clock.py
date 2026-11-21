"""Draw aligned sole loads directly from verified fixed-policy solver readings."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import fontManager
import matplotlib.pyplot as plt
import numpy as np

from figure_style import apply_figure_style
from s1_reference_inputs import ROOT, read, sha
from record_io import write
from render_roll_angle_results import checked_save


PLAN = ROOT/'configs/experiments/common_walk_backward_load_clock.json'
OUT = ROOT/'records/reports/common-walk-backward-load-clock'
ASSET = ROOT/'docs/reports/assets/backward-sole-load.png'


def main():
    spec = read(PLAN)
    seed, = spec['runs']
    proof = OUT/f'seed{seed}-load-analysis.json'
    result = read(proof)
    assert result['status']=='passed-load-clock-integrity' and sha(PLAN)==result['protocol_sha256']
    record = ROOT/'records/evaluations'/spec['runs'][seed]
    archive = record/'load-clock.npz'
    assert sha(archive)==result['archive_sha256']
    clock = read(record/'load-clock.json')
    design = spec['load_clock']
    offsets = np.arange(spec['matched_window_controls'][0],4)
    ms = np.rint(1000*(design['control_dt']*offsets-design['physics_dt'])).astype(int)
    groups = {'starters':[],'stalls':[]}
    with np.load(archive) as data:
        force = data['load_n'][...,2:]
        for row in result['rows']:
            idx = row['contact_control']+offsets-data['control'][0]
            assert idx.min()>=0 and idx.max()<len(force)
            side = ('left','right').index(row['stance'])
            groups['starters'].append(force[idx,row['world']][:,[side,1-side]])
            for stall in row['stalls']:
                groups['stalls'].append(force[idx,stall['world']][:,[side,1-side]])
    assert len(groups['starters'])==result['eligible_late_starts'] and len(groups['stalls'])==result['stall_windows']
    labels = dict(starters=f'自然启动（{len(groups["starters"])} 个场景）',
                  stalls=f'对齐停滞（{len(groups["stalls"])} 个窗口）')
    bands = {k:100*np.quantile(v,[.25,.5,.75],axis=0)/clock['nominal_weight_n'] for k,v in groups.items() if v}
    assert bands
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei',sizes=(11,10,9))
    matplotlib.rcParams['savefig.bbox'] = None
    fig,axes = plt.subplots(1,2,figsize=(8.8,3.75),sharey=True)
    fig.subplots_adjust(left=.105,right=.975,top=.76,bottom=.21,wspace=.06)
    colors = dict(starters='#156181',stalls='#a26b32')
    styles = dict(starters='-',stalls='--')
    for i,(ax,title) in enumerate(zip(axes,('支撑脚','摆动脚'),strict=True)):
        for key,(lo,mid,hi) in bands.items():
            ax.fill_between(ms,lo[:,i],hi[:,i],color=colors[key],alpha=.14)
            ax.plot(ms,mid[:,i],color=colors[key],linestyle=styles[key],label=labels[key],linewidth=1.8)
        ax.axvline(0,color='#768592',linewidth=.9)
        ax.set(xticks=[-160,-120,-80,-40,0,40],xlim=(-171,63))
        ax.set_title(title,loc='left')
    top = max(float(band.max()) for band in bands.values())
    axes[0].set(ylim=(-max(2.,.03*top),1.07*top),ylabel='法向力（机体重量 %）')
    fig.supxlabel('相对单脚支撑时刻（ms）',y=.055)
    handles,legend = axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.52,.97),ncols=len(legend))
    figure = checked_save(fig,ASSET)
    write(OUT/'load-figure.json',dict(status='rendered-saved-loads',source_sha256=sha(proof),script_sha256=sha(Path(__file__)),
          svg_sha256=sha(ASSET.with_suffix('.svg')),figure=figure,relative_ms=ms.tolist(),
          sample_counts={k:len(v) for k,v in groups.items()},bands={k:v.tolist() for k,v in bands.items()}))
    print('BACKWARD_LOAD_FIGURE',figure['sha256'],{k:len(v) for k,v in groups.items()},flush=True)


if __name__=='__main__':
    main()
