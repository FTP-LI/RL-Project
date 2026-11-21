"""Plot the fixed-actor value probe; no extra rollout or optimizer step."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from PIL import Image
from figure_style import apply_figure_style
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from render_roll_angle_results import checked_save


def main():
    out=ROOT/'records/reports/getup-value-bootstrap/startup-recovery'
    analysis=read(out/'analysis.json');assert analysis['status']=='passed'
    record=ROOT/analysis['record'];rows=read(record/'value-metrics.json')
    assert [r['update'] for r in rows]==[0,1,10,25,50]
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei',sizes=(11,10,9))
    fig,axes=plt.subplots(2,1,figsize=(7.2,7.6));fig.subplots_adjust(left=.15,right=.94,bottom=.10,top=.93,hspace=.48)
    colors={'train':'#2479a8','validation':'#be771c'};labels={'train':'训练探针','validation':'验证环境'}
    for key in colors:
        x=[r['update'] for r in rows]
        axes[0].plot(x,[r[key]['explained_variance'] for r in rows],'o-',color=colors[key],label=labels[key])
        axes[1].plot(x,[r[key]['normalized_mse'] for r in rows],'o-',color=colors[key])
    axes[0].axhline(.5,color='#89939b',linestyle='--',linewidth=1)
    axes[1].axhline(.25,color='#89939b',linestyle='--',linewidth=1)
    axes[0].set_title('价值拟合',loc='left');axes[0].set_ylabel('解释方差（越高越好）')
    axes[0].legend(loc='lower right',frameon=False)
    axes[1].set_ylabel('缩放后均方误差（越低越好）');axes[1].set_xlabel('critic 梯度更新次数')
    for ax in axes:ax.set_xlim(-1.5,52);ax.set_xticks([0,10,25,50]);ax.margins(y=.08)
    image=ROOT/'docs/reports/assets/getup-value-fit.png';info=checked_save(fig,image)
    im=Image.open(image);folder=out/'figure-review';folder.mkdir(exist_ok=True)
    for name,box in (('upper',(0,0,im.width,im.height//2)),('lower',(0,im.height//2,im.width,im.height))):
        im.crop(box).save(folder/f'{name}.png')
    result=dict(status='rendered-awaiting-visual-review',figure=info,analysis_sha256=sha(out/'analysis.json'),
        actor_updates=0,train_metric_states=8192,validation_worlds=64,validation_states=25600,
        caption='One fixed-actor dataset,50critic-only minibatch steps. Training probe and validation worlds are disjoint. Dashed lines are frozen admission thresholds.',
        script_sha256=sha(Path(__file__)))
    write(out/'figure-analysis.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
