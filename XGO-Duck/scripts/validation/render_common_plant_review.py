"""Render saved recovery evidence and atlas inspection crops, CPU only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from PIL import Image
from figure_style import apply_figure_style
from render_roll_angle_results import checked_save
from review_common_plant_screen import ROOT,OUT,read,sha,load
from standup_metrics import projected_gravity

def main():
    review=read(OUT/'failure-review.json');assert review['status']=='passed'
    record=read(OUT/'getup/common-verification.json')['record'];folder=ROOT/record
    t=load(folder/'supine-trace.npz');report=read(folder/'evaluation.json')
    index=report['joint_names'].index('neck_pitch')
    assert not read(OUT/'getup/analysis.json')['conditions'][1]['episodes'][1]['success']
    assert read(OUT/'getup/analysis.json')['conditions'][1]['episodes'][0]['success']
    tilt=np.rad2deg(np.arccos(np.clip(-projected_gravity(t['quaternion'])[...,2],-1,1)))
    neck=np.rad2deg(t['joint_positions'][:,:,index])
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei',sizes=(11,10,9))
    fig,axes=plt.subplots(2,1,sharex=True,figsize=(7.4,6.1))
    fig.subplots_adjust(left=.15,right=.95,bottom=.13,top=.90,hspace=.22)
    colors={0:'#2479a8',1:'#be771c'}
    for ax,data in zip(axes,(tilt,neck)):
        ax.axvspan(0,2,color='#e8ecef',zorder=-1)
        for i,label in ((0,'成功示例（环境 0）'),(1,'未持稳示例（环境 1）')):
            ax.plot(t['time'],data[:,i],color=colors[i],linestyle='-' if i==0 else '--',label=label)
        ax.set(xlim=(0,11.15),xticks=range(0,12,2))
    axes[0].axhline(15,color='#727c85',linestyle=':',linewidth=1,label='15° 姿态门槛')
    axes[0].set(ylabel='躯干倾角（°）',ylim=(-4,186),yticks=(0,45,90,135,180))
    axes[1].set(ylabel='颈部俯仰角（°）',xlabel='从 episode 开始计时（s）',ylim=(-155,46),yticks=(-150,-100,-50,0))
    axes[0].set_title('仰卧起身轨迹',loc='left');axes[0].legend(loc='upper right')
    path=ROOT/'docs/reports/assets/common-recovery-tilt-neck.png'
    result=checked_save(fig,path);crops=[]
    with Image.open(path) as im:
        boxes=[(0,0,im.width,round(im.height*.49)),(0,round(im.height*.47),im.width,im.height)]
        for i,box in enumerate(boxes):
            target=OUT/f'recovery-plot-panel-{i+1}.png'
            im.crop(box).save(target)
            crops.append(str(target.relative_to(ROOT)))
    atlas=read(ROOT/'records/reports/stage-media/all-video-overview.json');atlas_path=ROOT/atlas['image']
    assert sha(atlas_path)==atlas['image_sha256']
    with Image.open(atlas_path) as im:
        for label,box in (('header',(0,0,im.width,170)),('tail',(0,im.height-1460,im.width,im.height))):
            target=ROOT/'records/reports/stage-media'/f'common-overview-{label}-review.jpg'
            im.crop(box).save(target,quality=94);crops.append(str(target.relative_to(ROOT)))
    result.update(status='rendered-awaiting-visual-review',crops=crops,
        selected_episodes=[0,1],unit='Two individual trajectories from one32episode supine development stratum; grey0-2s is the freeze',
        source_trace_sha256=sha(folder/'supine-trace.npz'),physics_steps=0)
    (OUT/'figure-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
