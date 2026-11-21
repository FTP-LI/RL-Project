"""Plot verified development outcomes and actual all-world learning telemetry."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image
from figure_style import apply_figure_style
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from render_roll_angle_results import checked_save

OUT=ROOT/'records/reports/getup-late-contact'


def main():
    result=read(OUT/'analysis.json');assert result['status']=='passed'
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc');apply_figure_style(font='Microsoft YaHei',sizes=(11,10,9))
    colors={'original':'#808892','control':'#2479a8','candidate':'#be771c'}
    m=read(ROOT/'configs/experiments/getup_late_contact_pair.json');figures=[]
    evidence={k:read(OUT/f'eval-{k}/analysis.json') for k in ('original','control')}
    fig,axes=plt.subplots(1,2,figsize=(8,4.8),sharey=True)
    fig.subplots_adjust(left=.12,right=.96,bottom=.23,top=.87,wspace=.06)
    lows=[]
    for ax,pose,title in zip(axes,('prone','supine'),('俯卧','仰卧')):
        for x,key in enumerate(evidence):
            row=next(c for c in evidence[key]['conditions'] if c['condition']==pose)
            y=row['success_fraction']*100;lo,hi=np.array(row['wilson_95'])*100;lows.append(lo)
            ax.errorbar(x,y,yerr=np.array([[y-lo],[hi-y]]),fmt='o',capsize=4,color=colors[key],markersize=6)
            ax.text(x,hi+1.6,f'{row["successes"]}/32',ha='center',va='bottom',fontsize=10)
        ax.text(2,.12,'未评价',transform=ax.get_xaxis_transform(),ha='center',color='#606d78',fontsize=10)
        ax.axhline(95,color='#89939b',linestyle='--',linewidth=1)
        ax.set_title(title,loc='left');ax.set_xticks(range(3),['官方','原奖励','接触惩罚']);ax.set_xlim(-.45,2.45)
    axes[0].set_ylabel('起身并持稳比例（%）');axes[0].set_ylim(max(0,min(lows)-5),109);axes[0].set_yticks([80,85,90,95,100])
    fig.legend(handles=[Line2D([],[],color='#89939b',linestyle='--',label='95% 功能门槛')],loc='lower center',bbox_to_anchor=(.54,.015),frameon=False)
    target=ROOT/'docs/reports/assets/getup-contact-outcomes.png';figures.append(checked_save(fig,target))
    image=Image.open(target);review=OUT/'figure-review';review.mkdir(exist_ok=True)
    image.crop((0,0,image.width//2,image.height)).save(review/'prone.png')
    image.crop((image.width//2,0,image.width,image.height)).save(review/'supine.png')
    fig,learning_axes=plt.subplots(2,1,figsize=(8,7.4),sharex=True)
    fig.subplots_adjust(left=.13,right=.96,bottom=.22,top=.94,hspace=.36)
    minima=[]
    for ax,(arm,label) in zip(learning_axes,(('control','原奖励对照'),('candidate','接触惩罚候选'))):
        for a,b in ((0,2),(8,10),(16,18)):ax.axvspan(a,b,color='#e6eaed',zorder=0)
        p=m['arms'][arm];record=ROOT/'records/experiments'/p['run_id']
        with np.load(record/'training-trace.npz') as f:
            time=f['step']*.02;fraction=f['stable_fraction']*100
        ax.plot(time[::10],fraction[::10],color=colors[arm],label=label,linewidth=1.3)
        minima.append(p['watch_stable_min']*100)
        for a,b in ((7,8),(15,16),(23,24)):
            if a<time[-1]:ax.plot([a,min(b,time[-1])],[minima[-1]]*2,color='#89939b',linestyle='--',linewidth=1)
        if time[-1]<24:ax.axvspan(time[-1],24,facecolor='#f5f5f5',hatch='///',edgecolor='#b9c3c9',linewidth=0)
        ax.set(xlim=(-.4,24.4),ylim=(-3,104),xticks=[0,4,8,12,16,20,24],yticks=[0,25,50,75,100],ylabel='运动学站稳比例（%）')
        ax.set_title(label,loc='left')
    assert minima[0]==minima[1]
    learning_axes[-1].set_xlabel('累计控制时间（s）')
    handles=[Patch(facecolor='#e6eaed',label='两秒冻结阶段'),Line2D([],[],color='#89939b',linestyle='--',label='停止阈值（仅末段）'),
        Patch(facecolor='#f5f5f5',hatch='///',edgecolor='#b9c3c9',label='停止后无数据')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.015),ncol=2,frameon=False)
    target=ROOT/'docs/reports/assets/getup-contact-learning.png';figures.append(checked_save(fig,target))
    result=dict(status='rendered-awaiting-visual-review',figures=figures,
        outcomes_sha256=sha(OUT/'analysis.json'),script_sha256=sha(Path(__file__)),
        candidate_evaluation_missing=True,candidate_training_partial=True,unequal_training_budgets_explicit=True,
        evaluation_replication='One training seed,32fresh physical episodes per model per requested pose; Wilson95intervals, not trained-seed robustness',
        learning_replication='All1024worlds, actual qpos/qvel; height/tilt/angular-speed/joint bounds only, not feet/hold acceptance',
        actor_policy_frozen_seconds_per_episode=2,physics_steps_in_this_plot=0)
    write(OUT/'figure-analysis.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
