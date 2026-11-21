"""Render saved target-audit and fresh value-fit evidence; CPU only."""
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
from s1_reference_inputs import ROOT,read,sha
from record_io import write


def main():
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei',sizes=(11,10,9))
    audit=ROOT/'records/reports/getup-target-feasibility'
    data=read(audit/'analysis.json');assert data['status']=='reviewed-saved-data-only'
    phases=['release-2to3s','recovery-3to5s','late-5to8s','hold-8to11s']
    names=['左髋偏航','左髋侧摆','左髋俯仰','左膝','左踝','颈部俯仰','头部俯仰',
           '头部偏航','头部侧摆','右髋偏航','右髋侧摆','右髋俯仰','右膝','右踝']
    fig,axes=plt.subplots(1,2,figsize=(8.5,6.6),sharey=True)
    fig.subplots_adjust(left=.18,right=.84,top=.90,bottom=.14,wspace=.12)
    for ax,group,title in zip(axes,['end_stable','end_failed'],['末段站稳（291次）','末段失败（29次）']):
        rows=data['roles']['control'];values=np.array([[j['raw_requested_outside_fraction']*100 for j in rows[phase][group]['joints']] for phase in phases]).T
        assert values.shape==(14,4) and ((values>=0)&(values<=100)).all()
        im=ax.imshow(values,vmin=0,vmax=100,cmap='cividis',aspect='auto')
        ax.set(xticks=range(4),xticklabels=['2–3','3–5','5–8','8–11'],yticks=range(14),yticklabels=names,xlabel='时间（秒）')
        ax.set_title(title,loc='left')
        for y in range(14):
            for x in range(4):
                value=values[y,x]
                label='0' if value==0 else ('<1' if value<1 else f'{value:.0f}')
                ax.text(x,y,label,ha='center',va='center',fontsize=9,color='white' if value<45 else '#142431')
    cax=fig.add_axes([.88,.20,.025,.58]);bar=fig.colorbar(im,cax=cax);bar.set_label('请求越界比例（%）')
    target=ROOT/'docs/reports/assets/getup-target-feasibility.png';proof1=checked_save(fig,target)
    review=audit/'figure-review';review.mkdir(exist_ok=True)
    image=Image.open(target);image.crop((0,0,image.width//2,image.height)).save(review/'stable.png')
    image.crop((image.width//2,0,image.width,image.height)).save(review/'failed.png')
    value=ROOT/'records/reports/getup-interface-value/readback-recovery'
    admission=read(value/'analysis.json');assert admission['status']=='passed' and admission['value_gate_passed']
    record=ROOT/admission['record'];rows=read(record/'value-metrics.json');assert [r['update'] for r in rows]==[0,1,10,25,50]
    fig,axes=plt.subplots(1,2,figsize=(8.5,4.3));fig.subplots_adjust(left=.10,right=.97,top=.87,bottom=.26,wspace=.29)
    for ax,metric,ylabel,threshold in zip(axes,['explained_variance','normalized_mse'],['解释方差（越高越好）','归一化均方误差（越低越好）'],[.5,.25]):
        for key,label,color in [('train','训练探针','#2479a8'),('validation','开发验证','#bb7336')]:
            ax.plot([r['update'] for r in rows],[r[key][metric] for r in rows],marker='o',label=label,color=color)
        ax.axhline(threshold,color='#7c8992',linestyle='--',label='准入门槛')
        ax.set(xlim=(-1,52),ylim=(-.02,1.08),xticks=[0,10,25,50],xlabel='价值更新次数',ylabel=ylabel)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.55,.02),ncol=3,frameon=False)
    target=ROOT/'docs/reports/assets/getup-interface-value-fit.png';proof2=checked_save(fig,target)
    image=Image.open(target);image.crop((0,0,image.width//2,image.height)).save(review/'value-variance.png')
    image.crop((image.width//2,0,image.width,image.height)).save(review/'value-error.png')
    write(audit/'figure-analysis.json',dict(status='rendered-awaiting-visual-review',figures=[proof1,proof2],
        target_audit_sha256=sha(audit/'analysis.json'),value_metrics_sha256=sha(record/'value-metrics.json'),
        script_sha256=sha(Path(__file__)),physics_steps=0,optimizer_steps=0,generated_robot_images=False,
        interpretation='Target heatmap: per-joint fraction of saved controls by final outcome, descriptive not causal. Value fit: train-only normalization/scaling,64independent development worlds, no actor improvement.'))
    print('PREPARATION_FIGURES_RENDERED',str(target),flush=True)


if __name__=='__main__':main()
