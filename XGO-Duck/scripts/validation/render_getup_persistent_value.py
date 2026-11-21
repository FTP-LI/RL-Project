"""Saved fixed-policy value preparation plots and event-aligned native frames."""
from pathlib import Path
import math,sys
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/training'),str(ROOT/'scripts/validation')]
from s1_reference_inputs import read,sha
from record_io import write
from figure_style import apply_figure_style
from render_roll_angle_results import checked_save
OUT=ROOT/'records/reports/getup-persistent-contact'
ASSETS=ROOT/'docs/reports/assets'


def main():
    proof=read(OUT/'value/cpu-reciprocal-review/analysis.json');assert proof['both_value_gates_passed']
    record=ROOT/proof['record'];p=read(record/'protocol.json');state=read(record/'state.json')
    assert sha(record/'value-data.npz')==proof['dataset_sha256']
    with np.load(record/'value-data.npz') as f:
        contact=f['contact'].copy();ticks=f['ticks'].copy();potential=f['potential'].copy();qpos=f['qpos'][:,0].copy()
    time=np.arange(1,401)*.02
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc');apply_figure_style(font='Microsoft YaHei',sizes=(11,10,9))
    matplotlib.rcParams['savefig.bbox']=None;figures=[]
    fig,ax=plt.subplots(figsize=(7.4,4.8));fig.subplots_adjust(left=.13,right=.95,bottom=.29,top=.88)
    for values,label,color in [(contact,'头腿接触','#7b8794'),(ticks>=60,'连续至少1.2秒','#2479a8'),(potential>0,'符合惩罚条件','#be771c')]:
        ax.plot(time,values.sum(1),label=label,color=color,linewidth=1.6)
    ax.set(xlim=(0,8),ylim=(-5,max(100,int(contact.sum(1).max())+10)),xlabel='仿真时间（秒）',ylabel='环境数')
    ax.set_title('接触计时',loc='left');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncols=3,columnspacing=1.)
    figures.append(checked_save(fig,ASSETS/'getup-persistent-contact-count.png'))
    metrics=read(record/'value-metrics.json')
    fig,ax=plt.subplots(figsize=(7.4,4.8));fig.subplots_adjust(left=.13,right=.95,bottom=.29,top=.88)
    for key,label,color in [('train','训练样本','#7b8794'),('validation','开发验证','#2479a8')]:
        ax.plot([r['update'] for r in metrics],[r[key]['normalized_mse'] for r in metrics],'-o',label=label,color=color,markersize=4)
    ax.axhline(.25,color='#be771c',linestyle='--',linewidth=1,label='准入门槛')
    ax.set(xlim=(-1,51),ylim=(0,max(r[k]['normalized_mse'] for r in metrics for k in ('train','validation'))*1.12),
        xticks=[0,10,20,30,40,50],xlabel='价值网络小批量更新次数',ylabel='均值目标归一化MSE（越低越好）')
    ax.set_title('价值拟合',loc='left');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncols=3)
    figures.append(checked_save(fig,ASSETS/'getup-persistent-value-fit.png'))
    q=qpos[:,3:7];tilt=np.rad2deg(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2)/np.sum(q*q,-1),-1,1)))
    def first(mask):
        ids=np.flatnonzero((time>2+1e-8)&mask);return float(time[ids[0]]) if len(ids) else None
    events=[('起点',0.),('释放',2.),('首次头腿接触',first(contact[:,0])),('连续接触达1.2秒',first(ticks[:,0]>=60)),
        ('首次近站姿',first((abs(qpos[:,2]-p['criteria']['nominal_height_m'])<=p['criteria']['height_tolerance_m'])&(tilt<=p['criteria']['tilt_max_deg']))),('终点',8.)]
    steps=[min(400,2*math.ceil(round((8. if value is None else value)/.02)/2)) for _,value in events]
    clip=state['clips'][0];video=record/clip['file'];reader=imageio_ffmpeg.read_frames(str(video));meta=next(reader);frames={};count=0
    for i,raw in enumerate(reader):
        if i*2 in steps:frames[i*2]=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(480,640,3).copy())
        count+=1
    assert count==201 and meta['fps']==25 and sha(video)==proof['video_sha256']
    plate=Image.new('RGB',(1980,430),'white');draw=ImageDraw.Draw(plate)
    title=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',24);font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    draw.text((12,8),'起身采样关键时刻',font=title,fill='#15334b')
    draw.text((12,45),'固定相机环境0；官方策略加固定探索；完整8秒；未按结果选择相机，缺失事件明确保留',font=font,fill='#15334b')
    states=read(record/'recording-states.json');sources=[]
    for i,((name,value),step) in enumerate(zip(events,steps)):
        x=i*330+5;draw.text((x,84),name+('：未发生' if value is None else f' {value:.2f}s'),font=font,fill='#15334b')
        draw.text((x,115),f'原片帧 {step*.02:.2f}s',font=font,fill='#15334b')
        plate.paste(frames[step].resize((320,240),Image.Resampling.LANCZOS),(x,150))
        sources.append(dict(event=name,event_seconds=value,control_step=step,frame_index=step//2,
            qpos_sha256=states[step//2]['qpos_sha256']))
    target=ASSETS/'getup-persistent-value-sequence.jpg';plate.save(target,quality=94)
    write(OUT/'figure-review.json',dict(status='rendered-awaiting-visual-review',figures=figures,
        sequence=str(target.relative_to(ROOT)),sequence_sha256=sha(target),frames=sources,
        video=str(video.relative_to(ROOT)),video_sha256=sha(video),video_frames=201,
        camera_selected_by_outcome=False,analysis_sha256=sha(OUT/'value/cpu-reciprocal-review/analysis.json'),
        script_sha256=sha(Path(__file__)),physics_steps=0,learning_updates=0))
    print('PERSISTENT_VALUE_FIGURES',potential.sum(),events,flush=True)


if __name__=='__main__':main()
