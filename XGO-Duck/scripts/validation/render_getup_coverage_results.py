"""Verified release-coverage results and complete-action temporal examples.

The temporal plate uses declared supine representatives, not cherry-picked
successes. All frames come from the recorded eleven-second evaluation. Counts
are episode-level; one picture cannot certify a model or a training effect.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from PIL import Image,ImageDraw,ImageFont
from figure_style import apply_figure_style
from render_roll_angle_results import checked_save
from s1_reference_inputs import ROOT,read,sha
from record_io import write

OUT=ROOT/'records/reports/getup-release-coverage/fixed-evaluation'


def main(study='coverage',final_hold=False):
    out={'coverage':OUT,'neck-relief':ROOT/'records/reports/getup-neck-relief',
         'stand-handoff':ROOT/'records/reports/getup-stand-handoff/fixed-evaluation',
         'neck-home-order':ROOT/'records/reports/getup-neck-home-order',
         'early-guard':ROOT/'records/reports/getup-early-guard'}[study]
    prefix={'coverage':'getup-release-coverage','neck-relief':'getup-neck-relief-outcomes',
            'stand-handoff':'getup-stand-handoff-outcomes',
            'neck-home-order':'getup-neck-home-order-recovery',
            'early-guard':'getup-early-guard-recovery'}[study]
    source=out/'analysis.json';analysis=read(source);assert analysis['status']=='passed' 
    values=analysis['roles'];poses=('prone','supine','left-side','right-side')
    if final_hold:
        assert study in ('neck-home-order','early-guard')
        values={k:dict(v,**{key:item for key,item in v['end_hold'].items() if key!='scope'}) for k,v in values.items()}
        prefix='getup-'+study+'-end-hold'
    colors={'original':'#808892','control':'#2479a8'};labels={'original':'官方固定策略','control':'原奖励训练对照'}
    if study=='neck-relief':
        colors={'control':'#808892','candidate':'#2479a8'};labels={'control':'官方策略原样对照','candidate':'末段颈部解困'}
        labels={k:v for k,v in labels.items() if k in values}
    if study=='stand-handoff':
        colors={'control':'#808892','candidate':'#2479a8'}
        labels={k:v for k,v in {'control':'共同接口起身对照','candidate':'起身接力站立'}.items() if k in values}
    if study=='neck-home-order':
        colors={'control':'#808892','neck_home':'#2479a8','ordered_stand':'#bb763b'}
        labels={k:v for k,v in {'control':'共同接口起身','neck_home':'颈部归位起身','ordered_stand':'归位后站立'}.items() if k in values}
    if study=='early-guard':
        colors={'control':'#808892','candidate':'#2479a8'}
        labels={k:v for k,v in {'control':'官方原样对照','candidate':'早期持续卡住修正'}.items() if k in values}
    roles=list(labels)
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc');apply_figure_style(font='Microsoft YaHei',sizes=(11,10,9))
    if study=='early-guard':matplotlib.rcParams['savefig.bbox']=None
    count=len(roles)
    fig,axes=plt.subplots(1,2,figsize=(9,5.6+max(0,count-2)*.8),sharey=True)
    fig.subplots_adjust(left=.10,right=.97,top=.88,bottom=.25,wspace=.08)
    lows=[v[group][pose]['wilson95'][0]*100 for v in values.values() for group in ('requested','actual') for pose in poses if v[group][pose]['wilson95'] is not None]
    floor=max(0,10*np.floor((min(lows)-3)/10))
    for ax,group,title in zip(axes,('requested','actual'),('起始姿态','实际释放姿态')):
        for i,pose in enumerate(poses):
            for j,role in enumerate(roles):
                row=values[role][group][pose];y=(3-i)*count+count-1-j
                if row['rate'] is None:
                    ax.text(103.,y,'0/0',va='center',color=colors[role],fontsize=10);continue
                x=row['rate']*100;lo,hi=np.array(row['wilson95'])*100
                error=np.array([[x-lo],[hi-x]])
                assert (error>=-1e-10).all(),'Confidence interval does not contain estimate'
                ax.errorbar(x,y,xerr=np.maximum(error,0),fmt='o',capsize=3,color=colors[role],markersize=5)
                ax.text(103.,y,f'{row["timely_holds"]}/{row["episodes"]}',va='center',color=colors[role],fontsize=10)
        ax.axvline(95,color='#8d969e',linestyle='--',linewidth=1)
        ax.set(title=title,xlim=(floor,114),ylim=(-.7,4*count-.3),xticks=np.arange(floor,101,10))
        ax.set_yticks([(3-i)*count+(count-1)/2 for i in range(4)],['俯卧','仰卧','左侧倒','右侧倒'])
        ax.set_xlabel('最后3秒连续持稳比例（%）' if final_hold else '及时起身并持稳比例（%）')
    handles=[Line2D([],[],marker='o',color=colors[k],linestyle='none',label=v) for k,v in labels.items()]
    handles.append(Line2D([],[],linestyle='--',color='#8d969e',label='95% 功能门槛'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.015),ncol=2,frameon=False)
    target=ROOT/'docs/reports/assets'/f'{prefix}.png';proof=checked_save(fig,target)
    review=out/('end-hold-figure-review' if final_hold else 'figure-review');review.mkdir(exist_ok=True)
    im=Image.open(target)
    im.crop((0,0,im.width//2,im.height)).save(review/'requested.png')
    im.crop((im.width//2,0,im.width,im.height)).save(review/'actual.png')
    if final_hold:
        write(out/'end-hold-figure-analysis.json',dict(status='rendered-awaiting-visual-review',figure=proof,
            analysis_sha256=sha(source),script_sha256=sha(Path(__file__)),
            replication='320first episodes per arm; final8..11s all-sample hold, same group95percent gates',
            physics_steps_added=0,generated_robot_frames=False))
        print('GETUP_END_HOLD_FIGURE_RENDERED',str(target),flush=True);return
    # Each row is one preselected environment across a complete timeline.
    pose='left-side' if study=='neck-relief' else 'supine'
    steps=(0,100,150,250,400,550) if study=='coverage' else (0,100,250,324,400,550)
    phase=('初始跌落','冻结结束','主动控制1秒','主动控制3秒','恢复截止','观察结束') if study=='coverage' else ('初始跌落','冻结结束','末段开始','颈部响应','恢复截止','观察结束')
    if study in ('stand-handoff','neck-home-order'):
        steps=(0,100,250,300,400,550)
        phase=('初始跌落','冻结结束','切换前','策略响应','恢复截止','观察结束')
        if study=='neck-home-order':phase=('初始跌落','冻结结束','归位起点','目标响应','恢复截止','观察结束')
    if study=='early-guard':
        steps=(0,100,150,200,400,550)
        phase=('初始跌落','冻结结束','主动控制1秒','主动控制2秒','恢复截止','观察结束')
    row_stride=320 if study=='coverage' else 355
    plate=Image.new('RGB',(1980,100+len(labels)*row_stride),'white');draw=ImageDraw.Draw(plate)
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',22)
    small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    draw.text((12,8),'完整起身过程',font=font,fill='#15334b')
    draw.text((12,42),('固定预选仰卧环境' if pose=='supine' else '固定预选侧倒环境')+' · 0–11秒 · 实际采样画面',font=small,fill='#15334b')
    sources=[]
    for row,(role,label) in enumerate(labels.items()):
        record=ROOT/values[role]['record'];state=read(record/'evaluation.json')
        condition=next(c for c in state['conditions'] if c['condition']==pose);index=condition['video']['environment_index']
        frames={}
        if study!='coverage':
            import imageio_ffmpeg
            reader=imageio_ffmpeg.read_frames(str(record/condition['video']['path']),pix_fmt='rgb24');metadata=next(reader)
            assert metadata['fps']==25 and metadata['size']==(640,480)
            for frame_index,raw in enumerate(reader):
                if frame_index*2 in steps:frames[frame_index*2]=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(480,640,3).copy())
            assert frame_index+1==276 and len(frames)==len(steps)
            with np.load(record/'coverage-trace.npz') as f:
                if study=='neck-relief':correction=f['neck_correction'].copy()
                else:standing=f['phase_standing'].copy()
                if study in ('neck-home-order','early-guard'):head=f['neck_home_active'].copy()
        y=90+row*row_stride;draw.text((12,y),f'{label} · 环境{index}',font=font,fill=colors[role])
        for column,step in enumerate(steps):
            x=10+column*330
            if study=='coverage':
                path=record/'videos'/f'{pose}_t{step*.02:g}.jpg'
                with Image.open(path) as frame:plate.paste(frame.resize((320,240),Image.Resampling.LANCZOS),(x,y+65))
                provenance=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
            else:
                plate.paste(frames[step].resize((320,240),Image.Resampling.LANCZOS),(x,y+65))
                if study=='neck-relief':
                    amount=0. if step==0 else float(np.degrees(correction[step-1,index]))
                    note=f'目标修正 +{amount:.1f}°'
                else:
                    name='站立' if step and standing[step-1,index] else '起身'
                    if study=='neck-home-order' and step and head[step-1,index]:name+='＋颈归位'
                    note='执行：'+name
                    if study=='early-guard':note='解困规则：'+('已触发' if step and head[step-1,index] else '未触发')
                draw.text((x,y+310),note,font=small,fill=colors[role])
                provenance=dict(video=str((record/condition['video']['path']).relative_to(ROOT)),
                    video_sha256=condition['video']['sha256'],frame_index=step//2,
                    qpos_sha256=condition['video']['qpos_sha256_by_frame'][step//2])
            draw.text((x,y+36),f'{step*.02:g} s · {phase[column]}',font=small,fill='#15334b')
            sources.append(dict(role=role,environment_index=index,control_step=step,**provenance))
    timeline=ROOT/'docs/reports/assets'/({'coverage':'getup-coverage-sequence.jpg',
        'neck-relief':'getup-neck-relief-sequence.jpg','stand-handoff':'getup-stand-handoff-sequence.jpg',
        'neck-home-order':'getup-neck-home-order-sequence.jpg',
        'early-guard':'getup-early-guard-sequence.jpg'}[study]);plate.save(timeline,quality=94)
    write(out/'figure-analysis.json',dict(status='rendered-awaiting-visual-review',figure=proof,
        timeline=str(timeline.relative_to(ROOT)),timeline_sha256=sha(timeline),timeline_sources=sources,
        analysis_sha256=sha(source),script_sha256=sha(Path(__file__)),
        replication='320first episodes per model, one new development seed; Wilson95intervals',
        representative_selection='Previously frozen first requested-pose world; chosen group='+pose+'; not selected by success',
        physics_steps_added=0,generated_robot_frames=False))
    print('GETUP_COVERAGE_FIGURES_RENDERED',str(target),str(timeline),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',choices=('coverage','neck-relief','stand-handoff','neck-home-order','early-guard'),default='coverage')
    parser.add_argument('--final-hold',action='store_true');args=parser.parse_args();main(args.study,args.final_hold)
