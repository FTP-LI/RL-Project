"""Render independent official-roll results and actual event-aligned frames.

Each point is one fixed-model physical seed, not a training replicate.
Every camera is the prospectively fixed world0, even if it succeeds while
the population gate fails. Decode native9s media only; no new robot states.
Development failure categories guide later hypotheses; heldout stays closed.
"""
from pathlib import Path
import math
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from figure_style import apply_figure_style
from render_roll_angle_results import checked_save
from s1_reference_inputs import ROOT,read,sha
from record_io import write

OUT=ROOT/'records/reports/roll-common-confirmation/rng-recovery'
ASSETS=ROOT/'docs/reports/assets'


def main():
    source=OUT/'analysis.json';a=read(source);assert a['status']=='passed'
    assert a['decision']=='closed-functional-gate-failed' and not a['nominal_motion_confirmation_passed']
    roles=['development-a','development-b','heldout'];labels=['开发A\nseed1020','开发B\nseed1021','独立留出\nseed2020']
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc');apply_figure_style(font='Microsoft YaHei',sizes=(11,10,9))
    matplotlib.rcParams['savefig.bbox']=None
    fig,ax=plt.subplots(figsize=(7.3,4.9));fig.subplots_adjust(left=.14,right=.95,bottom=.23,top=.88)
    for x,role in enumerate(roles):
        p=a['roles'][role];v=p['successes']/p['episodes'];lo,hi=p['wilson95']
        ax.errorbar(x,v*100,yerr=np.asarray([[v-lo],[hi-v]])*100,fmt='o',color='#2479a8',capsize=5,markersize=7)
        ax.text(x+.12,v*100,f'{p["successes"]}/{p["episodes"]}',va='center',fontsize=10,color='#2479a8')
    ax.axhline(90,color='#89939b',linestyle='--',linewidth=1)
    ax.text(-.32,90.25,'90% 门槛',ha='left',va='bottom',fontsize=10)
    ax.set(xticks=range(3),xticklabels=labels,ylim=(78,100),xlim=(-.4,2.65),yticks=[80,85,90,95,100],ylabel='滚翻成功比例（%）')
    ax.set_title('官方滚翻独立确认',loc='left')
    figure=checked_save(fig,ASSETS/'roll-common-confirmation.png')
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',20);small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',17)
    plate=Image.new('RGB',(1980,1160),'white');draw=ImageDraw.Draw(plate)
    draw.text((12,8),'滚翻关键时刻',font=font,fill='#15334b')
    draw.text((12,38),'三组均为预定环境0的完整9秒原片；个案成功不代表总体通过',font=small,fill='#15334b')
    provenance=[];development=[]
    for row,role in enumerate(roles):
        proof=a['roles'][role];folder=ROOT/proof['evaluator'];report=read(folder/'evaluation.json')
        case=proof['rows'][0];assert case['env_index']==0 and case['success']
        v=report['video'];assert v['complete_horizon'] and v['frame_steps']==list(range(0,451,2))
        video=folder/v['path'];assert sha(video)==proof['video_sha256']==v['sha256']
        events=[('起点',0.),('头顶支撑',case['head_contact_seconds']),('有效旋转完成',case['completion_seconds']),
            ('持稳结束',case['hold_end_seconds']),('原截止',7.),('观察结束',9.)]
        wanted=[2*math.ceil(round(t/.02)/2) for _,t in events]
        frames={};reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader)
        for i,raw in enumerate(reader):
            if i*2 in wanted:frames[i*2]=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(480,640,3).copy())
        assert meta['fps']==25 and len(frames)==6
        y=82+row*355;title=labels[row].replace('\n',' · ')+' · 环境0 · 个案通过'
        draw.text((12,y),title,font=font,fill='#2479a8')
        for col,((event,actual),step) in enumerate(zip(events,wanted)):
            x=col*330+5
            draw.text((x,y+34),f'{event} {actual:.2f}s',font=small,fill='#15334b')
            draw.text((x,y+57),f'原片帧 t={step*.02:.2f}s',font=small,fill='#15334b')
            plate.paste(frames[step].resize((320,240),Image.Resampling.LANCZOS),(x,y+83))
            index=v['frame_steps'].index(step)
            provenance.append(dict(role=role,event=event,event_seconds=actual,sampled_control_step=step,
                video=str(video.relative_to(ROOT)),video_sha256=sha(video),frame_index=index,
                qpos_sha256=v['qpos_sha256_by_frame'][index]))
        if role!='heldout':
            failed=[x for x in proof['rows'] if not x['success']]
            development.append(dict(role=role,failed=len(failed),
                no_headtop_sequence=sum(x['head_contact_seconds'] is None for x in failed),
                incomplete_effective_rotation=sum(x['head_contact_seconds'] is not None and x['completion_seconds'] is None for x in failed),
                completed_but_failed_other_criteria=sum(x['completion_seconds'] is not None for x in failed),rows=failed))
    image=ASSETS/'roll-common-confirmation-sequence.jpg';plate.save(image,quality=94)
    write(OUT/'development-failure-review.json',dict(status='passed-saved-data-only',development=development,
        heldout_used_for_new_parameter_choice=False,new_physics_steps=0,new_learning_steps=0,analysis_sha256=sha(source)))
    write(OUT/'figure-review.json',dict(status='rendered-awaiting-visual-review',figure=figure,
        sequence=str(image.relative_to(ROOT)),sequence_sha256=sha(image),source_frames=provenance,
        analysis_sha256=sha(source),script_sha256=sha(Path(__file__)),physics_steps=0,
        native_original_video=True,representatives='Prospectively fixed environment0, not selected by outcome'))
    print('ROLL_CONFIRM_FIGURES_RENDERED',figure['path'],str(image.relative_to(ROOT)),flush=True)


if __name__=='__main__':main()
