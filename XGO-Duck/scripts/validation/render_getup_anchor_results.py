"""Saved-data figures and event-aligned native recovery frames, CPU only.

Do not pick cameras by outcome. Use the protocol's first supine camera for
all models, retain missing events, and bind extracted frames to the decoded
eleven-second original and its qpos hashes. Drift is a mechanism measure;
functional acceptance comes from all320first episodes and all pose gates.
"""
from pathlib import Path
import json,math
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

OUT=ROOT/'records/reports/getup-teacher-anchor-pair'
ASSETS=ROOT/'docs/reports/assets'
LABELS=dict(original='固定官方模型',control='原奖励训练',candidate='成功动作保持')
COLORS=dict(original='#5e656b',control='#657d8e',candidate='#2479a8')


def main():
    summary=read(OUT/'analysis.json');assert summary['status']=='passed-integrity'
    roles=[r for r in ('original','control','candidate') if (OUT/f'{r}-analysis.json').exists()]
    proofs={r:read(OUT/f'{r}-analysis.json') for r in roles}
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei',sizes=(11,10,9))
    matplotlib.rcParams['savefig.bbox']=None
    figures=[]
    fig,ax=plt.subplots(figsize=(7.3,4.9));fig.subplots_adjust(left=.14,right=.94,bottom=.22,top=.88)
    for x,role in enumerate(roles):
        result=proofs[role]['end_hold']['overall'];v=result['rate'];lo,hi=result['wilson95']
        ax.errorbar(x,v*100,yerr=np.asarray([[v-lo],[hi-v]])*100,fmt='o',color=COLORS[role],capsize=5,markersize=7)
        ax.text(x+.12,v*100,f'{result["timely_holds"]}/{result["episodes"]}',fontsize=10,va='center',color=COLORS[role])
    ax.axhline(95,color='#9b6b3c',linestyle='--',linewidth=1)
    ax.text(-.25,95.4,'总体95%门槛',fontsize=10,ha='left',va='bottom')
    ax.set(xticks=range(len(roles)),xticklabels=[LABELS[r] for r in roles],
        ylim=(84,99),yticks=[85,90,95],xlim=(-.4,len(roles)-.3),ylabel='最后三秒持稳比例（%）')
    ax.set_title('起身持稳',loc='left')
    figures.append(checked_save(fig,ASSETS/'getup-success-anchor-hold.png'))
    training={};curves={}
    fig,ax=plt.subplots(figsize=(7.3,4.9));fig.subplots_adjust(left=.14,right=.94,bottom=.30,top=.88)
    for role in ('control','candidate'):
        path=OUT/role/'training-verification.json'
        if not path.exists():continue
        proof=read(path);training[role]=proof;folder=ROOT/proof['record']
        rows=[json.loads(line) for line in (folder/'learning-metrics.jsonl').read_text().splitlines()]
        curves[role]=rows
        x=[r['completed_updates'] for r in rows];y=[r['raw_action_drift_rms'] for r in rows]
        ax.plot(x,y,color=COLORS[role],label=LABELS[role],linewidth=1.8)
        ax.scatter(x[-1],y[-1],color=COLORS[role],s=28)
    ax.axhline(.05,color='#9b6b3c',linestyle='--',linewidth=1,label='累计偏离停止线')
    ax.set(xlabel='完整PPO更新次数',ylabel='固定成功输入上的输出RMS偏离（rad）',
        ylim=(0,.06),xlim=(0,52),xticks=[0,10,20,30,40,50])
    ax.set_title('累计输出偏离',loc='left')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncols=3,fontsize=10,columnspacing=1.)
    figures.append(checked_save(fig,ASSETS/'getup-success-anchor-drift.png'))
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',20)
    small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',17)
    plate=Image.new('RGB',(1980,82+len(roles)*355),'white');draw=ImageDraw.Draw(plate)
    draw.text((12,8),'起身关键时刻',font=font,fill='#15334b')
    draw.text((12,38),'协议预定的仰卧相机；未按成功挑选；完整原片11秒，缺失事件保留',font=small,fill='#15334b')
    provenance=[];cases={};failures={};release_prefix={}
    for row,role in enumerate(roles):
        folder=ROOT/proofs[role]['record'];p=read(folder/'protocol.json');report=read(folder/'evaluation.json')
        condition=next(c for c in report['conditions'] if c['condition']=='supine');v=condition['video'];world=v['environment_index']
        assert world==p['video_worlds']['supine']
        video=folder/v['path'];assert sha(video)==v['sha256'] and v['complete_clip']
        case=next(c for c in read(folder/'coverage.json')['episodes'] if c['env_index']==world);cases[role]=case
        with np.load(folder/'coverage-trace.npz') as f:
            times=f['time'];all_height=f['height'];all_q=f['quaternion']
            all_speed=np.linalg.norm(f['angular_velocity'],axis=-1)
            all_feet=f['foot_contact'].all(-1);all_joints=f['joint_envelope_ok']
            all_contact=f['native_head_leg_contact'];release_prefix[role]=f['qpos'][:100].copy()
        height=all_height[:,world];q=all_q[:,world]
        vertical=1-2*(q[:,1]**2+q[:,2]**2)/np.sum(q*q,axis=-1)
        final=times>8+1e-8;c=p['criteria']
        all_tilt=np.arccos(np.clip(1-2*(all_q[:,:,1]**2+all_q[:,:,2]**2)/np.sum(all_q*all_q,axis=-1),-1,1))
        tests=dict(height=abs(all_height-c['nominal_height_m'])<=c['height_tolerance_m'],
            tilt=all_tilt<=np.deg2rad(c['tilt_max_deg']),angular_speed=all_speed<=c['angular_speed_max_rad_s'],
            both_feet=all_feet,joints=all_joints)
        failed=np.flatnonzero(~np.asarray(proofs[role]['end_hold']['stable_by_environment']))
        all_cases=read(folder/'coverage.json')['episodes']
        failures[role]=dict(failed=len(failed),env_indices=failed.tolist(),
            actual_release_counts={name:sum(all_cases[i]['release_pose']==name for i in failed)
                for name in ('prone','supine','left-side','right-side')},
            final_window_violation_worlds={name:int((~mask[final][:,failed]).any(0).sum()) for name,mask in tests.items()},
            final_window_head_leg_worlds=int(all_contact[final][:,failed].any(0).sum()),
            failure_median_tilt_deg=float(np.median(np.rad2deg(all_tilt[final][:,failed]))),
            rows=[dict(env_index=int(i),requested=all_cases[i]['requested_pose'],actual=all_cases[i]['release_pose'],
                median_tilt_deg=float(np.median(np.rad2deg(all_tilt[final,i]))),
                median_height=float(np.median(all_height[final,i])),
                final_head_leg_contact=bool(all_contact[final,i].any())) for i in failed])
        active=(times>=2)&(times<=8)
        def first(mask):
            ids=np.flatnonzero(active&mask)
            return float(times[ids[0]]) if len(ids) else None
        events=[('释放',2.),('达到站高',first(abs(height-.139158)<=.02)),
            ('躯干竖直',first(vertical>=np.cos(np.deg2rad(15)))),
            ('及时持稳起点',case['hold_start_seconds']),('动作截止',8.),('观察结束',11.)]
        wanted=[2*math.ceil(round((8. if t is None else t)/.02)/2) for _,t in events]
        frames={};reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);decoded=0
        for i,raw in enumerate(reader):
            if i*2 in wanted:frames[i*2]=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(480,640,3).copy())
            decoded+=1
        assert meta['fps']==25 and decoded==276
        y=82+row*355;draw.text((12,y),f'{LABELS[role]} · 环境{world} · '+('及时起身通过' if case['success'] else '未及时持稳'),font=font,fill=COLORS[role])
        for col,((event,actual),step) in enumerate(zip(events,wanted)):
            x=col*330+5
            text=f'{event} {actual:.2f}s' if actual is not None else event+'：未发生'
            draw.text((x,y+34),text,font=small,fill='#15334b')
            draw.text((x,y+57),f'原片帧 t={step*.02:.2f}s',font=small,fill='#15334b')
            plate.paste(frames[step].resize((320,240),Image.Resampling.LANCZOS),(x,y+83))
            index=int(step//2)
            provenance.append(dict(role=role,env_index=world,event=event,event_seconds=actual,
                sampled_control_step=step,frame_index=index,video=str(video.relative_to(ROOT)),
                video_sha256=sha(video),qpos_sha256=v['qpos_sha256_by_frame'][index]))
    image=ASSETS/'getup-success-anchor-sequence.jpg';plate.save(image,quality=94)
    control=training.get('control');candidate=training.get('candidate')
    matched=bool(control and candidate and control['rollout_updates']==candidate['rollout_updates'])
    rms_reduction=(1-candidate['successful_state_drift_rms']/control['successful_state_drift_rms']) if matched else None
    prefix={r:dict(exact_before_release=bool(np.array_equal(release_prefix[roles[0]],v)),
        maximum_absolute_qpos_difference=float(abs(release_prefix[roles[0]]-v).max())) for r,v in release_prefix.items()}
    write(OUT/'failure-review.json',dict(status='passed-saved-data-only',roles=failures,
        zero_action_first100_controls=prefix,classification_window_seconds=[8,11],
        criteria_unchanged=True,new_physics_steps=0,new_learning_steps=0,
        scope='Overlapping physical failure components; association is not an intervention mechanism'))
    write(OUT/'mechanism-review.json',dict(status='passed-saved-data-only',matched_actual_updates=matched,
        relative_success_replay_rms_reduction=rms_reduction,training=training,camera_cases=cases,
        functional_gates={r:proofs[r]['all_motion_gates_passed'] for r in roles},
        scope='Replay drift is a mechanism measure; one training seed and one development seed do not establish reliability',
        analysis_sha256=sha(OUT/'analysis.json'),new_physics_steps=0,new_learning_steps=0))
    write(OUT/'figure-review.json',dict(status='rendered-awaiting-visual-review',figures=figures,
        sequence=str(image.relative_to(ROOT)),sequence_sha256=sha(image),source_frames=provenance,
        analysis_sha256=sha(OUT/'analysis.json'),script_sha256=sha(Path(__file__)),
        native_original_video=True,physics_steps=0,representatives='Prospectively fixed first supine camera, not selected by outcome'))
    print('GETUP_ANCHOR_FIGURES_RENDERED',matched,rms_reduction,flush=True)


if __name__=='__main__':main()
