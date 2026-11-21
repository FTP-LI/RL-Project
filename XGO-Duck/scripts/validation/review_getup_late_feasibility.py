"""Review the frozen late-feasibility study from saved data, CPU only.

Keep update traces separate from fixed final-model evaluations. Recompute
late joint-request excess over all evaluation worlds and preserve original
timely/end-hold scores. Temporal examples use the preselected supine camera
in each arm; one example does not represent all320 episodes. Decode original
frames only, without interpolation, simulation, learning or resampling.
"""
from pathlib import Path
import json
import numpy as np
import onnx
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from figure_style import apply_figure_style
from render_roll_angle_results import checked_save
from s1_reference_inputs import ROOT,read,sha
from record_io import write

OUT=ROOT/'records/reports/getup-late-feasibility'
ASSETS=ROOT/'docs/reports/assets'
LABELS={'original':'官方固定策略','control':'原奖励训练对照','candidate':'末段可达目标候选'}
COLORS={'original':'#808892','control':'#2479a8','candidate':'#be771c'}
POSES=('prone','supine','left-side','right-side')


def excess(x,limits):
    return np.maximum(np.maximum(limits[:,0]-x,x-limits[:,1]),0.)


def crop_panels(path,fig,folder):
    im=Image.open(path)
    # savefig changes renderer DPI temporarily; redraw before pixel mapping.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    sx,sy=im.width/fig.bbox.width,im.height/fig.bbox.height
    for i,ax in enumerate(fig.axes):
        b=ax.get_tightbbox(renderer)
        box=(max(0,int(b.x0*sx)-8),max(0,int(im.height-b.y1*sy)-8),
             min(im.width,int(b.x1*sx)+8),min(im.height,int(im.height-b.y0*sy)+8))
        im.crop(box).save(folder/f'{path.stem}-{i}.png')


def main():
    master=read(ROOT/'configs/experiments/getup_late_feasibility_pair.json')
    result=read(OUT/'analysis.json');assert result['status']=='passed'
    assert sha(ROOT/'configs/experiments/getup_late_feasibility_pair.json')==result['protocol_sha256']
    for name,digest in master['source_sha256'].items():assert sha(ROOT/name)==digest,name
    roles=[k for k in LABELS if k in result['roles']]
    limits=np.asarray(master['training_arms']['control']['joint_limits'],np.float32)
    meta={x.key:x.value for x in onnx.load(ROOT/master['arms']['original']['model']).metadata_props}
    names=meta['joint_names'].split(',');assert len(names)==14
    review=OUT/'figure-review';review.mkdir(exist_ok=True)
    sources={};evaluations={};traces={};training={}
    for role in roles:
        proof=result['roles'][role];folder=ROOT/proof['record']
        trace=folder/'coverage-trace.npz';assert sha(trace)==proof['trace_sha256']
        with np.load(trace) as f:
            t={k:f[k] for k in ('episode_ages','initial_qpos','initial_qvel','qpos','requested_targets','applied_targets','native_head_leg_contact')}
        np.testing.assert_array_equal(t['episode_ages'],np.broadcast_to(np.arange(1,551)[:,None],(550,320)))
        stable=np.asarray(proof['end_hold']['stable_by_environment'],bool);assert stable.shape==(320,)
        assert int(stable.sum())==proof['end_hold']['overall']['timely_holds']
        blocks={}
        for label,lo,hi in (('release-2to3s',100,150),('late-5to8s',250,400),('hold-8to11s',400,550)):
            groups={}
            for group,mask in (('all',np.ones(320,bool)),('end_stable',stable),('end_failed',~stable)):
                if not mask.any():groups[group]=dict(episodes=0);continue
                raw=excess(t['requested_targets'][lo:hi,mask],limits)
                applied=excess(t['applied_targets'][lo:hi,mask],limits)
                groups[group]=dict(episodes=int(mask.sum()),controls_per_episode=hi-lo,
                    raw_any_joint_outside_fraction=float((raw>1e-6).any(-1).mean()),
                    raw_squared_excess_mean=float((raw*raw).sum(-1).mean()),
                    applied_any_joint_outside_fraction=float((applied>1e-6).any(-1).mean()),
                    joint_mean_excess_rad={name:float(raw[...,i].mean()) for i,name in enumerate(names)},
                    head_leg_episodes=int(t['native_head_leg_contact'][lo:hi,mask].any(0).sum()))
            blocks[label]=groups
        q=t['qpos'][399:,~stable,3:7]
        tilt=np.degrees(np.arccos(np.clip(1-2*(q[...,1]**2+q[...,2]**2),-1,1)))
        evaluations[role]=dict(overall=proof['overall'],end_hold=proof['end_hold']['overall'],
            motion_gates_passed=proof['all_motion_gates_passed'],phases=blocks,
            failed_episodes=int((~stable).sum()),
            failed_end_height_m=None if stable.all() else float(np.median(t['qpos'][399:,~stable,2])),
            failed_end_tilt_deg=None if stable.all() else float(np.median(tilt)))
        traces[role]=t;sources[role]=sha(trace)
    for role in ('control','candidate'):
        if not (OUT/role/'training-verification.json').exists():continue
        proof=read(OUT/role/'training-verification.json');assert proof['status']=='passed'
        folder=ROOT/proof['record'];metrics=folder/'learning-metrics.jsonl'
        rows=[json.loads(x) for x in metrics.read_text().splitlines()]
        assert len(rows)==50 and [r['completed_updates'] for r in rows]==list(range(1,51))
        runtime=read(folder/'runtime.json');assert runtime['control_transitions']==1228800
        training[role]=dict(rows=rows,metrics_sha256=sha(metrics),proof_sha256=sha(OUT/role/'training-verification.json'),
            actual_transitions=runtime['control_transitions'],actor_optimizer_steps=proof['actor_optimizer_steps'],
            elapsed_seconds=rows[-1]['elapsed_seconds'])
    # These are matched scenario indices, not guaranteed bitwise trajectories.
    paired={}
    if 'candidate' in traces:
        a=np.asarray(result['roles']['control']['end_hold']['stable_by_environment'],bool)
        b=np.asarray(result['roles']['candidate']['end_hold']['stable_by_environment'],bool)
        control,candidate=traces['control'],traces['candidate']
        delta=abs(control['qpos'][:100]-candidate['qpos'][:100]).max((1,2))
        nonzero=np.flatnonzero(delta>0)
        paired=dict(candidate_wins=int((~a&b).sum()),candidate_losses=int((a&~b).sum()),
            initial_qpos_exact=bool(np.array_equal(control['initial_qpos'],candidate['initial_qpos'])),
            initial_qvel_exact=bool(np.array_equal(control['initial_qvel'],candidate['initial_qvel'])),
            freeze_first_nonidentical_control=None if not len(nonzero) else int(nonzero[0])+1,
            freeze_qpos_max_abs=float(delta.max()),
            scope='Matched scenario indices only; any pre-release dynamics divergence prevents bitwise trajectory claims')
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc');apply_figure_style(font='Microsoft YaHei',sizes=(11,10,9))
    # Preserve the canvas origin for reproducible pixel-space panel crops.
    matplotlib.rcParams['savefig.bbox']=None
    figures=[];count=len(roles)
    fig,axes=plt.subplots(1,2,figsize=(9,6.4),sharey=True)
    fig.subplots_adjust(left=.10,right=.97,top=.88,bottom=.25,wspace=.08)
    vals={k:result['roles'][k]['end_hold'] for k in roles}
    lows=[v[group][pose]['wilson95'][0]*100 for v in vals.values() for group in ('requested','actual') for pose in POSES]
    floor=max(0,10*np.floor((min(lows)-3)/10))
    for ax,group,title in zip(axes,('requested','actual'),('起始姿态','实际释放姿态')):
        for i,pose in enumerate(POSES):
            for j,role in enumerate(roles):
                row=vals[role][group][pose];y=(3-i)*count+count-1-j;x=row['rate']*100
                lo,hi=np.asarray(row['wilson95'])*100
                ax.errorbar(x,y,xerr=np.maximum([[x-lo],[hi-x]],0),fmt='o',capsize=3,color=COLORS[role],markersize=5)
                ax.text(103,y,f'{row["timely_holds"]}/{row["episodes"]}',va='center',color=COLORS[role],fontsize=10)
        ax.axvline(95,color='#8d969e',linestyle='--',linewidth=1)
        ax.set(title=title,xlim=(floor,114),ylim=(-.7,4*count-.3),xticks=np.arange(floor,101,10),xlabel='最后3秒连续持稳比例（%）')
        ax.set_yticks([(3-i)*count+(count-1)/2 for i in range(4)],['俯卧','仰卧','左侧倒','右侧倒'])
    handles=[Line2D([],[],marker='o',color=COLORS[k],linestyle='none',label=LABELS[k]) for k in roles]
    handles.append(Line2D([],[],linestyle='--',color='#8d969e',label='95% 功能门槛'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.015),ncol=2,frameon=False)
    path=ASSETS/'getup-late-feasibility-hold.png';figures.append(checked_save(fig,path));crop_panels(path,fig,review)
    fig,axes=plt.subplots(1,2,figsize=(9,4.6));fig.subplots_adjust(left=.10,right=.97,bottom=.19,top=.85,wspace=.32)
    for role,t in training.items():
        rows=t['rows'];x=[r['completed_updates'] for r in rows]
        axes[0].plot(x,[r['raw_action_drift_rms'] for r in rows],color=COLORS[role],label=LABELS[role])
        # Auxiliary magnitude is measured even in the zero-weight control.
        eligible=[r for r in rows if r['eligible_late_fraction']>0]
        axes[1].plot([r['completed_updates'] for r in eligible],[r['auxiliary_loss'] for r in eligible],
            'o-',color=COLORS[role],markersize=3,label=LABELS[role])
    axes[0].set(title='固定观测动作偏离',xlabel='完成更新次数',ylabel='动作差 RMS（rad）',xlim=(0,52),xticks=range(0,51,10),ylim=(0,None))
    axes[1].set(title='末段越界目标损失',xlabel='完成更新次数',ylabel='关节平方距离之和（rad²）',xlim=(0,52),xticks=range(0,51,10),ylim=(0,None))
    axes[0].legend(loc='upper left');axes[1].legend(loc='lower left')
    path=ASSETS/'getup-late-feasibility-learning.png';figures.append(checked_save(fig,path));crop_panels(path,fig,review)
    # Actual original-camera frames: preserve the whole temporal sequence.
    times=(0.,2.,3.,5.,8.,11.)
    phases=('初始跌落','冻结结束','主动控制1秒','末段开始','恢复截止','观察终态')
    plate=Image.new('RGB',(1920,64+count*340),'white');draw=ImageDraw.Draw(plate)
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',19)
    draw.text((16,10),'起身完整过程',font=font,fill='#15334b');frames=[]
    for row,role in enumerate(roles):
        proof=result['roles'][role];folder=ROOT/proof['record'];state=read(folder/'evaluation.json')
        cond=next(c for c in state['conditions'] if c['condition']=='supine');v=cond['video'];source=folder/v['path']
        assert sha(source)==v['sha256'];ft=np.asarray(v['frame_times_seconds'])
        ids=[int(np.argmin(abs(ft-t))) for t in times];assert np.allclose(ft[ids],times,atol=1e-8)
        stream=imageio_ffmpeg.read_frames(str(source),pix_fmt='rgb24');info=next(stream);n=0;selected={}
        for i,raw in enumerate(stream):
            if i in ids:selected[i]=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(info['size'][1],info['size'][0],3).copy())
            n+=1
        assert n==276 and info['fps']==25 and len(selected)==6
        y=64+row*340;world=v['environment_index'];success=proof['end_hold']['stable_by_environment'][world]
        draw.text((16,y),f'{LABELS[role]} · 环境{world} · 起始仰卧 · 末段持稳{"通过" if success else "失败"}',font=font,fill='#15334b')
        for col,i in enumerate(ids):
            plate.paste(selected[i].resize((312,234),Image.Resampling.LANCZOS),(col*320+4,y+36))
            draw.text((col*320+12,y+278),f't={ft[i]:.2f}s · 步{round(ft[i]/.02)}',font=font,fill='#15334b')
            draw.text((col*320+12,y+306),phases[col],font=font,fill='#15334b')
        frames.append(dict(role=role,source=str(source.relative_to(ROOT)),source_sha256=sha(source),
            world=world,frame_indices=ids,times=ft[ids].tolist(),frames_decoded=n,end_hold=bool(success)))
    plate_path=ASSETS/'getup-late-feasibility-timeline.jpg';plate.save(plate_path,quality=94)
    for i,role in enumerate(roles):plate.crop((0,64+i*340,1920,64+(i+1)*340)).save(review/f'timeline-{role}.jpg',quality=94)
    report=dict(status='rendered-awaiting-visual-review',evaluations=evaluations,training=training,paired=paired,
        figures=figures,temporal_plate=dict(path=str(plate_path.relative_to(ROOT)),sha256=sha(plate_path),frames=frames),
        trace_sha256=sources,result_sha256=sha(OUT/'analysis.json'),script_sha256=sha(Path(__file__)),
        new_physics_steps=0,new_optimizer_steps=0,generated_robot_frames=False,
        replication='320development first episodes per evaluated model, one training seed; training traces are changing-policy diagnostics')
    write(OUT/'mechanism-review.json',report)
    print('LATE_FEASIBILITY_REVIEW',json.dumps(dict(
        counts={k:v['overall']['timely_holds'] for k,v in evaluations.items()},
        late_squared_excess={k:v['phases']['late-5to8s']['all']['raw_squared_excess_mean'] for k,v in evaluations.items()},
        paired=paired,status=report['status']),ensure_ascii=False))


if __name__=='__main__':main()
