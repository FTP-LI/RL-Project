"""Descriptive multi-scale velocity audit, not a changed acceptance protocol.

Use all 32 original official-model episodes, fixed 2 s warmup and disjoint
0.02/0.2/1/2 s windows. No interpolation, fitted normalization, prediction or
training split. Retain surviving episodes explicitly; abort rather than omit
a failure. Environment 0 in a predefined 2-8 s plot is illustrative only.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
from s1_reference_inputs import ROOT,read,sha
from record_io import write

OUT=ROOT/'records/reports/command-response'


def analyze():
    state=read(ROOT/'records/reports/official-actor-bootstrap/screen-analysis.json')
    folder=ROOT/state['rows']['official']['forward']['record']
    checked=read(folder/'verification.json');report=read(folder/'evaluation.json')
    assert checked['status']=='passed' and checked['report_sha256']==sha(folder/'evaluation.json')
    results={};plot_data=None
    for condition in ('forward','backward'):
        item=next(x for x in report['conditions'] if x['condition']==condition)
        path=folder/f'{condition}-trace.npz'
        expected=next(x['trace_sha256'] for x in checked['checks'] if x['condition']==condition)
        assert sha(path)==expected
        with np.load(path) as trace:
            velocity=trace['velocity'].copy();active=trace['active'].copy()
        assert velocity.shape==(1000,32,2) and active.all() and np.isfinite(velocity).all()
        command=np.asarray(item['command']);steady=velocity[100:];scales={}
        for width in (1,10,50,100):
            mean=steady.reshape(900//width,width,32,2).mean(axis=1)
            error=np.linalg.norm(mean-command[:2],axis=2).mean(axis=0)
            scales[str(width*.02)]=dict(error_mean=float(error.mean()),error_per_episode=error.tolist(),
                blocks_per_episode=900//width,mean_velocity=mean.mean(axis=(0,1)).tolist())
        lateral=steady[:,:,1];centered=lateral-lateral.mean(axis=0)
        frequencies=np.fft.rfftfreq(900,.02);band=(frequencies>=.5)&(frequencies<=10)
        power=abs(np.fft.rfft(centered*np.hanning(900)[:,None],axis=0))**2
        peaks=frequencies[band][np.argmax(power[band],axis=0)]
        results[condition]=dict(episodes=32,all_survived=True,warmup_seconds=2,analysis_seconds=18,
            command=command.tolist(),trace=str(path.relative_to(ROOT)),trace_sha256=sha(path),scales=scales,
            signed_forward_axis_speed=float(steady[:,:,0].mean()*np.sign(command[0])),
            mean_lateral_velocity=float(lateral.mean()),lateral_std_mean=float(lateral.std(axis=0).mean()),
            dominant_lateral_frequency_hz=dict(median=float(np.median(peaks)),q25=float(np.quantile(peaks,.25)),
                q75=float(np.quantile(peaks,.75)),per_episode=peaks.tolist(),search_band=[.5,10.]))
        if condition=='forward':plot_data=(velocity,command)
    return results,plot_data,folder


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter
    from matplotlib.text import Text
    from figure_style import apply_figure_style
    font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei',sizes=(12,11,10))
    matplotlib.rcParams['savefig.bbox']=None
    velocity,command=data;times=np.arange(1,1001)*.02;take=(times>=2)&(times<=8)
    fig,ax=plt.subplots(figsize=(8,4.6),dpi=300)
    fig.subplots_adjust(left=.12,right=.97,bottom=.32,top=.86)
    colors=('#0072B2','#D55E00')
    for axis,name in enumerate(('前向','侧向')):
        ax.plot(times[take],velocity[take,0,axis],color=colors[axis],alpha=.42,lw=.8,label='瞬时'+name)
    means=velocity[100:].reshape(18,50,32,2).mean(axis=1)[:6,0]
    for axis,name in enumerate(('前向','侧向')):
        ax.stairs(means[:,axis],np.arange(2,9),baseline=None,color=colors[axis],lw=1.7,label='一秒均值'+name)
        ax.scatter(np.arange(2.5,8,1),means[:,axis],color=colors[axis],marker='s',s=14,zorder=4)
    ax.axhline(command[0],color='#595959',ls='--',lw=1.,label='前向指令')
    ax.set(title='速度时间尺度',xlabel='时间（s）',ylabel='身体坐标速度（m/s）')
    ax.set_xlim(1.8,8.2);ax.set_xticks(np.arange(2,9))
    low=np.floor(velocity[take,0].min()*10)/10;high=np.ceil(velocity[take,0].max()*10)/10
    ax.set_ylim(low,high);ax.set_yticks(np.arange(low,high+.001,.1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:'0.0' if abs(value)<1e-10 else f'{value:.1f}'))
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.28),ncol=3,handlelength=1.7,columnspacing=1.5)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    labels=[(t,t.get_window_extent(renderer)) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    hits=[(a.get_text(),b.get_text()) for i,(a,ba) in enumerate(labels) for b,bb in labels[i+1:] if ba.overlaps(bb)]
    ticks={ax:set(ax.get_xticklabels()+ax.get_yticklabels())}
    for text,box in labels:
        assert fig.bbox.contains(box.x0,box.y0) and fig.bbox.contains(box.x1,box.y1),text.get_text()
        for spine in ax.spines.values():
            if spine.get_visible() and text not in ticks[ax] and box.overlaps(spine.get_window_extent(renderer)):
                hits.append((text.get_text(),'spine'))
    assert not hits,hits
    png=ROOT/'docs/reports/assets/velocity-timescales.png';svg=png.with_suffix('.svg')
    fig.savefig(png,dpi=300,facecolor='white');fig.savefig(svg,facecolor='white')
    plt.close(fig)
    return dict(png=str(png.relative_to(ROOT)),png_sha256=sha(png),svg=str(svg.relative_to(ROOT)),svg_sha256=sha(svg),
        collision_scan='passed',plot_episode=0,plot_interval_seconds=[2,8],
        caption='One predefined official-model forward episode, not a 32-episode average. Full summaries use all 32 episodes from 2 to 20 seconds.')


def main():
    results,data,folder=analyze();figure=plot(data)
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),conditions=results,
        script_sha256=sha(Path(__file__)),style_sha256=sha(ROOT/'scripts/validation/figure_style.py'),
        report_sha256=sha(folder/'evaluation.json'),figure=figure,learning_updates=0,simulation_steps=0,
        problem=dict(task='Interpret periodic versus sustained tracking error',sample_seconds=.02,
            entity='32 first episodes per fixed command, one unchanged official policy',
            scales='Single control step; 0.2 s subcycle; 1 s and 2 s multi-cycle means',
            counterevidence='Persistent directional bias or drift remains after averaging; reduced error alone cannot prove gait quality',
            leakage='No predictor or fitted scaler; disjoint retrospective windows, no online controller claim'),
        existing_verdicts_unchanged=True,new_acceptance_rule=False,
        scope='Descriptive scale sensitivity and spectrum; periodic motion can still be undesirable; no proof of healthy gait or reward cause')
    write(OUT/'scale-review.json',result)
    print(json.dumps(dict(status='passed',conditions={k:dict(
        signed_speed=v['signed_forward_axis_speed'],lateral_std=v['lateral_std_mean'],frequency=v['dominant_lateral_frequency_hz']['median'],
        scale_errors={s:r['error_mean'] for s,r in v['scales'].items()}) for k,v in results.items()},figure=figure),indent=2))


if __name__=='__main__':main()
