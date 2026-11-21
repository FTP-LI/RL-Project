"""Read actual head commands beside measured joint motion in the short benchmark.

Use every saved environment-0 sample and bind it to the completed verifier.
The policy is changing and commands are random: these are descriptive traces,
not fixed-policy gait scores or a comparison of head repairs. Plot the first
GPU/4 Hz trial by declared case order, never by its apparent movement quality.
No simulation, learning or display-window access.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    batch=parser.parse_args().batch.resolve()
    verification=batch/'verification.json'; checked=json.loads(verification.read_text())
    assert checked['status']=='passed' and len(checked['trials'])==6
    summaries=[]; inputs={str(verification.name):sha(verification)}; selected=None
    for trial in checked['trials']:
        folder=batch/f'trial-{trial["index"]+1:02d}-{trial["case"]}'
        status_path=folder/'live-view-status.json';trace_path=folder/'live-view.jsonl'
        for path in (status_path,trace_path):
            key=str(path.relative_to(batch));assert sha(path)==checked['inputs_sha256'][key]
            inputs[key]=sha(path)
        status=json.loads(status_path.read_text())
        rows=[json.loads(line) for line in trace_path.read_text().splitlines()]
        names=status['head_joint_names']; ids=[status['joint_names'].index(n) for n in names]
        limits=np.asarray(status['joint_ranges_rad'])[ids]
        actual=np.asarray([r['head_joint_positions_rad'] for r in rows])
        target=np.asarray([r['head_pose_target_rad'] for r in rows])
        velocity=np.asarray([r['head_joint_velocities_rad_s'] for r in rows])
        steps=np.asarray([r['common_step_counter'] for r in rows])
        assert np.isfinite(np.c_[actual,target,velocity]).all() and (np.diff(steps)>0).all()
        outside=(target<limits[:,0]-1e-6)|(target>limits[:,1]+1e-6)
        joints=[]
        for j,name in enumerate(names):
            joints.append(dict(joint=name,hard_limits_deg=np.rad2deg(limits[j]).tolist(),
                desired_pose_min_max_deg=np.rad2deg([target[:,j].min(),target[:,j].max()]).tolist(),
                actual_min_max_deg=np.rad2deg([actual[:,j].min(),actual[:,j].max()]).tolist(),
                desired_pose_outside_hard_limit_samples=int(outside[:,j].sum()),
                sample_fraction_outside_hard_limit=float(outside[:,j].mean()),
                median_abs_pose_error_deg=float(np.rad2deg(np.median(abs(actual[:,j]-target[:,j])))),
                p95_abs_joint_speed_deg_s=float(np.rad2deg(np.quantile(abs(velocity[:,j]),.95)))))
        summaries.append(dict(folder=folder.name,case=trial['case'],samples=len(rows),
            common_step_range=[int(steps[0]),int(steps[-1])],joints=joints))
        if trial['index']==1:
            assert trial['case']=='gpu4';selected=(steps,actual,target,limits,names)
    out=batch/'head-commands';out.mkdir(exist_ok=True)
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        inputs_sha256=inputs,analyzer_sha256=sha(Path(__file__)),trials=summaries,
        plotted_trial='trial-02-gpu4',scope='Actual training environment 0 only. Changing policy, random head commands, initial state included. Fractions count saved samples; differing sample rates prevent direct time-fraction comparisons. Desired poses are current commands plus HOME, not raw servo action targets.',
        head_repair_applied=False,head_balance_causality_verified=False)
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot(out,*selected)
    print(json.dumps({'output':str(out),'trials':len(summaries),'plotted_trial':result['plotted_trial']}))


def plot(out,steps,actual,target,limits,names):
    os.environ.setdefault('MPLCONFIGDIR',str(Path.home()/'XGO-Duck/cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from PIL import Image
    from figure_style import apply_figure_style
    apply_figure_style(sizes=(11,10,9),font='DejaVu Sans')
    fig,axes=plt.subplots(4,1,figsize=(9,8.2),sharex=True)
    fig.subplots_adjust(left=.11,right=.97,bottom=.14,top=.90,hspace=.48)
    times=(steps-120000)*.02
    for j,(ax,name) in enumerate(zip(axes,names)):
        ax.plot(times,np.rad2deg(target[:,j]),color='#D55E00',linewidth=1.2,label='Desired pose')
        ax.plot(times,np.rad2deg(actual[:,j]),color='#0072B2',linewidth=1,label='Measured joint')
        for limit in np.rad2deg(limits[j]):ax.axhline(limit,color='#6B7280',linestyle='--',linewidth=.8)
        ax.set_title(name.replace('_',' ').capitalize(),pad=7)
        ax.set_ylabel('Angle (deg)');ax.grid(axis='y',alpha=.12);ax.margins(x=.01,y=.12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4,prune='both'))
    axes[-1].set_xlabel('Accumulated simulation time (s)')
    axes[-1].set_xlim(0,20);axes[-1].set_xticks([0,5,10,15,20])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.5,.995))
    fig.text(.11,.061,'Training environment 0, first GPU / 4 Hz trial; policy updates and random head commands.',fontsize=9)
    fig.text(.11,.026,'Dashed: compiled hard limits. Raw samples; startup and command changes retained.',fontsize=9)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    collisions=[]
    for i,(a,ab) in enumerate(texts):
        assert fig.bbox.contains(ab.x0,ab.y0) and fig.bbox.contains(ab.x1,ab.y1),a.get_text()
        for b,bb in texts[i+1:]:
            if ab.overlaps(bb):collisions.append([a.get_text(),b.get_text()])
    assert not collisions,collisions
    png=out/'head-command-tracking.png'
    with plt.rc_context({'savefig.bbox':None}):fig.savefig(png,dpi=220,bbox_inches=None,facecolor='white')
    img=Image.open(png);scale=img.width/fig.bbox.width
    for ax,name in zip(axes,names):
        box=ax.get_tightbbox(renderer).expanded(1.03,1.10)
        img.crop((max(0,int(box.x0*scale)),max(0,int(img.height-box.y1*scale)),min(img.width,int(np.ceil(box.x1*scale))),min(img.height,int(np.ceil(img.height-box.y0*scale))))).save(out/(name+'-panel.png'))
    plt.close(fig)
    (out/'plot-check.json').write_text(json.dumps(dict(status='passed',text_collisions=collisions,
        analysis_sha256=sha(out/'analysis.json'),png_sha256=sha(png),visual_inspection='pending'),indent=2)+'\n')


if __name__=='__main__':main()
