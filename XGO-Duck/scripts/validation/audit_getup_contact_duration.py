"""Saved-development-trace contact duration audit; no dynamics or learning.

Compare every success and failure. Contact duration alone can also mark a
successful recovery, so report overlap rather than selecting a reward from
the best-looking camera. A stateful future reward needs privileged contact
memory, reset rules, a new matched value start and a separate source freeze.
"""
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/training'),str(ROOT/'scripts/validation')]
from s1_reference_inputs import read,sha
from record_io import write
from getup_coverage_metrics import evaluate
from recovery_end_hold import evaluate_end_hold
OUT=ROOT/'records/reports/getup-teacher-anchor-pair/contact-duration'


def age_seconds(mask,dt=.02):
    age=np.zeros(mask.shape[1],np.int64);result=[]
    for row in mask:
        age=np.where(row,age+1,0);result.append(age.copy())
    return np.asarray(result)*dt


def quantiles(x):
    return {name:float(v) for name,v in zip(('min','p25','median','p75','max'),np.quantile(x,[0,.25,.5,.75,1]))} if len(x) else None


def main():
    OUT.mkdir(exist_ok=True)
    parent=OUT.parent;roles={};sources={};episode_rows=[]
    # Timing is fixed by the previous getup protocol, not chosen from outcomes.
    thresholds=[.1,.2,.4,.8,1.2]
    for role in ('original','control','candidate'):
        proof=read(parent/f'{role}-analysis.json');folder=ROOT/proof['record']
        p=read(folder/'protocol.json');state=read(folder/'evaluation.json')
        path=folder/'coverage-trace.npz';assert sha(path)==state['trace_sha256']
        with np.load(path) as f:t={k:f[k].copy() for k in f.files}
        assert t['native_head_leg_contact'].shape==(550,320) and t['active'].all()
        assert not t['terminated'].any() and not t['truncated'][:-1].any() and t['truncated'][-1].all()
        coverage=evaluate(t,p);assert coverage==read(folder/'coverage.json')
        end=evaluate_end_hold(t,p,coverage['episodes']);assert end==read(folder/'end-hold.json')
        passed=np.asarray(end['stable_by_environment'],bool);times=t['time']
        live=(times>2+1e-8)&(times<=8+1e-8);full=times>2+1e-8;late=times>8+1e-8
        contact=t['native_head_leg_contact'].astype(bool)
        # Zero contact memory at the actual control release boundary.
        age=age_seconds(contact&full[:,None]);peak=age[live].max(0)
        q=t['quaternion'];tilt=np.rad2deg(np.arccos(np.clip(1-2*(q[...,1]**2+q[...,2]**2)/np.sum(q*q,-1),-1,1)))
        bad_pose=(abs(t['height']-p['criteria']['nominal_height_m'])>p['criteria']['height_tolerance_m'])|(tilt>p['criteria']['tilt_max_deg'])
        rows=[]
        for threshold in thresholds:
            persistent=(age>=threshold-1e-8)&live[:,None]
            flagged=persistent.any(0)
            pose_flagged=(persistent&bad_pose).any(0)
            rows.append(dict(threshold_seconds=threshold,
                successful_flagged=int((flagged&passed).sum()),failed_flagged=int((flagged&~passed).sum()),
                successful_pose_flagged=int((pose_flagged&passed).sum()),failed_pose_flagged=int((pose_flagged&~passed).sum()),
                successful_flagged_fraction=float(flagged[passed].mean()),failed_flagged_fraction=float(flagged[~passed].mean())))
        roles[role]=dict(successes=int(passed.sum()),failures=int((~passed).sum()),
            successful_peak_contact_seconds=quantiles(peak[passed]),failed_peak_contact_seconds=quantiles(peak[~passed]),
            successful_total_contact_seconds=quantiles(contact[live][:,passed].sum(0)*.02),
            failed_total_contact_seconds=quantiles(contact[live][:,~passed].sum(0)*.02),
            successful_late_contact_worlds=int(contact[late][:,passed].any(0).sum()),
            failed_late_contact_worlds=int(contact[late][:,~passed].any(0).sum()),thresholds=rows)
        for i,case in enumerate(coverage['episodes']):
            episode_rows.append(dict(role=role,world=i,requested=case['requested_pose'],actual=case['release_pose'],
                success=bool(passed[i]),peak_contact_before_deadline_seconds=float(peak[i]),
                total_contact_before_deadline_seconds=float(contact[live,i].sum()*.02),
                late_contact_seconds=float(contact[late,i].sum()*.02)))
        for path in (path,folder/'protocol.json',folder/'evaluation.json',folder/'coverage.json',folder/'end-hold.json',parent/f'{role}-analysis.json'):
            sources[str(path.relative_to(ROOT))]=sha(path)
    assert len(episode_rows)==960
    result=dict(status='passed-saved-development-data-only',checked_at=datetime.now(timezone.utc).isoformat(),
        roles=roles,episodes=960,rows=episode_rows,source_sha256=sources,script_sha256=sha(Path(__file__)),
        age_definition='Consecutive native head-leg contact controls after2s release; clear on every noncontact control; dt0.02s',
        descriptive_sweep_seconds=thresholds,measurement_window_seconds=[2,8],final_window_seconds=[8,11],
        reward_coefficient_selected=False,threshold_selected=False,heldout_consumed=False,
        physics_steps=0,learning_updates=0,
        scope='Association and successful-motion overlap on existing development episodes; no causal or policy-improvement claim')
    write(OUT/'analysis.json',result)
    render(result)
    print('CONTACT_DURATION_AUDIT',[(r,x['successes'],x['failures'],x['successful_peak_contact_seconds'],x['failed_peak_contact_seconds']) for r,x in roles.items()],flush=True)


def render(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager
    from figure_style import apply_figure_style
    from render_roll_angle_results import checked_save
    fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    apply_figure_style(font='Microsoft YaHei',sizes=(11,10,9))
    matplotlib.rcParams['savefig.bbox']=None
    fig,ax=plt.subplots(figsize=(8.2,4.8));fig.subplots_adjust(left=.13,right=.97,bottom=.25,top=.89)
    labels=[];positions=[];x=0;random=np.random.default_rng(300)
    for role,name in [('original','官方'),('control','原奖励'),('candidate','保持约束')]:
        for passed,color,text in [(True,'#2479a8','成功'),(False,'#be771c','失败')]:
            values=np.asarray([r['peak_contact_before_deadline_seconds'] for r in result['rows'] if r['role']==role and r['success']==passed])
            ax.scatter(x+random.uniform(-.16,.16,len(values)),values,s=8,alpha=.35,color=color,linewidths=0)
            ax.plot([x-.2,x+.2],[np.median(values)]*2,color=color,linewidth=2.3)
            labels.append(f'{name}{text}\nn={len(values)}');positions.append(x);x+=1
        x+=.4
    ax.set(xticks=positions,xticklabels=labels,ylim=(-.2,6),xlim=(-.5,x-.9),
        ylabel='2–8秒内最长连续头腿接触（秒）',yticks=[0,1,2,3,4,5,6])
    ax.set_title('头腿接触时长',loc='left')
    path=ROOT/'docs/reports/assets/getup-contact-duration.png'
    proof=checked_save(fig,path)
    write(OUT/'figure-review.json',dict(status='rendered-awaiting-visual-review',figure=proof,
        analysis_sha256=sha(OUT/'analysis.json'),script_sha256=sha(Path(__file__)),
        episodes_plotted=960,all_samples_plotted=True,median_marks=True,physics_steps=0))


if __name__=='__main__':main()
