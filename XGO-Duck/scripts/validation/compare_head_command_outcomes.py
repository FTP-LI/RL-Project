"""Recompute saved head outcomes and paired episode differences on CPU.

The frozen pilot supplies two final policies and unchanged five-command traces.
This independent analysis reads those traces, checks the frozen verifier's
episode values, then plots all 32 first episodes without smoothing or choosing
a best checkpoint. A single training seed does not support a general efficacy
claim. --check-reference checks the measurement path against already verified
official data; it never creates a simulator, renders a scene or trains.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HEAD = ('neck_pitch', 'head_pitch', 'head_yaw', 'head_roll')
CONDITIONS = ('stop', 'forward', 'backward', 'turn_left', 'turn_right')
METRICS = ('mean_offset_deg', 'centered_angle_rms_deg', 'absolute_speed_p95_deg_s')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(folder, arm):
    report, verified = read(folder/'evaluation.json'), read(folder/'verification.json')
    assert verified['status'] == 'passed' and verified['report_sha256'] == sha(folder/'evaluation.json')
    assert (report['seed'], report['num_envs'], report['episode_seconds']) == (1001, 32, 20.0)
    names = report['model_metadata']['joint_names'].split(',')
    indices = [names.index(joint) for joint in HEAD]
    home = np.asarray(report['policy_home'], dtype=float)[indices]
    assert len(names) == 14
    rows, aggregate, inputs = [], [], {}
    for item in report['conditions']:
        condition = item['condition']
        file = folder/f'{condition}-trace.npz'
        binding = next(x for x in verified['checks'] if x['condition'] == condition)
        assert sha(file) == binding['trace_sha256']
        inputs[str(file.relative_to(ROOT))] = sha(file)
        with np.load(file, allow_pickle=False) as data:
            np.testing.assert_allclose(data['observations'][:, :, 51:61], 0, atol=1e-6)
            q = np.rad2deg(data['joint_positions'][:, :, indices] - home)
            speed = np.rad2deg(data['joint_velocities'][:, :, indices])
            valid = data['active']
            assert valid.dtype == bool and valid.shape == q.shape[:2]
            assert np.isfinite(q).all() and np.isfinite(speed).all()
            for window, begin in (('full_episode', 0), ('after_first_second', round(1/report['step_dt']))):
                mask = valid.copy()
                mask[:begin] = False
                for k, joint in enumerate(HEAD):
                    for env in range(32):
                        use = mask[:, env]
                        if not use.any():
                            continue
                        values, velocities = q[use, env, k], speed[use, env, k]
                        mean = float(np.sum(values)/len(values))
                        rows.append(dict(arm=arm, condition=condition, window=window,
                            joint=joint, env=env, samples=int(use.sum()), mean_offset_deg=mean,
                            centered_angle_rms_deg=float(np.sqrt(np.mean((values-mean)**2))),
                            absolute_speed_p95_deg_s=float(np.quantile(np.abs(velocities), .95))))
                    aggregate.append(dict(arm=arm, condition=condition, window=window, joint=joint,
                        actual_offset_min_median_max_deg=np.quantile(q[:, :, k][mask], [0, .5, 1]).tolist(),
                        actual_absolute_speed_p95_deg_s=float(np.quantile(np.abs(speed[:, :, k][mask]), .95))))
    assert {row['condition'] for row in rows} == set(CONDITIONS)
    inputs[str((folder/'evaluation.json').relative_to(ROOT))] = sha(folder/'evaluation.json')
    inputs[str((folder/'verification.json').relative_to(ROOT))] = sha(folder/'verification.json')
    return rows, aggregate, inputs


def key(row):
    return row['arm'], row['condition'], row['window'], row['joint'], int(row['env'])


def reference_check():
    """Independent aggregate check against an earlier, immutable analysis."""
    folder = ROOT/'records/evaluations/custom_onnx_reference_20261101T195044_s1001'
    old_file = folder/'head-motion/analysis.json'
    old = read(old_file)
    assert old['status'] == 'passed'
    rows, aggregates, inputs = extract(folder, 'current_official')
    comparisons = 0
    for row in aggregates:
        prior = next(x for x in old['results'] if x['model'] == 'current_official'
            and all(x[k] == row[k] for k in ('condition', 'window', 'joint')))
        for name in ('actual_offset_min_median_max_deg', 'actual_absolute_speed_p95_deg_s'):
            np.testing.assert_allclose(row[name], prior[name], atol=2e-6, rtol=2e-7)
            comparisons += 1
    inputs[str(old_file.relative_to(ROOT))] = sha(old_file)
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)), inputs_sha256=inputs,
        compared_aggregate_values=comparisons, episode_metric_rows=len(rows),
        actual_current_pilot_results_checked=False, simulation_started=False, training_started=False,
        scope='Same saved current-official traces independently recomputed; measurement path only')
    out = ROOT/'records/reports/head-comparison-entry-check.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'inputs_sha256'}, indent=2))


def plot(rows, out):
    os.environ.setdefault('MPLCONFIGDIR', str(Path.home()/'XGO-Duck/cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from PIL import Image
    from figure_style import apply_figure_style
    apply_figure_style(sizes=(11,10,9), font='DejaVu Sans')
    fig, axes = plt.subplots(2,2,figsize=(11,7.2),sharex=True)
    fig.subplots_adjust(left=.10,right=.97,bottom=.12,top=.87,hspace=.35,wspace=.28)
    labels = ('Stop','Forward','Backward','Left turn','Right turn')
    handles = []
    for ax, joint, title in zip(axes.flat, HEAD, ('Neck pitch','Head pitch','Head yaw','Head roll'), strict=True):
        for arm, offset, color, marker, label in (
                ('control',-.15,'#0072B2','o','Original commands'),
                ('reachable_head',.15,'#D55E00','s','Reachable neck commands')):
            for index, condition in enumerate(CONDITIONS):
                subset = sorted([r for r in rows if r['arm']==arm and r['joint']==joint
                    and r['window']=='after_first_second' and r['condition']==condition], key=lambda r:r['env'])
                assert subset
                values = np.array([x['centered_angle_rms_deg'] for x in subset])
                # Fixed, deterministic offsets; do not consume experiment RNG.
                jitter = ((np.arange(len(subset))*13)%len(subset))/(max(1,len(subset)-1))*.14-.07
                points = ax.scatter(index+offset+jitter,values,s=12,color=color,marker=marker,alpha=.42,edgecolors='none')
                ax.plot([index+offset-.09,index+offset+.09],[np.median(values)]*2,color=color,lw=2.2)
                if joint == HEAD[0] and index == 0:
                    handles.append((points,label))
        ax.set_title(title,loc='left',pad=8)
        ax.set_ylabel('Centered angle RMS (deg)')
        ax.set_xticks(range(5),labels)
        ax.set_xlim(-.5,4.5)
        ax.margins(y=.14)
        ax.yaxis.set_major_locator(MaxNLocator(4,prune='both'))
    fig.legend([x[0] for x in handles],[x[1] for x in handles],loc='upper center',bbox_to_anchor=(.53,.99),ncol=2,frameon=False)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    hits=[]
    for i,(a,box) in enumerate(texts):
        assert fig.bbox.contains(box.x0,box.y0) and fig.bbox.contains(box.x1,box.y1),a.get_text()
        for b,other in texts[i+1:]:
            if box.overlaps(other):hits.append([a.get_text(),b.get_text()])
    for ax in axes.flat:
        own=set(ax.get_xticklabels(which='both')+ax.get_yticklabels(which='both'))
        for spine in ax.spines.values():
            if spine.get_visible():
                box=spine.get_window_extent(renderer)
                hits.extend([t.get_text(),'spine'] for t,b in texts if t not in own and b.overlaps(box))
    assert not hits,hits
    file=out/'head-oscillation.png'
    with plt.rc_context({'savefig.bbox':None}):fig.savefig(file,dpi=300,bbox_inches=None,facecolor='white')
    picture=Image.open(file);scale=picture.width/fig.bbox.width
    for ax,joint in zip(axes.flat,HEAD,strict=True):
        box=ax.get_tightbbox(renderer).expanded(1.03,1.07)
        picture.crop((max(0,int(box.x0*scale)),max(0,int(picture.height-box.y1*scale)),
            min(picture.width,int(np.ceil(box.x1*scale))),min(picture.height,int(np.ceil(picture.height-box.y0*scale))))).save(out/f'{joint}-panel.png')
    plt.close(fig)
    (out/'plot-check.json').write_text(json.dumps(dict(status='passed',figure_sha256=sha(file),
        text_collisions=hits,visual_inspection='pending',points='Each available first episode; both policies, all five commands',
        markers='Small points are episodes; horizontal strokes are medians',window='Measured states 1.02 to 20 s, first episode only',
        training_seeds=1,smoothing=False),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path,nargs='?')
    parser.add_argument('--check-reference',action='store_true')
    args=parser.parse_args()
    if args.check_reference:
        assert args.batch is None
        return reference_check()
    assert args.batch is not None
    batch=args.batch.resolve();assert batch.parent==ROOT/'records/experiments'
    verified=read(batch/'verification.json');post=read(batch/'postprocessing.json')
    assert verified['status']=='passed' and post['status']=='completed'
    assert verified['matched_start']['status']=='passed' and verified['only_declared_neck_curriculum_changed']
    assert sha(batch/'head-episodes.csv')==verified['head_episode_table_sha256']
    with (batch/'head-episodes.csv').open() as f:previous={key(row):row for row in csv.DictReader(f)}
    rows=[];inputs={}
    for arm in ('control','reachable_head'):
        episodes,_,bindings=extract(Path(post['arms'][arm]['evaluation_record']),arm)
        rows.extend(episodes);inputs.update(bindings)
    assert set(map(key,rows))==set(previous)
    for row in rows:
        old=previous[key(row)]
        assert row['samples']==int(old['samples'])
        for name in METRICS:np.testing.assert_allclose(row[name],float(old[name]),atol=2e-6,rtol=2e-7)
    indexed={key(row):row for row in rows};paired=[]
    for candidate in (x for x in rows if x['arm']=='reachable_head'):
        k=key(candidate);control=indexed.get(('control',*k[1:]))
        if control is None:continue
        paired.append({**{k:candidate[k] for k in ('condition','window','joint','env')},
            'control_samples':control['samples'],'candidate_samples':candidate['samples'],
            **{name:candidate[name]-control[name] for name in METRICS}})
    assert paired
    out=batch/'head-comparison';out.mkdir(exist_ok=True)
    with (out/'paired-episode-differences.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    inputs.update({str((batch/name).relative_to(ROOT)):sha(batch/name) for name in ('verification.json','head-episodes.csv','postprocessing.json')})
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),
        inputs_sha256=inputs,independently_recomputed_episode_rows=len(rows),paired_episode_rows=len(paired),
        paired_table_sha256=sha(out/'paired-episode-differences.csv'),
        unmatched_episode_windows=len(rows)-2*len(paired),difference_sign='reachable_head minus control',
        simulation_started=False,training_started=False,general_head_improvement_verified=False,
        scope='Descriptive paired episodes from one trained seed; all head joints, commands and declared windows retained')
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(rows,out)
    print(json.dumps({k:v for k,v in result.items() if k!='inputs_sha256'},indent=2))


if __name__=='__main__':main()
