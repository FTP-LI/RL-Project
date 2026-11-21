"""Freeze and analyze actual live-observer samples; never run a simulator.

Diagnoses the user's leg-adduction observation using signed foot-site spacing,
hip angles and compiled contact masks. One changing training environment is
not a fixed-policy gait evaluation, and a collision omission is not by itself
proof of the optimizer's causal preference. No training parameter is changed.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml
os.environ.setdefault('MPLCONFIGDIR', str(Path.home()/'XGO-Duck/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
from figure_style import apply_figure_style

ROOT=Path(__file__).resolve().parents[2]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('--snapshot',type=Path,help='Recompute the exact saved inputs without collecting new samples')
    args=parser.parse_args()
    record=ROOT/'records/experiments'/args.run_id
    assert record.parent==ROOT/'records/experiments' and record.is_dir()
    if args.snapshot:
        out=args.snapshot.resolve()
        assert out.parent==record/'gait-diagnostics'
    else:
        out=record/'gait-diagnostics'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        out.mkdir(parents=True)
        raw=(record/'live-view.jsonl').read_bytes()
        complete=raw[:raw.rfind(b'\n')+1]
        (out/'live-samples.jsonl').write_bytes(complete)
        (out/'live-status.json').write_bytes((record/'live-view-status.json').read_bytes())
    rows=[json.loads(line) for line in (out/'live-samples.jsonl').read_text().splitlines()]
    meta=json.loads((out/'live-status.json').read_text())
    if not (out/'env.yaml').exists():
        logs=list((Path.home()/'XGO-Duck/workspace/xgoduck_rl/logs/rsl_rl/xgoduck_velocity').glob('*_'+args.run_id))
        assert len(logs)==1
        (out/'env.yaml').write_bytes((logs[0]/'params/env.yaml').read_bytes())
    env=yaml.load((out/'env.yaml').read_text(),Loader=yaml.BaseLoader)
    step_dt=float(env['sim']['mujoco']['timestep'])*int(env['decimation'])
    assert len(rows)>2 and meta['extra_environment_steps']==0
    steps=np.asarray([r['common_step_counter'] for r in rows])
    assert np.all(np.diff(steps)>0)
    q=np.asarray([r['joint_positions_rad'] for r in rows])
    feet=np.asarray([r['foot_site_positions_world_m'] for r in rows])
    gap=np.asarray([r['signed_foot_lateral_gap_m'] for r in rows])
    command=np.asarray([r['command'] for r in rows])
    assert all(np.isfinite(x).all() for x in (q,feet,gap,command))
    recomputed=np.linalg.norm(feet[:,0,:2]-feet[:,1,:2],axis=1)
    assert np.allclose(recomputed,[r['foot_site_distance_xy_m'] for r in rows],atol=1e-8)
    geoms={g['name']:g for g in meta['compiled_geometries']}
    left,right=[geoms['robot/'+n+'_foot_collision'] for n in ('left','right')]
    mask_eligible=bool((left['contype'] & right['conaffinity']) or (right['contype'] & left['conaffinity']))
    explicit=any({p['geom1'],p['geom2']}=={left['id'],right['id']} for p in meta['explicit_contact_pairs'])
    roll_idx=[meta['joint_names'].index(n+'_hip_roll') for n in ('left','right')]
    angles=q[:,roll_idx]*180/np.pi
    limits=np.asarray(meta['joint_ranges_rad'])[roll_idx]*180/np.pi
    self_contact=sum(r['gpu_self_collision_detected'] for r in rows)
    wall=(datetime.fromisoformat(rows[-1]['at'])-datetime.fromisoformat(rows[0]['at'])).total_seconds()
    observer_cost=sum(r['observer_wall_seconds'] for r in rows)
    table=[dict(control_step=int(s),relative_control_step=int(s-steps[0]),
        signed_foot_gap_mm=float(g*1000),left_hip_roll_deg=float(a[0]),right_hip_roll_deg=float(a[1]),
        command_vx=float(c[0]),command_vy=float(c[1]),command_yaw=float(c[2]))
        for s,g,a,c in zip(steps,gap,angles,command)]
    with (out/'gait-samples.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    result=dict(status='passed',run_id=args.run_id,samples=len(rows),environment_index=0,
        first_control_step=int(steps[0]),last_control_step=int(steps[-1]),
        first_at=rows[0]['at'],last_at=rows[-1]['at'],
        signed_gap_mm_min_median_max=(np.quantile(gap,[0,.5,1])*1000).tolist(),
        crossed_lateral_samples=int((gap<0).sum()),
        left_hip_roll_deg_min_median_max=np.quantile(angles[:,0],[0,.5,1]).tolist(),
        right_hip_roll_deg_min_median_max=np.quantile(angles[:,1],[0,.5,1]).tolist(),
        hip_roll_limits_deg=limits.tolist(),foot_geometries=[left,right],
        foot_to_foot_mask_eligible=mask_eligible,foot_to_foot_explicit_pair=explicit,
        gpu_self_collision_samples=self_contact,
        command_min=command.min(axis=0).tolist(),command_max=command.max(axis=0).tolist(),
        measured_observer_seconds=observer_cost,elapsed_wall_seconds=wall,
        observer_timed_fraction=observer_cost/wall,observed_wall_fps=(len(rows)-1)/wall,
        control_step_dt_seconds=step_dt,control_steps_per_wall_second=float(steps[-1]-steps[0])/wall,
        simulated_seconds=float(steps[-1]-steps[0])*step_dt,
        simulation_time_per_wall_second=float(steps[-1]-steps[0])*step_dt/wall,
        observer_cost_scope='Timed copy/forward/log/sync only; excludes asynchronous native rendering and GPU contention',
        snapshot_sha256={p.name:sha(p) for p in (out/'live-samples.jsonl',out/'live-status.json',out/'env.yaml')},
        analysis_sha256=sha(Path(__file__)),stable_gait_verified=False,
        scope='One live training environment, changing weights/random commands/resets; descriptive samples, not independent evaluation episodes',
        causal_reward_explanation_verified=False)
    apply_figure_style(sizes=(11,10,9),font='DejaVu Sans')
    matplotlib.rcParams['savefig.bbox']=None
    fig,axes=plt.subplots(2,1,figsize=(8,5.5),sharex=True)
    fig.subplots_adjust(left=.16,right=.96,top=.96,bottom=.12,hspace=.25)
    x=steps-steps[0]
    axes[0].plot(x,gap*1000,color='#0072B2',linewidth=1.2)
    axes[0].axhline(0,color='#666666',linestyle=':',linewidth=1,label='Zero lateral separation')
    axes[0].set_ylabel('Left - right foot gap (mm)')
    axes[0].legend(loc='upper right')
    for i,(name,color) in enumerate((('Left','#0072B2'),('Right','#D55E00'))):
        axes[1].plot(x,angles[:,i],color=color,linewidth=1.2,label=name)
    for i,limit in enumerate(sorted(set(limits.ravel()))):
        axes[1].axhline(limit,color='#777777',linestyle=':',linewidth=1,
                       label='Joint limits' if i==0 else None)
    axes[1].set_ylabel('Hip roll (degrees)')
    axes[1].set_xlabel('Control steps since restored state')
    axes[1].legend(loc='center')
    axes[-1].xaxis.set_major_locator(MaxNLocator(6,integer=True,prune='both'))
    for ax in axes:
        ax.margins(x=.04,y=.12)
        ax.yaxis.set_major_locator(MaxNLocator(5,prune='both'))
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
           if t.get_visible() and t.get_text().strip()]
    collisions=[]
    for i,(a,box) in enumerate(texts):
        assert fig.bbox.contains(box.x0,box.y0) and fig.bbox.contains(box.x1,box.y1),a.get_text()
        collisions.extend([a.get_text(),b.get_text()] for b,other in texts[i+1:] if box.overlaps(other))
    for ax in axes:
        own=set(ax.get_xticklabels(which='both')+ax.get_yticklabels(which='both'))
        for spine in ax.spines.values():
            if spine.get_visible():
                box=spine.get_window_extent(renderer)
                collisions.extend([t.get_text(),'spine'] for t,tbox in texts if t not in own and tbox.overlaps(box))
    assert not collisions,collisions
    figure=out/'leg-adduction.png';fig.savefig(figure,dpi=300,bbox_inches=None,facecolor='white')
    pic=Image.open(figure);sx,sy=pic.width/fig.bbox.width,pic.height/fig.bbox.height
    for i,ax in enumerate(axes):
        box=ax.get_tightbbox(renderer)
        pic.crop((max(0,int((box.x0-4)*sx)),max(0,int(pic.height-(box.y1+4)*sy)),
                  min(pic.width,int((box.x1+4)*sx)+1),min(pic.height,int(pic.height-box.y0*sy)+1))).save(out/f'panel-{i+1}.png')
    result.update(figure_sha256=sha(figure),csv_sha256=sha(out/'gait-samples.csv'),
                  text_collisions=collisions,visual_inspection='pending')
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(out),**result},indent=2))


if __name__=='__main__': main()
