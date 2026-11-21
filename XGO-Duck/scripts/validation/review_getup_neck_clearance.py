"""Saved-state neck-clearance diagnostic, without dynamics or new episodes.

Compare every failed recovery with an equal number of successful actual-supine
examples. Count geometric head/leg overlap in the saved poses. Then inspect
one-dimensional neck lower bounds at the recovery deadline, keeping every
other coordinate exact. This can justify a prospective lightweight hypothesis,
but a static pose edit is never a recovered rollout or a successful getup.
"""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import mujoco
import torch
import onnx
from mjlab.scene import Scene
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from common_scene_contract import model_arrays,compare_arrays
from getup_release_coverage import prepare

OUT=ROOT/'records/reports/getup-release-coverage/neck-clearance'
INPUT=ROOT/'records/reports/getup-release-coverage/fixed-evaluation'
ANGLES=(-120,-105,-90,-75,-60,-45,-30,-15,0,15,30,45,60)


def contacts(model,data,head,legs):
    pairs=set();head_leg=False;head_floor_min=0.
    for c in data.contact:
        a,b=map(int,c.geom)
        if c.dist>=-1e-5:continue
        if head in (a,b) and ({a,b}-{head}) & legs:head_leg=True
        names=tuple(sorted(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in (a,b)))
        if all(n.startswith('robot/') for n in names):pairs.add(' | '.join(names))
        elif head in (a,b):head_floor_min=min(head_floor_min,float(c.dist))
    return dict(head_leg_overlap=head_leg,self_pairs=sorted(pairs),head_floor_min_m=head_floor_min)


def main():
    assert not torch.cuda.is_initialized()
    result=read(INPUT/'analysis.json');assert result['status']=='passed'
    all_rows=[];sources={};sweeps=[];summaries={}
    for role,proof in result['roles'].items():
        record=ROOT/proof['record'];path=record/'coverage-trace.npz'
        assert sha(path)==proof['trace_sha256'];sources[str(path.relative_to(ROOT))]=sha(path)
        with np.load(path) as f:t={k:f[k] for k in f.files}
        p=read(record/'protocol.json');cfg,_,_=prepare(p);model=Scene(cfg.env.scene,device='cpu').compile()
        with np.load(record/'scene-static-fields.npz') as f:compare_arrays(model_arrays(model),{k:f[k] for k in f.files})
        metadata={x.key:x.value for x in onnx.load(ROOT/p['model']).metadata_props}
        names=metadata['joint_names'].split(',');neck=names.index('neck_pitch')
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'robot/neck_pitch');address=int(model.jnt_qposadr[jid])
        np.testing.assert_allclose(t['qpos'][...,address],t['joint_positions'][...,neck],atol=2e-6,rtol=0)
        head=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'robot/head_collision')
        legs={mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'robot/'+n) for n in ('left_foot_collision','right_foot_collision','left_hip_yaw_collision','right_hip_yaw_collision')}
        assert min([head,*legs])>=0
        episodes=read(record/'coverage.json')['episodes'];failed=[r['env_index'] for r in episodes if r['hold_start_seconds'] is None]
        # Fixed index order, matched on actual release; no selection by looks.
        success=[r['env_index'] for r in episodes if r['hold_start_seconds'] is not None and r['release_pose']=='supine'][:len(failed)]
        data=mujoco.MjData(model)
        for label,worlds in (('failed',failed),('timely-hold',success)):
            for i in worlds:
                sample_rows=[]
                for step in range(250,551,10):
                    q=t['qpos'][step-1,i];data.qpos[:]=q;data.qvel[:]=t['qvel'][step-1,i]
                    mujoco.mj_forward(model,data);np.testing.assert_array_equal(data.qpos,q)
                    c=contacts(model,data,head,legs)
                    g=t['observations'][step-1,i,3:6];tilt=float(np.degrees(np.arccos(np.clip(-g[2]/np.linalg.norm(g),-1,1))))
                    sample_rows.append(dict(step=step,head_leg=c['head_leg_overlap'],pairs=c['self_pairs'],
                        neck_deg=float(np.degrees(q[address])),requested_neck_deg=float(np.degrees(t['applied_targets'][step-1,i,neck])),
                        observed_tilt_deg=tilt))
                late=[r for r in sample_rows if r['step']>=400]
                row=dict(role=role,outcome=label,environment_index=i,requested_pose=str(t['requested_pose'][i]),
                    release_pose=episodes[i]['release_pose'],samples=sample_rows,
                    late_head_leg_samples=sum(r['head_leg'] for r in late),late_samples=len(late),
                    late_neck_median_deg=float(np.median([r['neck_deg'] for r in late])),
                    late_requested_neck_median_deg=float(np.median([r['requested_neck_deg'] for r in late])))
                all_rows.append(row)
                if label=='failed':
                    q=t['qpos'][399,i];data.qpos[:]=q;data.qvel[:]=t['qvel'][399,i];mujoco.mj_forward(model,data)
                    original=contacts(model,data,head,legs)
                    for angle in ANGLES:
                        data.qpos[:]=q;data.qpos[address]=max(q[address],np.deg2rad(angle));mujoco.mj_forward(model,data)
                        c=contacts(model,data,head,legs);new=set(c['self_pairs'])-set(original['self_pairs'])
                        sweeps.append(dict(role=role,environment_index=i,lower_bound_deg=angle,
                            original_neck_deg=float(np.degrees(q[address])),applied_neck_deg=float(np.degrees(data.qpos[address])),
                            original_head_leg=original['head_leg_overlap'],head_leg=c['head_leg_overlap'],new_self_pairs=sorted(new),
                            head_floor_min_m=c['head_floor_min_m']))
        summaries[role]={}
        for label in ('failed','timely-hold'):
            rows=[r for r in all_rows if r['role']==role and r['outcome']==label]
            summaries[role][label]=dict(episodes=len(rows),any_late_head_leg=sum(r['late_head_leg_samples']>0 for r in rows),
                continuous_late_head_leg=sum(r['late_head_leg_samples']==r['late_samples'] for r in rows),
                median_actual_neck_deg=float(np.median([r['late_neck_median_deg'] for r in rows])),
                median_requested_neck_deg=float(np.median([r['late_requested_neck_median_deg'] for r in rows])))
    aggregation=[]
    for angle in ANGLES:
        rows=[r for r in sweeps if r['lower_bound_deg']==angle]
        aggregation.append(dict(lower_bound_deg=angle,cases=len(rows),original_head_leg_cases=sum(r['original_head_leg'] for r in rows),
            remaining_head_leg_cases=sum(r['head_leg'] for r in rows),new_self_contact_cases=sum(bool(r['new_self_pairs']) for r in rows),
            geometric_clear_no_new_self=sum(not r['head_leg'] and not r['new_self_pairs'] for r in rows),
            head_floor_penetration_over2mm=sum(r['head_floor_min_m']<-.002 for r in rows)))
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=True)
    report=dict(status='diagnostic-complete',summaries=summaries,sweep_summary=aggregation,rows=all_rows,sweeps=sweeps,
        source_trace_sha256=sources,script_sha256=sha(Path(__file__)),physics_steps=0,new_episodes=0,learning_updates=0,
        scope='CPU collision geometry at saved states, not captured GPU impulses or causal dynamics; angle sweeps are counterfactual static geometry only',
        unchanged_original_results=True,original_qpos_preserved=True,penetration_threshold_m=-1e-5)
    write(OUT/'analysis.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ('status','summaries','sweep_summary','physics_steps','scope')},indent=2))


if __name__=='__main__':main()
