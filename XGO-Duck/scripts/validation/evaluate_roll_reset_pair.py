"""Matched fixed-policy reset-pose diagnostic after the 28/32 roll reference.

Change only four initial hip-pitch/ankle HOME values: source 22 degrees versus
the model's stored ~24-degree reference. Absolute targets and relative inputs
still use model metadata in both arms. Grounding height may change as a direct
consequence of posture. Keep all criteria, contact physics and 32-trial budget.
No learning, no silent extension, and no claim of three-seed trained robustness.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import numpy as np
import onnx
import evaluate_roll_reference as core
from evaluate_standup_reference import ROOT,WORK,read,sha,write,Tee

PLAN=ROOT/'configs/experiments/roll_reset_pair.json'
OUT=ROOT/'records/reports/roll-reset-pair'
BASE_CONFIG=core.config
SELECTED=('left_hip_pitch','right_hip_pitch','left_ankle','right_ankle')


def model_home():
    meta={m.key:m.value for m in onnx.load(str(core.MODEL)).metadata_props}
    return dict(zip(meta['joint_names'].split(','),map(float,meta['default_joint_pos'].split(','))))


def config(plan):
    cfg=BASE_CONFIG(plan)
    if plan['arm']=='candidate':
        home=model_home();pose=cfg.scene.entities['robot'].init_state.joint_pos
        for name in SELECTED:
            key=f'.*{name}.*';assert key in pose
            pose[key]=home[name]
    return cfg


def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        assert set(a)==set(b),path
        return sum((differences(a[k],b[k],path+'.'+str(k)) for k in a),[])
    if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
        assert len(a)==len(b),path
        return sum((differences(x,y,path+f'[{i}]') for i,(x,y) in enumerate(zip(a,b))),[])
    return [] if repr(a)==repr(b) else [dict(path=path,control=repr(a),candidate=repr(b))]


def freeze():
    import torch
    import mujoco
    assert not PLAN.exists()
    evidence=read(ROOT/'records/reports/roll-reference/failure-diagnostic.json')
    assert evidence['status']=='passed' and evidence['original_successes']==28 and not evidence['original_gate_passed']
    base=read(ROOT/'configs/experiments/roll_reference.json');plans={};cfgs={}
    for arm in ('control','candidate'):
        plan=deepcopy(base);plan.update(arm=arm,seed=1002,frozen_at=datetime.now(timezone.utc).isoformat(),
            scope='Development reset-pose diagnostic only; same fixed policy; original roll verdict preserved')
        plan['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
        cfgs[arm]=config(plan);plans[arm]=plan
    diff=differences(asdict(cfgs['control']),asdict(cfgs['candidate']))
    assert len(diff)==4 and all('.init_state.joint_pos.' in d['path'] for d in diff),diff
    for name in SELECTED:assert any(name in d['path'] for d in diff)
    # CPU model compilation is used only to locate qpos coordinates for later
    # start-state comparison. No physics step or CUDA context is created.
    model=cfgs['control'].scene.entities['robot'].spec_fn().compile()
    positions={}
    for name in SELECTED:
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name);assert jid>=0
        positions[name]=int(model.jnt_qposadr[jid])
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=True)
    paths={}
    for arm,plan in plans.items():
        file=ROOT/f'configs/experiments/roll_reset_{arm}.json';assert not file.exists()
        write(file,plan);paths[arm]=str(file.relative_to(ROOT))
    pair=dict(status='frozen-before-simulation',frozen_at=datetime.now(timezone.utc).isoformat(),
        arms=paths,arm_protocol_sha256={a:sha(ROOT/p) for a,p in paths.items()},
        source_protocol_sha256=sha(ROOT/'configs/experiments/roll_reference.json'),
        rationale_sha256=sha(ROOT/'records/reports/roll-reference/failure-diagnostic.json'),
        episodes_each=32,seed=1002,learning_updates=0,automatic_extension=False,
        changed_initial_joints=list(SELECTED),qpos_indices=positions,configuration_differences=diff,
        continue_gate=dict(candidate_fraction_min=.9,minimum_additional_successes_over_control=1),
        scope='Two-arm fixed-policy initialization screen; no training or heldout consumption')
    write(PLAN,pair);write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
        configuration_differences=diff,cuda_initialized=False,physics_stepped=False))
    print(json.dumps(pair,indent=2))


def run_arm(arm):
    assert os.isatty(1),'Use the shared visible VS Code terminal'
    pair=read(PLAN);planfile=ROOT/pair['arms'][arm];plan=read(planfile)
    assert sha(planfile)==pair['arm_protocol_sha256'][arm]
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(PLAN)
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    assert sha(core.MODEL)==plan['model_sha256']
    out=OUT/arm;out.mkdir(exist_ok=True);assert not (out/'progress.json').exists(),'Existing attempt retained'
    run_id=f'duck_roll_reset-pose_{arm}_s1002_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=ROOT/'records/evaluations'/run_id;folder.mkdir();(folder/'videos').mkdir()
    log=(folder/'execution.log').open('a',buffering=1);sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    state=dict(status='running',run_id=run_id,pid=os.getpid(),path=str(folder),conditions=[],
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(planfile),
        model_sha256=sha(core.MODEL),learning_updates=0,arm=arm)
    write(out/'progress.json',state);write(folder/'evaluation.json',state)
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=run_id,pid=os.getpid()))
    shutil.copy2(planfile,folder/'protocol.json')
    for name in plan['source_sha256']:shutil.copy2(ROOT/name,folder/Path(name).name)
    core.config=config
    try:core.evaluate(plan,folder,state);state['status']='completed'
    except BaseException as e:
        state.update(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=repr(e));raise
    finally:
        state['ended_at']=datetime.now(timezone.utc).isoformat()
        write(folder/'evaluation.json',state);write(out/'progress.json',state)
        print('ROLL_RESET_ARM:',arm,state['status'],flush=True)
    import verify_roll_reference as verifier
    verifier.OUT=out;verifier.main()


def analyze():
    pair=read(PLAN);results={};starts={};reports={}
    for arm in ('control','candidate'):
        results[arm]=read(OUT/arm/'analysis.json');assert results[arm]['status']=='passed'
        progress=read(OUT/arm/'progress.json');folder=Path(progress['path'])
        report=read(folder/'evaluation.json');reports[arm]=report
        assert sha(folder/'evaluation.json')==results[arm]['report_sha256']
        with np.load(folder/'forward-roll-trace.npz') as t:starts[arm]=t['initial_qpos'].copy()
    a,b=starts['control'],starts['candidate'];assert a.shape==b.shape
    allowed=[2]+list(pair['qpos_indices'].values())
    fixed=[j for j in range(a.shape[1]) if j not in allowed]
    np.testing.assert_allclose(a[:,fixed],b[:,fixed],rtol=0,atol=1e-7)
    cfg_a,cfg_b=config(read(ROOT/pair['arms']['control'])),config(read(ROOT/pair['arms']['candidate']))
    changes=differences(asdict(cfg_a),asdict(cfg_b));assert changes==pair['configuration_differences']
    gain=results['candidate']['successes']-results['control']['successes'];gate=pair['continue_gate']
    worth=(results['candidate']['success_fraction']>=gate['candidate_fraction_min'] and
           gain>=gate['minimum_additional_successes_over_control'])
    rows_a=results['control']['episodes'];rows_b=results['candidate']['episodes']
    wins=sum((not x['success']) and y['success'] for x,y in zip(rows_a,rows_b))
    losses=sum(x['success'] and (not y['success']) for x,y in zip(rows_a,rows_b))
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        results={a:{k:v for k,v in r.items() if k!='episodes'} for a,r in results.items()},
        candidate_additional_successes=gain,paired_wins=wins,paired_losses=losses,
        further_confirmation_worthwhile=worth,configuration_differences=changes,
        same_initial_root_xy_orientation_and_other_joints=True,
        initial_root_z_max_difference_m=float(abs(a[:,2]-b[:,2]).max()),
        initial_changed_joint_differences_rad={name:(b[:,j]-a[:,j]).tolist() for name,j in pair['qpos_indices'].items()},
        protocol_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),original_verdict_rewritten=False,
        learning_updates=0,scope=pair['scope'],
        next_action='Independent development confirmation before adoption' if worth else 'Review initialization hypothesis; no automatic budget extension')
    write(OUT/'analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='initial_changed_joint_differences_rad'},indent=2))


def run():
    assert os.isatty(1),'Use the single shared visible terminal'
    assert not (OUT/'progress.json').exists(),'Do not repeat an existing pair'
    state=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),arms=[])
    write(OUT/'progress.json',state)
    try:
        for arm in ('control','candidate'):
            code=subprocess.call([sys.executable,'-B',str(Path(__file__)),'--arm',arm],cwd=WORK)
            state['arms'].append(dict(arm=arm,exit_code=code));write(OUT/'progress.json',state)
            assert code==0,(arm,code)
        analyze();state['status']='completed'
    except BaseException as e:state.update(status='failed',error=repr(e));raise
    finally:
        state['ended_at']=datetime.now(timezone.utc).isoformat();write(OUT/'progress.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--arm',choices=('control','candidate'));p.add_argument('--analyze',action='store_true');args=p.parse_args()
    if args.freeze:freeze()
    elif args.arm:run_arm(args.arm)
    elif args.analyze:analyze()
    else:run()
