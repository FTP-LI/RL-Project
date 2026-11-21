"""Fixed-size lateral-release coverage supplement, without policy training.

Why: 38/64 lateral proposals settled supine during the official HOME freeze.
Keep the original verdict. Predefine 128 falls per lateral direction, classify
by actual pose at release BEFORE recovery, retain every outcome, and require
at least 32 actual releases on EACH lateral side with >=95% stable recovery.
No resampling until enough successes, no mutation of the official controller.
"""
from datetime import datetime,timezone
from pathlib import Path
from collections import defaultdict
import argparse
import json
import os
import shutil
import sys
import numpy as np
import evaluate_standup_reference as core
from evaluate_standup_reference import ROOT,WORK,read,sha,write,Tee

PLAN=ROOT/'configs/experiments/lateral_recovery_coverage.json'
OUT=ROOT/'records/reports/lateral-recovery'


def freeze():
    import torch
    assert not PLAN.exists()
    evidence=ROOT/'records/reports/standup-reference/release-diagnostic.json'
    assert read(evidence)['status']=='passed'
    plan=read(ROOT/'configs/experiments/standup_reference.json')
    plan.update(frozen_at=datetime.now(timezone.utc).isoformat(),episodes_per_pose=128,seed=1002,
        poses=['left-side','right-side'],scope='Fixed-size development coverage supplement; original 128-trial verdict unchanged',
        actual_release_gate=dict(poses=['left-side','right-side'],minimum_episodes_each=32,success_fraction_min=.95),
        release_policy='All 256 proposals retained; primary bins use actual release pose before policy activation; no outcome selection',
        rationale_evidence_sha256=sha(evidence))
    plan['source_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in
        (Path(__file__),ROOT/'scripts/validation/verify_standup_reference.py')})
    cfg=core.config(plan);assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=True);write(PLAN,plan)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),cuda_initialized=False,
        episodes=256,physics_step_seconds=cfg.sim.mujoco.timestep))
    print(json.dumps(plan,indent=2))


def analyze():
    import verify_standup_reference as verifier
    verifier.OUT=OUT;verifier.main()
    progress=read(OUT/'progress.json');folder=Path(progress['path']);plan=read(PLAN)
    assert sha(PLAN)==sha(folder/'protocol.json')
    report=read(folder/'evaluation.json');groups=defaultdict(list)
    for c in report['conditions']:
        for e in c['episodes']:
            groups[e['release_pose']].append(dict(requested=c['condition'],env_index=e['env_index'],
                recovered_and_held=e['hold_start_seconds'] is not None,hold_start_seconds=e['hold_start_seconds']))
    rows=[];gate=plan['actual_release_gate']
    for pose,entries in sorted(groups.items()):
        k=sum(e['recovered_and_held'] for e in entries);n=len(entries);p=k/n;z=1.959963984540054
        center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        rows.append(dict(actual_release_pose=pose,attempts=n,successes=k,success_fraction=p,wilson_95=[center-half,center+half],
            primary=pose in gate['poses'],gate_passed=n>=gate['minimum_episodes_each'] and p>=gate['success_fraction_min'],episodes=entries))
    passed=all(any(r['actual_release_pose']==pose and r['gate_passed'] for r in rows) for pose in gate['poses'])
    result=dict(status='passed',all_primary_conditions_passed=passed,groups=rows,
        protocol_sha256=sha(PLAN),report_sha256=sha(folder/'evaluation.json'),script_sha256=sha(Path(__file__)),
        all_proposals_retained=sum(r['attempts'] for r in rows)==256,
        learning_updates=0,scope=plan['scope'],original_protocol_rewritten=False,
        next_stage='Random-drop and transition evaluation' if passed else 'Review coverage or failure trajectories; no automatic extension')
    assert result['all_proposals_retained']
    write(OUT/'coverage-analysis.json',result)
    print(json.dumps({**result,'groups':[{k:v for k,v in row.items() if k!='episodes'} for row in rows]},indent=2))


def run():
    assert os.isatty(1),'Use the single visible VS Code terminal'
    plan=read(PLAN)
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    assert sha(core.MODEL)==plan['model_sha256']
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(PLAN)
    assert not (OUT/'progress.json').exists(),'Existing attempt retained; no automatic rerun'
    run_id='duck_getup_lateral-coverage_reference_s1002_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    output=ROOT/'records/evaluations'/run_id;output.mkdir();(output/'videos').mkdir()
    log=(output/'execution.log').open('a',buffering=1);sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    state=dict(status='running',run_id=run_id,pid=os.getpid(),path=str(output),conditions=[],
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PLAN),model_sha256=sha(core.MODEL),learning_updates=0)
    write(OUT/'progress.json',state);write(output/'evaluation.json',state)
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(output),run_id=run_id,pid=os.getpid()))
    shutil.copy2(PLAN,output/'protocol.json')
    for name in plan['source_sha256']:shutil.copy2(ROOT/name,output/Path(name).name)
    core.OUT=OUT
    try:
        core.evaluate(plan,output,state);state['status']='completed'
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error));raise
    finally:
        state['ended_at']=datetime.now(timezone.utc).isoformat()
        write(output/'evaluation.json',state);write(OUT/'progress.json',state)
        print('LATERAL_COVERAGE:',state['status'],str(output),flush=True)
    analyze()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--analyze',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.analyze:analyze()
    else:run()
