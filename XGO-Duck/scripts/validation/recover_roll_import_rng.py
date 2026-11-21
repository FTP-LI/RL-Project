"""Separate zero-step repair: preserve CPU and CUDA RNG during actor import.

The parent87-source protocol stopped before physics because its imported
actor used torch.manual_seed inside a CPU-only fork, thereby reseeding CUDA.
Keep that failed attempt immutable. Wrap ONLY the import in a complete RNG
fork; retain its unchanged numerical/physical guards, criteria and budgets.
The new protocol starts afresh, not from a failed or partially run episode.
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
import torch
import yaml
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
import roll_common_confirmation as base
import collect_roll_confirmation as collection

PARENT_PLAN=ROOT/'configs/experiments/roll_common_confirmation.json'
PARENT_OUT=ROOT/'records/reports/roll-common-confirmation'
PLAN=ROOT/'configs/experiments/roll_common_rng_recovery.json'
OUT=PARENT_OUT/'rng-recovery'
STAGE='S4-roll-rng-preserving-confirmation'
ORIGINAL_ACTOR=collection.actor_model


def preserving_actor(plan,device):
    assert torch.device(device).type=='cuda' and torch.cuda.is_initialized()
    assert torch.cuda.device_count()==1,'This repair is admitted only on the measured single-GPU host'
    with torch.random.fork_rng(devices=[0]):
        return ORIGINAL_ACTOR(plan,device)


def bind():
    # Change runtime bindings only; every parent source byte remains frozen.
    base.PLAN=PLAN;base.OUT=OUT;base.STAGE=STAGE
    collection.actor_model=preserving_actor


def freeze():
    assert not PLAN.exists() and not OUT.exists()
    parent=read(PARENT_PLAN);pre=read(PARENT_OUT/'preflight.json')
    assert sha(PARENT_PLAN)==pre['protocol_sha256']
    stopped=read(PARENT_OUT/'startup/runtime-contract.json')
    assert stopped['control_steps']==stopped['control_transitions']==stopped['physics_substeps']==0
    assert read(PARENT_OUT/'coordinator.json')['status']=='stopped-review-required'
    assert read(PARENT_OUT/'startup-progress.json')['status']=='stopped-review-required'
    assert not any((PARENT_OUT/f'{role}-progress.json').exists() for role in parent['evaluations'])
    cause=read(PARENT_OUT/'rng-root-cause.json')
    assert cause['status']=='root-cause-reproduced-cpu-only' and cause['cpu_rng_restored']
    assert cause['cuda_manual_seed_all_calls']==1 and not cause['cuda_initialized']
    for root,key in ((ROOT,'source_sha256'),(WORK,'training_source_sha256'),(Path('/'),'library_source_sha256'),(ROOT,'reference_sha256')):
        for name,expected in parent[key].items():assert sha(root/name)==expected,name
    sources=dict(parent['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    m=deepcopy(parent);stamp=datetime.now().strftime('%Y%m%dT%H%M%S')
    m.update(status='frozen-before-dynamics',stage=STAGE,stamp=stamp,source_sha256=sources,
        parent_protocol_sha256=sha(PARENT_PLAN),change='Preserve CPU and the single CUDA RNG around unchanged actor import only',
        parent_physical_steps=0,parent_dynamic_episodes=0,parent_heldout_used=False,
        seed_reuse_reason='Parent stopped atzero physical steps before all development episodes; same prospective seeds and criteria retained, not counted as independent new samples')
    configs={}
    for role,p in m['evaluations'].items():
        p.update(stage=STAGE,source_sha256=sources,
            run_id=f'duck_roll_rng-preserved_{role}_s{p["seed"]}_{stamp}')
        configs[role]=yaml.load(yaml.dump(asdict(collection.prepare(p))),Loader=yaml.BaseLoader)
        assert configs[role]==pre['evaluation_configs'][role]
        old=parent['evaluations'][role]
        assert all(p[k]==old[k] for k in ('model','model_sha256','seed','episodes','criteria','success_fraction_min','actual_joint_limit_tolerance_deg'))
    for name in ('configs/experiments/roll_common_confirmation.json',
        'records/reports/roll-common-confirmation/preflight.json',
        'records/reports/roll-common-confirmation/startup/runtime-contract.json',
        'records/reports/roll-common-confirmation/rng-root-cause.json'):
        m['reference_sha256'][name]=sha(ROOT/name)
    assert not torch.cuda.is_initialized()
    OUT.mkdir();shutil.copy2(PARENT_OUT/'cpu-scene-fields.npz',OUT/'cpu-scene-fields.npz')
    write(PLAN,m)
    fixed=deepcopy(pre);fixed.update(protocol_sha256=sha(PLAN),source_count=len(sources),evaluation_configs=configs,
        parent87sources_unchanged=True,root_cause=cause,
        repair_scope='Only import RNG fork changed; actualGPU CPU/CUDA unchanged assertion remains mandatory before physics',
        cpu_scene_fields_sha256=sha(OUT/'cpu-scene-fields.npz'),gpu_zero_step_pending=True)
    write(OUT/'preflight.json',fixed)
    print('ROLL_RNG_REPAIR_FROZEN',sha(PLAN),len(sources),'sources; parent0steps, no dynamic result',flush=True)


def run():
    assert os.isatty(1),'Use the original visible VS Code shared task terminal'
    m=base.checked();assert not (OUT/'coordinator.json').exists(),'Separate repair also cannot retry or extend'
    project=read(ROOT/'project.json');assert project['staged_training_request']['current_stage']==STAGE
    project['staged_training_request'].update(simulation_running=True,learning_running=False);write(ROOT/'project.json',project)
    state=dict(status='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),completed=[])
    write(OUT/'coordinator.json',state)
    def child(*args):
        print('VISIBLE_RNG_PRESERVED_ROLL_STAGE',*args,flush=True)
        subprocess.run([sys.executable,'-B',str(Path(__file__)),*args],cwd=WORK,check=True)
    try:
        child('--startup');state['completed'].append('gpu-zero-step');write(OUT/'coordinator.json',state)
        proofs={}
        for role in m['evaluations']:
            if role=='heldout':assert len(proofs)==2 and all(x['numeric_gate_passed'] for x in proofs.values())
            child('--evaluate',role);child('--verify',role)
            proofs[role]=read(OUT/f'{role}-analysis.json');state['completed'].append(role);write(OUT/'coordinator.json',state)
            if not proofs[role]['numeric_gate_passed']:break
        passed=len(proofs)==3 and all(x['numeric_gate_passed'] for x in proofs.values())
        result=dict(status='passed',roles=proofs,successes={k:v['successes'] for k,v in proofs.items()},
            nominal_motion_confirmation_passed=passed,actual_episodes=sum(x['episodes'] for x in proofs.values()),
            actual_control_transitions=sum(x['actual_control_transitions'] for x in proofs.values()),
            heldout_used='heldout' in proofs,learning_updates=0,hardware_io=False,automatic_extension=False,
            decision='nominal-confirmed-transitions-pending' if passed else 'closed-functional-gate-failed',
            master_sha256=sha(PLAN),parent_physical_steps=0,scope=m['scope'])
        write(OUT/'analysis.json',result);state['status']='completed-review-required'
        print('RNG_PRESERVED_ROLL_FINAL',json.dumps({k:v for k,v in result.items() if k!='roles'}),flush=True)
    except BaseException as error:
        state.update(status='stopped-review-required',error=repr(error));raise
    finally:
        state['ended_at']=datetime.now(timezone.utc).isoformat();write(OUT/'coordinator.json',state)
        project=read(ROOT/'project.json');project['staged_training_request'].update(simulation_running=False,learning_running=False);write(ROOT/'project.json',project)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--startup',action='store_true')
    parser.add_argument('--evaluate',choices=('development-a','development-b','heldout'));parser.add_argument('--verify',choices=('development-a','development-b','heldout'));a=parser.parse_args()
    if a.freeze:freeze()
    else:
        bind()
        if a.startup:base.run_one('development-a',startup=True)
        elif a.evaluate:base.run_one(a.evaluate)
        elif a.verify:
            from verify_roll_confirmation import verify
            verify(a.verify,base.checked(),PLAN,OUT)
        else:run()
