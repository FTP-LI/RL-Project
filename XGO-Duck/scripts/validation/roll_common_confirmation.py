"""Bounded independent confirmation of the fixed official roll, visibly.

CPU admission reuses saved physical data, compares complete compiled model
fields and freezes the exact sources. The visible coordinator then requires
a real GPU zero-step check, two development seeds and only conditionally one
heldout seed. Stop at the first failed motion gate; never retry or extend.
Nine-second first episodes and independent original scoring are preserved.
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
import traceback
import numpy as np
import torch
import yaml
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
from common_scene_contract import model_arrays,compare_arrays,physical_description
from inspect_common_skill_plant import digest
from collect_roll_confirmation import prepare,model_input,collect

PLAN=ROOT/'configs/experiments/roll_common_confirmation.json'
OUT=ROOT/'records/reports/roll-common-confirmation'
STAGE='S4-official-roll-confirmation'


def cpu_components(parent,draft):
    from verify_roll_reference import independent
    from roll_metrics import summarize
    from verify_roll_metrics import trace as synthetic
    proof=read(OUT/'cpu-import-check.json')
    assert proof['status']=='passed-cpu-only' and not proof['cuda_initialized']
    assert proof['script_sha256']==sha(ROOT/'scripts/validation/check_roll_batched_import.py')
    assert proof['draft_sha256']==sha(ROOT/'configs/experiments/roll_common_confirmation_draft.json')
    source=ROOT/proof['source'];assert sha(source)==proof['source_sha256']
    report=read(source.parent/'evaluation.json')
    with np.load(source) as f:t={k:f[k].copy() for k in f.files}
    a=summarize(t,draft['criteria'])
    b=independent(t,draft['criteria'],np.asarray(report['joint_limits']),np.deg2rad(parent['actual_joint_limit_tolerance_deg']))
    assert a['successes']==sum(x['success'] for x in b)==30
    assert [x['success'] for x in a['episodes']]==[x['success'] for x in b]
    with np.load(OUT/'cpu-import-inputs.npz') as f:adapted=f['observations'].copy()
    original=adapted.copy();sim=np.asarray(report['simulation_home'],np.float32);home=np.asarray(report['policy_home'],np.float32)
    original[:,6:20]-=sim-home;original[:,34:48]=0
    rebuilt=model_input(torch.from_numpy(original),torch.from_numpy(sim),torch.from_numpy(home),torch.from_numpy(adapted[:,34:48]))
    np.testing.assert_allclose(rebuilt.numpy(),adapted,atol=1e-6,rtol=0)
    np.testing.assert_array_equal(original[:,34:48],0)
    cases=[]
    for name,fixture,expected in [('forward',synthetic(),True),('backward',synthetic(direction=-1),False),
        ('exact-deadline',synthetic(finish=7.02),True),('late-deadline',synthetic(finish=7.04),False)]:
        fixture['joint_positions']=np.zeros((451,1,14))
        left=summarize(fixture,draft['criteria'])['successes']
        right=sum(x['success'] for x in independent(fixture,draft['criteria'],np.tile([-2.,2.],(14,1)),0.))
        assert bool(left)==bool(right)==expected,(name,left,right)
        cases.append(dict(case=name,success=expected))
    return dict(status='passed-cpu-only',saved_episodes_reconstructed=32,saved_successes=30,
        actual_adapter_inputs=512,input_not_mutated=True,metric_boundary_cases=cases,
        physics_steps=0,learning_updates=0,cuda_initialized=False)


def freeze():
    from mjlab.scene import Scene
    assert not PLAN.exists() and not (OUT/'preflight.json').exists(),'Do not overwrite any attempt'
    draftpath=ROOT/'configs/experiments/roll_common_confirmation_draft.json';d=read(draftpath)
    parentpath=ROOT/d['parent_protocol'];assert sha(parentpath)==d['parent_sha256'];parent=read(parentpath)
    assert d['episodes_per_seed']==128 and d['development_seeds']==[1020,1021] and d['heldout_seed']==2020
    assert d['criteria']==parent['criteria'] and d['success_fraction_min']==parent['success_fraction_min']==.9
    assert d['minimum_successes_each']==int(np.ceil(.9*128))==116
    assert sha(ROOT/d['model'])==d['model_sha256']==parent['model_sha256']
    components=cpu_components(parent,d)
    ancestry=read(ROOT/'configs/experiments/getup_early_guard_study.json')
    sources=dict(ancestry['source_sha256']);sources.update(parent['source_sha256'])
    for name in ('roll_common_confirmation.py','collect_roll_confirmation.py','verify_roll_confirmation.py','check_roll_batched_import.py'):
        sources['scripts/validation/'+name]=sha(ROOT/'scripts/validation'/name)
    sources['configs/environments/action_recording.json']=sha(ROOT/'configs/environments/action_recording.json')
    training_sources=parent['training_source_sha256']
    libraries=ancestry['arms']['control']['library_source_sha256']
    for root,files in ((ROOT,sources),(WORK,training_sources),(Path('/'),libraries)):
        for name,expected in files.items():assert sha(root/name)==expected,name
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');evaluations={};configs={};fingerprints={}
    for role,seed in [('development-a',1020),('development-b',1021),('heldout',2020)]:
        p=deepcopy(parent);p.update(status='frozen-before-simulation',stage=STAGE,role=role,seed=seed,episodes=128,
            heldout_used=role=='heldout',run_id=f'duck_roll_common-confirm_{role}_s{seed}_{stamp}',
            source_sha256=sources,training_source_sha256=training_sources,library_source_sha256=libraries,
            scope='Independent nominal fixed-official roll; shared-noise transition/robustness remain separate S5 work')
        cfg=prepare(p);assert not cfg.auto_reset and cfg.episode_length_s==9 and cfg.scene.num_envs==128
        fingerprints[role]=digest(physical_description(cfg))
        configs[role]=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader);evaluations[role]=p
    assert len(set(fingerprints.values()))==1
    fields=model_arrays(Scene(prepare(evaluations['development-a']).scene,device='cpu').compile())
    with np.load(ROOT/'records/reports/common-plant-screen/roll-cpu-scene-fields.npz') as f:
        compare_arrays(fields,{k:f[k] for k in f.files})
    assert not torch.cuda.is_initialized()
    reference_paths=['records/reports/common-plant-screen/roll-cpu-scene-fields.npz',
        'records/reports/common-plant-screen/startup/roll/scene-static-fields.npz',
        'records/reports/common-plant-screen/startup/roll/nominal-primary-fields.npz',
        'records/reports/roll-common-confirmation/cpu-import-check.json',
        'records/reports/roll-common-confirmation/cpu-import-inputs.npz',
        'configs/experiments/common_plant/roll.json','configs/experiments/roll_common_confirmation_draft.json']
    m=dict(status='frozen-before-dynamics',stage=STAGE,stamp=stamp,evaluations=evaluations,source_sha256=sources,
        training_source_sha256=training_sources,library_source_sha256=libraries,
        reference_sha256={name:sha(ROOT/name) for name in reference_paths},minimum_successes_each=116,
        episodes_each=128,maximum_episodes=384,maximum_control_transitions=172800,learning_updates=0,
        startup_steps=0,heldout_only_after_both_development_pass=True,stop_at_first_failed_seed=True,
        automatic_retry=False,automatic_extension=False,hardware_io=False,full_goal_preserved=True,
        scope=d['scope'],recording=d['recording'])
    np.savez_compressed(OUT/'cpu-scene-fields.npz',**fields)
    write(PLAN,m);write(OUT/'preflight.json',dict(status='passed-cpu-only',protocol_sha256=sha(PLAN),
        cpu_components=components,evaluation_configs=configs,physical_fingerprints=fingerprints,
        full_static_fields_exact=True,cuda_initialized=False,source_count=len(sources),
        cpu_scene_fields_sha256=sha(OUT/'cpu-scene-fields.npz')))
    print('ROLL_CONFIRM_FROZEN',sha(PLAN),len(sources),'sources; no physical steps',flush=True)


def checked():
    m=read(PLAN);pre=read(OUT/'preflight.json');assert sha(PLAN)==pre['protocol_sha256']
    for root,key in ((ROOT,'source_sha256'),(WORK,'training_source_sha256'),(Path('/'),'library_source_sha256'),(ROOT,'reference_sha256')):
        for name,expected in m[key].items():assert sha(root/name)==expected,name
    assert sha(OUT/'cpu-scene-fields.npz')==pre['cpu_scene_fields_sha256']
    for p in m['evaluations'].values():assert sha(ROOT/p['model'])==p['model_sha256']
    return m


def run_one(role,startup=False):
    from evaluate_standup_reference import Tee
    assert os.isatty(1),'Use the original visible VS Code shared task terminal'
    m=checked();p=m['evaluations'][role]
    if not startup:
        assert read(OUT/'startup-progress.json')['status']=='completed-zero-step'
        assert read(OUT/'startup/runtime-contract.json')['control_steps']==0
        required=[] if role=='development-a' else ['development-a']
        if role=='heldout':required.append('development-b')
        for previous in required:
            proof=read(OUT/f'{previous}-analysis.json')
            assert proof['numeric_gate_passed'] and proof['master_sha256']==sha(PLAN)
    folder=OUT/'startup' if startup else ROOT/'records/evaluations'/p['run_id']
    folder.mkdir(exist_ok=False);(folder/'videos').mkdir()
    write(folder/'protocol.json',p)
    for name in m['source_sha256']:
        target=folder/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    state=dict(status='running',run_id=p['run_id'],role=role,seed=p['seed'],path=str(folder),pid=os.getpid(),
        argv=sys.argv,foreground_tty=os.isatty(1),started_at=datetime.now(timezone.utc).isoformat(),
        master_sha256=sha(PLAN),protocol_sha256=sha(folder/'protocol.json'),model_sha256=p['model_sha256'],
        learning_updates=0,heldout_used=p['heldout_used'],hardware_io=False)
    progress=OUT/('startup-progress.json' if startup else f'{role}-progress.json')
    assert not progress.exists()
    def save():write(folder/'evaluation.json',state);write(progress,state)
    save()
    if not startup:write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),pid=os.getpid(),run_id=p['run_id']))
    with (folder/'execution.log').open('x',buffering=1) as log:
        oldout,olderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(oldout,log),Tee(olderr,log)
        try:
            print('VISIBLE_ROLL_CONFIRM_COMMAND',json.dumps(state),flush=True)
            collect(p,folder,state,save,startup=startup)
            state['status']='completed-zero-step' if startup else 'completed'
        except BaseException as error:
            state.update(status='stopped-review-required',error=repr(error));traceback.print_exc();raise
        finally:
            state['ended_at']=datetime.now(timezone.utc).isoformat();save();sys.stdout,sys.stderr=oldout,olderr
    if startup:
        assert read(folder/'runtime-contract.json')['control_steps']==0
        print('ROLL_CONFIRM_GPU_ZERO_STEP_PASSED',flush=True)


def run():
    assert os.isatty(1),'Use the original visible VS Code shared task terminal'
    m=checked();assert not (OUT/'coordinator.json').exists(),'No retry or extension'
    project=read(ROOT/'project.json');assert project['staged_training_request']['current_stage']==STAGE
    project['staged_training_request'].update(simulation_running=True,learning_running=False);write(ROOT/'project.json',project)
    state=dict(status='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),completed=[])
    write(OUT/'coordinator.json',state)
    def child(*args):
        print('VISIBLE_ROLL_STAGE',*args,flush=True)
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
            master_sha256=sha(PLAN),scope=m['scope'])
        write(OUT/'analysis.json',result);state['status']='completed-review-required'
        print('ROLL_CONFIRM_FINAL',json.dumps({k:v for k,v in result.items() if k!='roles'}),flush=True)
    except BaseException as error:
        state.update(status='stopped-review-required',error=repr(error));raise
    finally:
        state['ended_at']=datetime.now(timezone.utc).isoformat();write(OUT/'coordinator.json',state)
        project=read(ROOT/'project.json');project['staged_training_request'].update(simulation_running=False,learning_running=False);write(ROOT/'project.json',project)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--startup',action='store_true')
    parser.add_argument('--evaluate',choices=('development-a','development-b','heldout'));parser.add_argument('--verify',choices=('development-a','development-b','heldout'));a=parser.parse_args()
    if a.freeze:freeze()
    elif a.startup:run_one('development-a',startup=True)
    elif a.evaluate:run_one(a.evaluate)
    elif a.verify:
        from verify_roll_confirmation import verify
        verify(a.verify,checked(),PLAN,OUT)
    else:run()
