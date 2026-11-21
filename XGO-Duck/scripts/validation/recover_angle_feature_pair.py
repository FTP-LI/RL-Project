"""Explicit zero-step guard repair; same pair budget, source evaluation reused.

The source task samples forward commands at48..50 despite zero range bounds.
The failed guard treated all command slots as padding. Admit those three
official inputs; preserve every model input and all non-padding checks. A new
frozen protocol retains the failed attempt and cannot extend either50budget.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import argparse
import ast
import json
import os
import shutil
import sys
import numpy as np
import torch
import yaml
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
from evaluate_roll_reset_pair import differences
import angle_feature_pair as parent
import angle_feature_startup_recovery as pilot

PARENT=parent.PLAN
PLAN=ROOT/'configs/experiments/roll_angle_feature_startup_recovery.json'
OUT=ROOT/'records/reports/roll-angle-feature/startup-recovery'


def functions(path):
    tree=ast.parse(path.read_text())
    return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}


def freeze():
    assert os.isatty(1) and not PLAN.exists() and not OUT.exists()
    old=read(PARENT);root=ROOT/old['output'];progress=read(root/'progress.json')
    assert progress['status']=='stopped-review-required' and progress['completed_stages']==['original-evaluation']
    failed=read(ROOT/old['protocols']['control']);record=ROOT/'records/experiments'/failed['run_id']
    state=read(record/'state.json');assert state['completed_updates']==0 and not state['clips']
    assert 'Mismatched elements: 5 / 416' in state['error']
    assert not (record/'first-step-check.json').exists() and not (record/'live-import-parity.json').exists()
    assert 'Learning iteration' not in (record/'training.log').read_text()
    assert not (ROOT/'records/experiments'/read(ROOT/old['protocols']['candidate'])['run_id']).exists()
    for name,digest in old['source_sha256'].items():assert sha(ROOT/name)==digest,name
    # Check the actual code delta, including all nested runner checks.
    original=functions(ROOT/'scripts/training/angle_feature_pilot.py')
    repaired=functions(ROOT/'scripts/training/angle_feature_startup_recovery.py')
    first=ast.unparse(original['runner_class'])
    before='np.testing.assert_allclose(reference_obs[:, 48:], 0, atol=1e-07, rtol=0)'
    assert first.count(before)==1
    assert first.replace(before,'assert_startup_padding(reference_obs)')==ast.unparse(repaired['runner_class'])
    assert ast.unparse(original['run']).replace('S4-angle-feature-pilot','S4-angle-feature-startup-recovery')==ast.unparse(repaired['run'])
    fixture=np.zeros((32,61),np.float32);fixture[[4,13,24,25,31],48]=.3
    try:np.testing.assert_allclose(fixture[:,48:],0,atol=1e-7,rtol=0)
    except AssertionError:pass
    else:raise AssertionError('Old guard should reproduce the observed command failure')
    pilot.assert_startup_padding(fixture)
    for slot in (51,56,60):
        bad=fixture.copy();bad[0,slot]=.01
        try:pilot.assert_startup_padding(bad)
        except AssertionError:pass
        else:raise AssertionError('Padding corruption admitted')
    pre=read(root/'preflight.json')
    assert pre['training_configs']['control']['env']['commands']['twist']['rel_forward_envs']=='0.2'
    sources=dict(old['source_sha256'])
    for name in ('scripts/training/angle_feature_startup_recovery.py','scripts/validation/recover_angle_feature_pair.py'):
        sources[name]=sha(ROOT/name)
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');paths={};configs={};diffs={}
    OUT.mkdir(exist_ok=False);folder=ROOT/'configs/experiments/roll_angle_feature/startup-recovery';folder.mkdir()
    for arm in ('control','candidate'):
        p=deepcopy(read(ROOT/old['protocols'][arm]))
        p.update(run_id=f'duck_roll_angle-feature-recovery_{arm}_s42_{stamp}',
            matching_control_run=f'duck_roll_angle-feature-recovery_control_s42_{stamp}',source_sha256=sources,
            startup_repair='Only zero-padding assertion starts51 instead48;48..50 are official sampled commands',
            parent_protocol_sha256=sha(PARENT))
        cfg,_,_=pilot.prepare(p);observed=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader)
        diff=differences(pre['training_configs'][arm],observed)
        assert len(diff)==1 and diff[0]['path']=='.agent.run_name',diff
        configs[arm]=observed;diffs[arm]=diff
        file=folder/f'{arm}.json';write(file,p);paths[arm]=str(file.relative_to(ROOT));(OUT/arm).mkdir()
    master=dict(old,status='frozen-before-zero-step-recovery',stage='S4-angle-feature-startup-recovery',
        change='angle-feature-recovery-eval',output=str(OUT.relative_to(ROOT)),protocols=paths,
        protocol_sha256={k:sha(ROOT/v) for k,v in paths.items()},source_sha256=sources,
        parent_protocol=str(PARENT.relative_to(ROOT)),parent_protocol_sha256=sha(PARENT),
        reused_original=str((root/'original-analysis.json').relative_to(ROOT)),
        reused_original_sha256=sha(root/'original-analysis.json'),maximum_remaining_learning_transitions=2457600,
        maximum_remaining_evaluation_transitions=28800,failed_attempt_updates=0,budget_not_increased=True,
        failed_record=str(record.relative_to(ROOT)),failed_state_sha256=sha(record/'state.json'))
    master['evaluation_template']=dict(old['evaluation_template'],source_sha256=sources)
    write(PLAN,master)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),training_configs=configs,
        evaluation_configs=pre['evaluation_configs'],configuration_differences=diffs,
        only_startup_guard_and_stage_name_changed=True,source_forward_probability=.2,
        recorded_failure_reproduced=True,padding_counterexamples_rejected=True,
        failed_updates=0,maximum_updates_each=50,no_model_or_physics_mutation=True,cuda_initialized=False))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='passed',protocol_sha256=sha(PLAN),failed_updates=0,
        runs=paths,only_startup_guard_changed=True,budget_not_increased=True),indent=2),flush=True)


def run():
    assert os.isatty(1)
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']=='S4-angle-feature-startup-recovery'
    master=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    assert sha(PARENT)==master['parent_protocol_sha256']
    assert sha(ROOT/master['reused_original'])==master['reused_original_sha256']
    for name,digest in master['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in master['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    for name,digest in master['environment_source_sha256'].items():assert sha(Path(name))==digest,name
    assert not (OUT/'progress.json').exists()
    parent.PLAN=PLAN;parent.OUT=OUT
    shutil.copy2(ROOT/master['reused_original'],OUT/'original-analysis.json')
    state=dict(status='running',pid=os.getpid(),protocol_sha256=sha(PLAN),completed_stages=['original-reused'])
    write(OUT/'progress.json',state)
    try:
        for arm in ('control','candidate'):
            file=ROOT/master['protocols'][arm];assert sha(file)==master['protocol_sha256'][arm]
            parent.child('scripts/training/angle_feature_startup_recovery.py',file)
            parent.child('scripts/validation/recover_angle_feature_pair.py','--verify-training',file)
            parent.child('scripts/validation/evaluate_roll_angle.py',parent.bind(master,arm),'--role',arm)
            state['completed_stages'].append(arm);write(OUT/'progress.json',state)
        parent.analyze();state['status']='completed-review-required'
    except BaseException as error:state.update(status='stopped-review-required',error=repr(error));raise
    finally:write(OUT/'progress.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--verify-training');p.add_argument('--analyze',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.verify_training:
        import verify_angle_feature_training as verifier
        verifier.OUT=OUT;verifier.verify(a.verify_training)
    elif a.analyze:parent.PLAN=PLAN;parent.OUT=OUT;parent.analyze()
    else:run()
