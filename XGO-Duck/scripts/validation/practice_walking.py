"""Named user practice from the confirmed fixed-controller protocol.

Create a new frozen protocol and fresh first episodes, never overwrite a
closed study or reuse its heldout. Run through the shared visible VS Code
task. This does not change weights, project goals or acceptance decisions.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import re
import shutil
import sys
import numpy as np
import yaml
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
from grouped_motion_collection import collect,prepare


def create_plan(name,seed):
    assert re.fullmatch(r'[a-z][a-z0-9_-]{1,39}',name),'Use 2-40 lowercase letters, digits, underscore or hyphen'
    assert 1100<=seed<2000,'Practice seeds: 1100..1999; final heldout 2001 is excluded'
    suite=read(ROOT/'records/reports/seven-command/suite-progress.json')
    assert suite['status']=='all-seven-functional-confirmed','Wait for the selected baseline to finish confirmation'
    origin=ROOT/'configs/experiments/seven_command/s1007.json';plan=deepcopy(read(origin))
    for path,digest in plan['source_sha256'].items():assert sha(ROOT/path)==digest,path
    for path,digest in plan['training_source_sha256'].items():assert sha(WORK/path)==digest,path
    assert sha(ROOT/plan['model'])==plan['model_sha256']
    key=f'{name}_s{seed}';file=ROOT/f'configs/experiments/user_trials/{key}.json'
    out=ROOT/f'records/reports/user-trials/{key}';assert not file.exists() and not out.exists(),'Choose a new name; previous practice is retained'
    plan.update(seed=seed,assignment_seed=70000+seed,status='frozen-before-simulation',
        frozen_at=datetime.now(timezone.utc).isoformat(),stage='user-practice',practice_name=name,
        origin_protocol=str(origin.relative_to(ROOT)),origin_protocol_sha256=sha(origin),heldout_used=False,
        scope='Named user practice of the fixed official model/controller; no stage acceptance or hardware claim')
    plan['source_sha256']['scripts/validation/practice_walking.py']=sha(Path(__file__))
    perm=np.random.default_rng(plan['assignment_seed']).permutation(224)
    plan['groups']={k:perm[i*32:(i+1)*32].tolist() for i,k in enumerate(plan['commands'])}
    actual=yaml.load(yaml.dump(asdict(prepare(plan))),Loader=yaml.BaseLoader)
    expected=read(ROOT/'records/reports/seven-command/s1007/preflight.json')['config'];expected['seed']=str(seed)
    assert actual==expected
    import torch
    assert not torch.cuda.is_initialized()
    return file,out,plan,actual


def run(name,seed):
    assert os.isatty(1),'Use the shared visible VS Code task terminal'
    file,out,plan,actual=create_plan(name,seed)
    file.parent.mkdir(exist_ok=True);out.mkdir(parents=True)
    write(file,plan);detail=out/f's{seed}';detail.mkdir()
    write(detail/'preflight.json',dict(status='passed',protocol_sha256=sha(file),config=actual,cuda_initialized=False))
    run_id=f'duck_walk_user-{name}_reference_s{seed}_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=ROOT/'records/evaluations'/run_id;folder.mkdir();(folder/'videos').mkdir()
    state=dict(status='running',run_id=run_id,path=str(folder),pid=os.getpid(),seed=seed,num_envs=32,parallel_worlds=224,
        episode_seconds=20,conditions=[],model=str(ROOT/plan['model']),model_sha256=plan['model_sha256'],
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(file),learning_updates=0,
        heldout_used=False,hardware_io=False,formal_acceptance_eligible=False,scope=plan['scope'])
    def save():write(folder/'evaluation.json',state);write(detail/'progress.json',state)
    save();shutil.copy2(file,folder/'protocol.json')
    for path in plan['source_sha256']:shutil.copy2(ROOT/path,folder/Path(path).name)
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=run_id,pid=os.getpid()))
    desired=np.zeros((224,3),np.float32);groups={k:np.asarray(v) for k,v in plan['groups'].items()}
    for condition,ids in groups.items():desired[ids]=plan['commands'][condition]
    from evaluate_standup_reference import Tee
    oldout,olderr=sys.stdout,sys.stderr
    with (folder/'execution.log').open('a',buffering=1) as log:
        sys.stdout,sys.stderr=Tee(oldout,log),Tee(olderr,log)
        try:
            print('USER_PRACTICE',name,'seed',seed,'20s x32 x7; no learning',flush=True)
            collect(plan,dict(config=actual),folder,state,save,desired,groups,224,1000)
            import verify_seven_command_confirmation as verifier
            verifier.OUT=out;analysis=verifier.verify(folder)
            state['status']='collected-and-verified';write(detail/'progress.json',state)
            print('PRACTICE_COMPLETE',json.dumps(dict(report=str(detail/'analysis.json'),functional_pass=analysis['functional_development_passed'])),flush=True)
        except BaseException as error:
            if state['status']=='running':state.update(status='failed',error=repr(error));save()
            raise
        finally:sys.stdout,sys.stderr=oldout,olderr


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--seed',required=True,type=int)
    args=parser.parse_args();run(args.name,args.seed)
