"""One fixed-model roll output-filter comparison after the failed pose trial.

Both arms use source reset, full contacts, model-HOME conversion and raw
policy history. Both compute .15-old/.85-new memory, but only candidate sends
it. Original 90% contact/rotation/landing gate stays unchanged. No learning,
automatic retry or extra budget; this is not the archived runtime's shared
HOME interface or timed skill switch.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import numpy as np
import yaml
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
from collect_roll_action_memory import evaluate,config

PLAN=ROOT/'configs/experiments/roll_action_memory_pair.json'
OUT=ROOT/'records/reports/roll-action-memory'


def freeze():
    import torch
    assert not PLAN.exists()
    review=read(OUT/'prior-action-review.json')
    assert review['source_successes']==28 and review['checks']==dict(numeric_recurrence=True,selective_reset=True,nonfinite_rejected=True)
    assert review['implementation_sha256']==sha(ROOT/'scripts/validation/roll_action_memory.py')
    base=read(ROOT/'configs/experiments/roll_reference.json');plans={};actual={}
    additions=['roll_action_memory_pair.py','collect_roll_action_memory.py','verify_roll_action_memory.py','roll_action_memory.py']
    for arm in ('control','candidate'):
        p=deepcopy(base);p.update(arm=arm,seed=1003,frozen_at=datetime.now(timezone.utc).isoformat(),
            output_memory_old_coefficient=.15,apply_memory=arm=='candidate',
            scope='Development output-filter comparison only; metadata HOME preserved; not full runtime/shared-HOME or skill switching')
        p['source_sha256'].update({'scripts/validation/'+f:sha(ROOT/'scripts/validation'/f) for f in additions})
        p['source_sha256']['scripts/validation/s1_reference_inputs.py']=sha(ROOT/'scripts/validation/s1_reference_inputs.py')
        for source,digest in p['training_source_sha256'].items():assert sha(WORK/source)==digest
        assert sha(ROOT/p['model'])==p['model_sha256']
        actual[arm]=yaml.load(yaml.dump(asdict(config(p))),Loader=yaml.BaseLoader);plans[arm]=p
    assert actual['control']==actual['candidate'] and not torch.cuda.is_initialized()
    folder=ROOT/'configs/experiments/roll_action_memory';folder.mkdir(exist_ok=True);paths={}
    for arm,p in plans.items():
        file=folder/f'{arm}.json';assert not file.exists();write(file,p);paths[arm]=str(file.relative_to(ROOT))
    master=dict(status='frozen-before-simulation',stage='S4-action-memory-pair',seed=1003,episodes_each=32,
        protocols=paths,arm_protocol_sha256={k:sha(ROOT/v) for k,v in paths.items()},
        maximum_control_steps=900,maximum_transitions=28800,learning_updates=0,heldout_used=False,
        automatic_retry=False,automatic_extension=False,
        changed_behavior='Only candidate applies .15 old + .85 new output memory; both compute it; raw history and all physical config unchanged',
        continue_gate=dict(candidate_success_fraction_min=.9,minimum_additional_successes=1),
        prior_review_sha256=sha(OUT/'prior-action-review.json'),
        initial_match_rule='Compare qpos/qvel and captured expanded model fields; do not require trajectories to coincide')
    write(PLAN,master);write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
        arm_protocol_sha256=master['arm_protocol_sha256'],config=actual['control'],physical_configs_identical=True,cuda_initialized=False))
    print(json.dumps(dict(status='passed',protocol_sha256=sha(PLAN),episodes_each=32,seed=1003),indent=2))


def run_arm(master,arm):
    out=OUT/arm;out.mkdir(exist_ok=True);assert not (out/'progress.json').exists()
    file=ROOT/master['protocols'][arm];assert sha(file)==master['arm_protocol_sha256'][arm];plan=read(file)
    for source,digest in plan['source_sha256'].items():assert sha(ROOT/source)==digest,source
    for source,digest in plan['training_source_sha256'].items():assert sha(WORK/source)==digest,source
    assert sha(ROOT/plan['model'])==plan['model_sha256']
    run_id=f'duck_roll_action-memory_{arm}_s1003_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=ROOT/'records/evaluations'/run_id;folder.mkdir();(folder/'videos').mkdir()
    state=dict(status='running',run_id=run_id,pid=os.getpid(),path=str(folder),arm=arm,conditions=[],
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(file),
        model=str(ROOT/plan['model']),model_sha256=plan['model_sha256'],learning_updates=0)
    def save():write(folder/'evaluation.json',state);write(out/'progress.json',state)
    save();shutil.copy2(file,folder/'protocol.json')
    for source in plan['source_sha256']:shutil.copy2(ROOT/source,folder/Path(source).name)
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=run_id,pid=os.getpid()))
    from evaluate_standup_reference import Tee
    oldout,olderr=sys.stdout,sys.stderr
    with (folder/'execution.log').open('a',buffering=1) as log:
        sys.stdout,sys.stderr=Tee(oldout,log),Tee(olderr,log)
        try:evaluate(plan,folder,state);state['status']='completed'
        except BaseException as error:state.update(status='failed',error=repr(error));raise
        finally:state['ended_at']=datetime.now(timezone.utc).isoformat();save();sys.stdout,sys.stderr=oldout,olderr
    from verify_roll_action_memory import verify
    return verify(arm)


def analyze():
    a,b=read(OUT/'control/analysis.json'),read(OUT/'candidate/analysis.json')
    starts={};models={};first={}
    for arm in ('control','candidate'):
        p=read(OUT/arm/'progress.json');folder=Path(p['path'])
        with np.load(folder/'forward-roll-trace.npz') as t:
            starts[arm]=(t['initial_qpos'].copy(),t['initial_qvel'].copy());first[arm]=t['observations'][1].copy()
        with np.load(folder/'expanded-model-initial.npz') as m:models[arm]={k:m[k].copy() for k in m.files}
    for x,y in zip(starts['control'],starts['candidate']):np.testing.assert_allclose(x,y,atol=1e-7,rtol=0)
    np.testing.assert_allclose(first['control'],first['candidate'],atol=1e-7,rtol=0)
    assert set(models['control'])==set(models['candidate'])
    for key in models['control']:np.testing.assert_array_equal(models['control'][key],models['candidate'][key])
    sa=np.asarray([r['success'] for r in a['rows']]);sb=np.asarray([r['success'] for r in b['rows']])
    advance=bool(b['numeric_gate_passed'] and b['successes']>=a['successes']+1)
    result=dict(status='passed',control_successes=a['successes'],candidate_successes=b['successes'],episodes_each=32,
        candidate_continue=advance,wins=int((~sa&sb).sum()),losses=int((sa&~sb).sum()),
        initial_state_match=True,expanded_model_fields_match=True,first_observations_match=True,
        captured_fields=list(models['control']),control_analysis_sha256=sha(OUT/'control/analysis.json'),
        candidate_analysis_sha256=sha(OUT/'candidate/analysis.json'),learning_updates=0,
        decision='independent-confirmation-warranted' if advance else 'close-output-memory-hypothesis',
        scope='Fixed-model development comparison; original failures preserved; hidden solver-state parity not claimed')
    write(OUT/'analysis.json',result);print(json.dumps(result,indent=2));return result


def run():
    assert os.isatty(1) and read(ROOT/'project.json')['staged_training_request']['current_stage']=='S4-action-memory-pair'
    master=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256'] and not (OUT/'pair-progress.json').exists()
    progress=dict(status='running',protocol_sha256=sha(PLAN),completed_arms=[],pid=os.getpid());write(OUT/'pair-progress.json',progress)
    try:
        for arm in ('control','candidate'):run_arm(master,arm);progress['completed_arms'].append(arm);write(OUT/'pair-progress.json',progress)
        analyze();progress['status']='completed-review-required';write(OUT/'pair-progress.json',progress)
    except BaseException as error:progress.update(status='stopped-review-required',error=repr(error));write(OUT/'pair-progress.json',progress);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    freeze() if args.freeze else run()
