"""Bounded completion-reward hypothesis after explicit support-loss diagnosis.

Fresh equal-budget control/candidate learners capture initial randomized model
fields. The five completion-gated terms are masked only below330degrees in the
candidate; source rise velocity remains active. Same old roll acceptance rule.
No extension, hidden child, heldout selection or retry of a failed experiment.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import subprocess
import sys
import numpy as np
import torch
import yaml
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
from completion_frontier_pilot import prepare,completion_mask,LANDING_TERMS
from collect_roll_policy import config
from evaluate_roll_reset_pair import differences
from roll_model_screen import matched

PLAN=ROOT/'configs/experiments/roll_completion_frontier_pair.json'
OUT=ROOT/'records/reports/roll-completion-frontier'


def cpu_check():
    from mjlab_microduck.tasks import mdp
    angles=torch.tensor([0.,259.,260.,300.,311.51,327.12,329.36,329.999,330.,335.,360.],dtype=torch.float64)
    qualified=torch.deg2rad(angles)
    fake=SimpleNamespace(_roulade_accum=qualified.clone(),_roulade_max=qualified.clone(),
                         _roulade_paid=torch.zeros_like(qualified),_roulade_head_latch=torch.ones_like(qualified,dtype=torch.bool))
    gate=mdp._roulade_completion_gate(fake,np.deg2rad(260),np.deg2rad(330),require_head=True)
    values=torch.stack((gate,-gate))
    assert torch.equal(completion_mask(values,qualified,False),values)
    result=completion_mask(values,qualified,True)
    assert torch.equal(result[:,angles>=330],values[:,angles>=330])
    assert torch.equal(result[:,angles<330],torch.zeros_like(result[:,angles<330]))
    fake._roulade_head_latch.zero_()
    assert torch.equal(mdp._roulade_completion_gate(fake,np.deg2rad(260),np.deg2rad(330),True),torch.zeros_like(qualified))
    return dict(status='passed',angles_deg=angles.tolist(),source_gate=gate.tolist(),
                candidate_gate=result[0].tolist(),control_exact=True,positive_and_negative_values_checked=True,
                source_head_latch_preserved=True,terms=list(LANDING_TERMS),early_rise_velocity_not_masked=True)


def freeze():
    assert not PLAN.exists()
    closed=ROOT/'records/reports/roll-adaptation-screen/analysis.json'
    assert not read(closed)['adaptation_continue']
    evidence=ROOT/'records/reports/roll-support-diagnosis/analysis.json';diagnosis=read(evidence)
    assert diagnosis['status']=='passed' and diagnosis['episodes']==160
    assert diagnosis['failed_with_head_top']==21 and diagnosis['lateral_loss_in_head_top_failures_deg']['maximum']==0
    fixture=cpu_check()
    old=read(ROOT/'configs/experiments/roll_top_contact/control.json')
    for name,digest in old['source_sha256'].items():assert sha(ROOT/name)==digest,name
    sources=dict(old['source_sha256'])
    for name in ('scripts/training/completion_frontier_pilot.py','scripts/validation/completion_frontier_pair.py',
                 'scripts/validation/roll_model_screen.py','scripts/validation/analyze_roll_support_losses.py'):
        sources[name]=sha(ROOT/name)
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');plans={};cfgs={};displays={}
    for arm in ('control','candidate'):
        p=deepcopy(old)
        p.update(change='completion-frontier',arm=arm,stamp=stamp,source_sha256=sources,
                 run_id=f'duck_roll_completion-frontier_{arm}_s42_{stamp}',
                 matching_control_run=f'duck_roll_completion-frontier_control_s42_{stamp}',
                 candidate_only_change='Mask five source completion rewards below qualified330degrees; source rise velocity unchanged',
                 eval_seed=1006,support_diagnosis_sha256=sha(evidence))
        cfgs[arm],displays[arm],_=prepare(p);plans[arm]=p
    diff=differences(asdict(cfgs['control']),asdict(cfgs['candidate']))
    assert len(diff)==6 and sum('only_complete' in x['path'] for x in diff)==5
    assert sum('run_name' in x['path'] for x in diff)==1,diff
    raw=read(ROOT/'configs/experiments/roll_reference.json')
    raw.update(seed=1006,source_sha256=sources,scope='Completion-frontier development pilot; no heldout or hardware claim')
    evaluation=yaml.load(yaml.dump(asdict(config(raw))),Loader=yaml.BaseLoader)
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=False);folder=ROOT/'configs/experiments/roll_completion_frontier';folder.mkdir(exist_ok=False)
    paths={}
    for arm,p in plans.items():
        file=folder/f'{arm}.json';write(file,p);paths[arm]=str(file.relative_to(ROOT));(OUT/arm).mkdir()
    master=dict(status='frozen-before-learning',stage='S4-completion-frontier-pilot',change='completion-frontier-eval',
        output=str(OUT.relative_to(ROOT)),stamp=stamp,protocols=paths,protocol_sha256={k:sha(ROOT/v) for k,v in paths.items()},
        source_sha256=sources,training_source_sha256=old['training_source_sha256'],
        environment_source_sha256=old['environment_source_sha256'],evaluation_template=raw,
        maximum_learning_transitions=2457600,maximum_evaluation_transitions=43200,
        continue_gate='candidate>=29/32 and candidate>=control+1 and candidate>=original',
        automatic_extension=False,automatic_retry=False,heldout_used=False,
        support_diagnosis_sha256=sha(evidence),closed_previous_lead_sha256=sha(closed),
        scope='One training seed; development pilot only; all old gates and failed verdicts retained')
    write(PLAN,master)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
        training_configs={k:yaml.load(yaml.dump(asdict(c)),Loader=yaml.BaseLoader) for k,c in cfgs.items()},
        evaluation_configs={role:evaluation for role in ('original','control','candidate')},
        configuration_differences=diff,display=displays,cuda_initialized=False,reward_fixture=fixture))
    print(json.dumps(dict(status='passed',protocol_sha256=sha(PLAN),configuration_differences=diff,
                          runs={k:p['run_id'] for k,p in plans.items()},fixture=fixture),indent=2))


def child(script,*args):
    print('VISIBLE_FOREGROUND_STAGE',script,*args,flush=True)
    subprocess.run([sys.executable,'-B',str(ROOT/script),*[str(a) for a in args]],cwd=WORK,check=True)


def verify_training(planfile):
    import verify_roll_top_contact_training as base
    base.OUT=OUT
    result=base.verify(planfile)
    p=read(Path(planfile));record=ROOT/'records/experiments'/p['run_id']
    contract=read(record/'completion-reward-contract.json')
    assert contract['status']=='passed' and contract['physical_steps']==0
    assert sha(record/'training-expanded-initial.npz')==contract['expanded_initial_sha256']
    result['completion_reward_contract']=contract
    write(OUT/p['arm']/'training-verification.json',result)


def bind(master,role):
    p=deepcopy(master['evaluation_template']);p.update(role=role,master_sha256=sha(PLAN))
    if role!='original':
        verified=read(OUT/role/'training-verification.json');assert verified['status']=='passed'
        p.update(model=verified['model'],model_sha256=verified['model_sha256'])
    p['master_sha256']=sha(PLAN)
    path=OUT/f'{role}-binding.json';assert not path.exists();write(path,p)
    # The generic evaluator binds from master.evaluations, immutable per-role
    # child master below; it never alters the original frozen training master.
    child_master=dict(master,evaluations=[p],parent_master_sha256=sha(PLAN))
    file=OUT/f'{role}-evaluation.json';assert not file.exists();write(file,child_master)
    pre=read(OUT/'preflight.json')
    # Each child records its actual master hash in its own immutable protocol;
    # parent freeze fixes every physical/evaluation value before training.
    return file


def analyze():
    rows={role:read(OUT/f'{role}-analysis.json') for role in ('original','control','candidate')}
    assert all(r['status']=='passed' for r in rows.values());matched(list(rows.values()))
    k={role:r['successes'] for role,r in rows.items()}
    advance=bool(k['candidate']>=29 and k['candidate']>=k['control']+1 and k['candidate']>=k['original'])
    result=dict(status='passed',successes=k,episodes_each=32,candidate_continue=advance,
        decision='independent-confirmation-warranted' if advance else 'close-completion-frontier-pilot',
        captured_evaluation_initials_and_fields_match=True,heldout_used=False,training_seed=42,
        protocol_sha256=sha(PLAN),scope='One training seed, one matched physical development seed only')
    write(OUT/'analysis.json',result);print(json.dumps(result,indent=2))


def run():
    assert os.isatty(1)
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']=='S4-completion-frontier-pilot'
    master=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    for name,digest in master['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in master['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    for name,digest in master['environment_source_sha256'].items():assert sha(Path(name))==digest,name
    assert not (OUT/'progress.json').exists()
    state=dict(status='running',pid=os.getpid(),protocol_sha256=sha(PLAN),completed_stages=[]);write(OUT/'progress.json',state)
    try:
        child('scripts/validation/roll_model_screen.py',bind(master,'original'),'--role','original')
        state['completed_stages'].append('original-evaluation');write(OUT/'progress.json',state)
        for arm in ('control','candidate'):
            file=ROOT/master['protocols'][arm];assert sha(file)==master['protocol_sha256'][arm]
            child('scripts/training/completion_frontier_pilot.py',file)
            child('scripts/validation/completion_frontier_pair.py','--verify-training',file)
            child('scripts/validation/roll_model_screen.py',bind(master,arm),'--role',arm)
            state['completed_stages'].append(arm);write(OUT/'progress.json',state)
        analyze();state['status']='completed-review-required'
    except BaseException as error:state.update(status='stopped-review-required',error=repr(error));raise
    finally:write(OUT/'progress.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--verify-training');p.add_argument('--analyze',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.verify_training:verify_training(a.verify_training)
    elif a.analyze:analyze()
    else:run()
