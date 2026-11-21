"""Freeze and visibly coordinate a bounded single-factor roll-learning pilot.

All child processes inherit this one terminal and finish sequentially. The
three model screens retain the original roll rule. A model hash is bound only
after export verification, while physics/seed/budget were frozen beforehand.
Stop on integrity or phase-watch failure; never silently extend or retry.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import inspect
import json
import os
import subprocess
import sys
import numpy as np
import onnx
import torch
import yaml
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
from roll_top_contact_pilot import prepare
from collect_roll_policy import config as evaluation_config
from evaluate_roll_reset_pair import differences

PLAN=ROOT/'configs/experiments/roll_top_contact_pair.json'
OUT=ROOT/'records/reports/roll-top-contact'


def freeze():
    assert not PLAN.exists()
    check=read(OUT/'input-checks.json');assert check['status']=='passed'
    assert check['importer_sha256']==sha(ROOT/'scripts/training/metadata_actor_import.py')
    assert check['adapter_sha256']==sha(ROOT/'scripts/training/roll_top_contact.py')
    draft=read(ROOT/'configs/experiments/roll_top_contact_draft.json');base=read(ROOT/'configs/experiments/roll_reference.json')
    model=ROOT/base['model'];metadata={x.key:x.value for x in onnx.load(model).metadata_props}
    assert sha(model)==check['model_sha256']==base['model_sha256']
    prior=read(ROOT/'records/reports/roll-action-memory/startup-recovery/control/progress.json')
    simulation_home=read(Path(prior['path'])/'evaluation.json')['simulation_home']
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');plans={};cfgs={};display={}
    training=['metadata_actor_import.py','roll_top_contact.py','roll_top_contact_pilot.py','training_watch.py',
        'live_observer.py','display_preferences.py','record_io.py','guarded_training.py']
    validation=['roll_top_contact_pair.py','verify_roll_top_contact_inputs.py','verify_roll_top_contact_training.py',
        'evaluate_roll_top_contact.py','collect_roll_policy.py','evaluate_roll_reference.py','verify_roll_reference.py',
        'roll_metrics.py','evaluate_standup_reference.py','standup_metrics.py','s1_reference_inputs.py','s1_observer_integration.py','evaluate_fixed_policy.py']
    sources={'scripts/training/'+f:sha(ROOT/'scripts/training'/f) for f in training}
    sources.update({'scripts/validation/'+f:sha(ROOT/'scripts/validation'/f) for f in validation})
    sources['configs/environments/training_display.json']=sha(ROOT/'configs/environments/training_display.json')
    for path,digest in base['training_source_sha256'].items():assert sha(WORK/path)==digest,path
    for arm in ('control','candidate'):
        p=deepcopy(draft);p.update(status='frozen',arm=arm,stamp=stamp,
            run_id=f'duck_roll_top-contact_{arm}_s42_{stamp}',model=base['model'],model_sha256=base['model_sha256'],
            model_metadata=metadata,simulation_home=simulation_home,source_sha256=sources,
            training_source_sha256=base['training_source_sha256'],input_check_sha256=sha(OUT/'input-checks.json'))
        p['online_nominal_height_m']=base['criteria']['nominal_height_m']
        p['drift_probe_trace_sha256']=sha(Path(prior['path'])/'forward-roll-trace.npz')
        p['matching_control_run']=f'duck_roll_top-contact_control_s42_{stamp}'
        cfgs[arm],display[arm],_=prepare(p);plans[arm]=p
    diff=differences(asdict(cfgs['control']),asdict(cfgs['candidate']))
    assert len(diff)==2 and any('only_top' in d['path'] for d in diff) and any('run_name' in d['path'] for d in diff),diff
    raw=deepcopy(base);raw.update(seed=1004,episodes=32,role='original')
    eval_cfg=yaml.load(yaml.dump(asdict(evaluation_config(raw))),Loader=yaml.BaseLoader)
    from mjlab.scripts import train
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    classes=[train.load_runner_cls(draft['task']),MjlabOnPolicyRunner,PPO,MLPModel]
    environment_sources={str(Path(inspect.getfile(c)).resolve()):sha(Path(inspect.getfile(c))) for c in classes}
    assert not torch.cuda.is_initialized()
    folder=ROOT/'configs/experiments/roll_top_contact';folder.mkdir(exist_ok=True);paths={}
    for arm,p in plans.items():
        p['environment_source_sha256']=environment_sources
        file=folder/f'{arm}.json';assert not file.exists();write(file,p);paths[arm]=str(file.relative_to(ROOT))
        (OUT/arm).mkdir(exist_ok=True)
    master=dict(status='frozen-before-learning',stage='S4-top-contact-pilot',task=draft['task'],stamp=stamp,
        protocols=paths,protocol_sha256={k:sha(ROOT/v) for k,v in paths.items()},source_sha256=sources,
        training_source_sha256=base['training_source_sha256'],environment_source_sha256=environment_sources,
        evaluation_template=raw,maximum_learning_transitions=2457600,maximum_evaluation_transitions=43200,
        continue_gate=draft['continue_gate'],automatic_retry=False,automatic_extension=False,
        live_checks_required=['actor parity before physics','source reset HOME','raw history after first step','candidate matched initial state'],
        scope='Single training-seed development pilot; no heldout, three-seed robustness or hardware claim')
    write(PLAN,master);write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
        training_configs={k:yaml.load(yaml.dump(asdict(c)),Loader=yaml.BaseLoader) for k,c in cfgs.items()},
        evaluation_config=eval_cfg,configuration_differences=diff,display=display,cuda_initialized=False))
    print(json.dumps(dict(status='passed',protocol_sha256=sha(PLAN),runs={k:p['run_id'] for k,p in plans.items()},configuration_differences=diff),indent=2))


def binding(master,role):
    p=deepcopy(master['evaluation_template']);p.update(role=role,source_sha256=master['source_sha256'],frozen_master_sha256=sha(PLAN))
    if role!='original':
        verified=read(OUT/role/'training-verification.json');assert verified['status']=='passed'
        p.update(model=verified['model'],model_sha256=verified['model_sha256'],
            training_verification_sha256=sha(OUT/role/'training-verification.json'))
    assert sha(ROOT/p['model'])==p['model_sha256']
    directory=OUT/'evaluation-bindings';directory.mkdir(exist_ok=True);file=directory/f'{role}.json'
    assert not file.exists();write(file,p);return file


def child(script,*args):
    command=[sys.executable,'-B',str(ROOT/script),*[str(a) for a in args]]
    print('VISIBLE_FOREGROUND_STAGE',script,*args,flush=True)
    subprocess.run(command,cwd=WORK,check=True)


def analyze():
    rows={role:read(OUT/role/'evaluation-analysis.json') for role in ('original','control','candidate')}
    for role in rows:assert rows[role]['status']=='passed'
    arrays={};fields={}
    for role,r in rows.items():
        folder=ROOT/r['evaluator']
        with np.load(folder/'forward-roll-trace.npz') as t:arrays[role]=(t['initial_qpos'].copy(),t['initial_qvel'].copy(),t['observations'][1].copy())
        with np.load(folder/'expanded-model-initial.npz') as m:fields[role]={k:m[k].copy() for k in m.files}
    for role in ('control','candidate'):
        for a,b in zip(arrays['original'],arrays[role]):np.testing.assert_allclose(a,b,atol=1e-7,rtol=0)
        assert set(fields[role])==set(fields['original'])
        for key in fields[role]:np.testing.assert_array_equal(fields[role][key],fields['original'][key])
    k={role:r['successes'] for role,r in rows.items()}
    advance=bool(k['candidate']>=29 and k['candidate']>=k['control']+1 and k['candidate']>=k['original'])
    result=dict(status='passed',successes=k,episodes_each=32,candidate_continue=advance,
        matching_physical_initials_and_first_inputs=True,matching_captured_model_fields=True,
        decision='independent-confirmation-warranted' if advance else 'close-top-contact-pilot',
        heldout_used=False,scope='One training seed and matched development scenarios only',script_sha256=sha(Path(__file__)))
    write(OUT/'analysis.json',result);print(json.dumps(result,indent=2))


def run():
    assert os.isatty(1) and read(ROOT/'project.json')['staged_training_request']['current_stage']=='S4-top-contact-pilot'
    master=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    assert not (OUT/'progress.json').exists()
    for source,digest in master['source_sha256'].items():assert sha(ROOT/source)==digest,source
    for source,digest in master['environment_source_sha256'].items():assert sha(Path(source))==digest,source
    state=dict(status='running',protocol_sha256=sha(PLAN),completed_stages=[],pid=os.getpid());write(OUT/'progress.json',state)
    try:
        child('scripts/validation/evaluate_roll_top_contact.py',binding(master,'original'))
        state['completed_stages'].append('original-evaluation');write(OUT/'progress.json',state)
        for arm in ('control','candidate'):
            file=ROOT/master['protocols'][arm];assert sha(file)==master['protocol_sha256'][arm]
            child('scripts/training/roll_top_contact_pilot.py',file)
            child('scripts/validation/verify_roll_top_contact_training.py',file)
            child('scripts/validation/evaluate_roll_top_contact.py',binding(master,arm))
            state['completed_stages'].append(arm);write(OUT/'progress.json',state)
        analyze();state['status']='completed-review-required';write(OUT/'progress.json',state)
    except BaseException as error:state.update(status='stopped-review-required',error=repr(error));write(OUT/'progress.json',state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    freeze() if args.freeze else run()
