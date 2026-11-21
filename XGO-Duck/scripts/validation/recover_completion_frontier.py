"""One explicit zero-step startup repair; keep failed source/records intact.

Only inverse-inertia derived fields get an8ULP/1e-6 relative comparison;
all other captured fields still require exact equality. No field is edited.
Source declarations and actual differences are archived. Reuse the completed
50-update control; candidate still has at most50updates, no added budget.
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
import torch
import yaml
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
import completion_frontier_pilot as pilot
import roll_top_contact_pilot as core
from evaluate_roll_reset_pair import differences
from roll_model_screen import config,matched

PARENT=ROOT/'configs/experiments/roll_completion_frontier_pair.json'
PLAN=ROOT/'configs/experiments/roll_completion_frontier_startup_recovery.json'
OUT=ROOT/'records/reports/roll-completion-frontier/startup-recovery'
DERIVED={'body_invweight0','dof_invweight0'}


def compare_fields(actual,expected):
    assert set(actual)==set(expected)
    rows=[]
    for key,a in actual.items():
        b=expected[key];assert a.dtype==b.dtype and a.shape==b.shape
        assert np.isfinite(a).all() and np.isfinite(b).all()
        if key in DERIVED:
            assert a.dtype==np.float32 and np.all(a>=0) and np.all(b>=0)
            ulp=abs(a.view(np.int32).astype(np.int64)-b.view(np.int32).astype(np.int64))
            relative=abs(a.astype(float)-b.astype(float))/np.maximum(abs(b.astype(float)),1e-30)
            assert int(ulp.max(initial=0))<=8,key
            assert float(relative.max(initial=0))<=1e-6,key
            rows.append(dict(field=key,mismatches=int((a!=b).sum()),max_ulp=int(ulp.max(initial=0)),max_relative=float(relative.max(initial=0))))
        else:
            np.testing.assert_array_equal(a,b,err_msg=key)
            rows.append(dict(field=key,exact=True,mismatches=0))
    return rows


def freeze():
    assert not PLAN.exists()
    old=read(PARENT);root=ROOT/old['output'];state=read(root/'progress.json')
    assert state['status']=='stopped-review-required' and state['completed_stages']==['original-evaluation','control']
    candidate=read(ROOT/old['protocols']['candidate']);record=ROOT/'records/experiments'/candidate['run_id']
    assert read(record/'state.json')['completed_updates']==0
    assert not (record/'first-step-check.json').exists()
    if (record/'clips.json').exists():assert not read(record/'clips.json')
    assert read(record/'live-import-parity.json')['physical_steps']==0
    control=ROOT/'records/experiments'/candidate['matching_control_run']
    with np.load(record/'training-expanded-initial.npz') as a,np.load(control/'training-expanded-initial.npz') as b:
        actual={k:a[k].copy() for k in a.files};expected={k:b[k].copy() for k in b.files}
    checked=compare_fields(actual,expected)
    # Real failed input plus boundary counterexamples: do not relax other fields.
    for key,transform in (('geom_friction',lambda v:np.nextafter(v,np.inf)),
                          ('body_invweight0',lambda v:(v.view(np.int32)+9).view(np.float32))):
        bad={k:v.copy() for k,v in expected.items()};bad[key]=transform(bad[key])
        try:compare_fields(bad,expected)
        except AssertionError:pass
        else:raise AssertionError('Invalid startup comparison admitted: '+key)
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');plan=deepcopy(candidate)
    plan.update(run_id=f'duck_roll_completion-frontier-recovery_candidate_s42_{stamp}',
                startup_repair='two inverse-inertia derived fields only;8ULP and1e-6 relative; no physical mutation',
                parent_protocol_sha256=sha(PARENT),failed_record=str(record.relative_to(ROOT)))
    plan['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    cfg,_,_=pilot.prepare(plan)
    observed=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader)
    expected_config=read(root/'preflight.json')['training_configs']['candidate']
    diff=differences(expected_config,observed)
    assert len(diff)==1 and diff[0]['path']=='.agent.run_name',diff
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=False)
    master=dict(status='frozen-before-startup-recovery',stage='S4-completion-frontier-startup-recovery',
        change='completion-frontier-recovery-eval',output=str(OUT.relative_to(ROOT)),training=plan,
        parent_protocol=str(PARENT.relative_to(ROOT)),parent_protocol_sha256=sha(PARENT),
        source_sha256=plan['source_sha256'],evaluation_template=old['evaluation_template'],
        reused_results={role:str((root/f'{role}-analysis.json').relative_to(ROOT)) for role in ('original','control')},
        reused_result_sha256={role:sha(root/f'{role}-analysis.json') for role in ('original','control')},
        maximum_remaining_learning_transitions=1228800,maximum_remaining_evaluation_transitions=14400,
        candidate_updates_remaining=50,continue_gate=old['continue_gate'],heldout_used=False,
        automatic_retry=False,automatic_extension=False,
        scope='Explicit zero-physics startup comparison repair; failed protocol preserved, outcome criteria unchanged')
    write(PLAN,master)
    eval_cfg=read(root/'preflight.json')['evaluation_configs']['candidate']
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),comparison=checked,
        configuration_differences=diff,evaluation_configs={'candidate':eval_cfg},cuda_initialized=False,
        inverse_weight_source_sha256=sha(root/'inverse-weight-source.json'),
        original_fields_sha256={role:sha(folder/'training-expanded-initial.npz') for role,folder in (('control',control),('candidate',record))},
        comparison_boundaries_checked=True,no_fields_changed=True))
    file=OUT/'candidate-training.json';write(file,plan)
    print(json.dumps(dict(status='passed',protocol_sha256=sha(PLAN),run_id=plan['run_id'],comparison=checked),indent=2))


def runner_class(base,plan,record,metadata):
    parent=core.runner_class(base,plan,record,metadata)
    class StartupRecovered(parent):
        def _live_check(self,imported):
            parent._live_check(self,imported)
            env=self.env.unwrapped
            actual={k:getattr(env.sim.model,k).detach().cpu().numpy().copy() for k in sorted(env.sim.expanded_fields)}
            control=ROOT/'records/experiments'/plan['matching_control_run']/'training-expanded-initial.npz'
            with np.load(control) as old:expected={k:old[k].copy() for k in old.files}
            compared=compare_fields(actual,expected)
            np.savez_compressed(record/'training-expanded-initial.npz',**actual)
            write(record/'completion-reward-contract.json',dict(status='passed',physical_steps=0,
                arm=plan['arm'],qualified_frontier_deg=330.,terms=list(pilot.LANDING_TERMS),
                source_head_pivot_retained=True,early_rise_velocity_retained=True,
                expanded_initial_sha256=sha(record/'training-expanded-initial.npz'),
                matched_recorded_primary_fields=True,derived_comparison=compared,all_fields_bitwise_equal=False,
                no_fields_changed=True))
    return StartupRecovered


def train(planfile):
    assert os.isatty(1)
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']=='S4-completion-frontier-startup-recovery'
    planfile=Path(planfile);plan=read(planfile)
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    for name,digest in plan['environment_source_sha256'].items():assert sha(Path(name))==digest,name
    cfg,display,metadata=pilot.prepare(plan)
    record=ROOT/'records/experiments'/plan['run_id'];record.mkdir(exist_ok=False)
    write(record/'protocol.json',plan);write(record/'display-state.json',display)
    (record/'sources').mkdir()
    for name in plan['source_sha256']:shutil.copy2(ROOT/name,record/'sources'/Path(name).name)
    write(ROOT/'records/training-current.json',dict(record=str(record),protocol=str(planfile)))
    write(record/'run-event.json',dict(started_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        foreground_tty=os.isatty(1),protocol=str(planfile),protocol_sha256=sha(planfile)))
    write(record/'state.json',dict(status='starting',run_id=plan['run_id'],completed_updates=0,clips=[]))
    from evaluate_standup_reference import Tee
    with (record/'training.log').open('a',buffering=1) as output:
        oldout,olderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(oldout,output),Tee(olderr,output)
        try:
            from mjlab.scripts import train as training
            base=training.load_runner_cls(plan['task']);training.load_runner_cls=lambda task:runner_class(base,plan,record,metadata)
            training.launch_training(plan['task'],cfg)
        except BaseException as error:
            state=read(record/'state.json');state.update(status='stopped-review-required',error=repr(error));write(record/'state.json',state);raise
        finally:sys.stdout,sys.stderr=oldout,olderr


def run():
    assert os.isatty(1)
    master=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    assert sha(PARENT)==master['parent_protocol_sha256']
    for role,file in master['reused_results'].items():assert sha(ROOT/file)==master['reused_result_sha256'][role]
    assert not (OUT/'progress.json').exists()
    state=dict(status='running',pid=os.getpid(),protocol_sha256=sha(PLAN));write(OUT/'progress.json',state)
    def child(*args):subprocess.run([sys.executable,'-B',str(Path(__file__)),*map(str,args)],cwd=WORK,check=True)
    try:
        planfile=OUT/'candidate-training.json';child('--train',planfile);child('--verify-training',planfile)
        verified=read(OUT/'candidate/training-verification.json')
        p=deepcopy(master['evaluation_template']);p.update(role='candidate',source_sha256=master['source_sha256'],model=verified['model'],model_sha256=verified['model_sha256'])
        binding=dict(master,evaluations=[p]);file=OUT/'candidate-evaluation.json';write(file,binding)
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validation/roll_model_screen.py'),str(file),'--role','candidate'],cwd=WORK,check=True)
        rows={role:read(ROOT/path) for role,path in master['reused_results'].items()};rows['candidate']=read(OUT/'candidate-analysis.json')
        matched(list(rows.values()));k={r:v['successes'] for r,v in rows.items()}
        advance=bool(k['candidate']>=29 and k['candidate']>=k['control']+1 and k['candidate']>=k['original'])
        write(OUT/'analysis.json',dict(status='passed',successes=k,episodes_each=32,candidate_continue=advance,
            decision='independent-confirmation-warranted' if advance else 'close-completion-frontier-pilot',
            protocol_sha256=sha(PLAN),failed_parent_preserved=True,budget_not_increased=True,heldout_used=False))
        state['status']='completed-review-required'
    except BaseException as error:state.update(status='stopped-review-required',error=repr(error));raise
    finally:write(OUT/'progress.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--train');p.add_argument('--verify-training');a=p.parse_args()
    if a.freeze:freeze()
    elif a.train:train(a.train)
    elif a.verify_training:
        import verify_roll_top_contact_training as verifier
        verifier.OUT=OUT; (OUT/'candidate').mkdir(exist_ok=True);verifier.verify(a.verify_training)
    else:run()
