"""Prospectively freeze and run a bounded actor-angle access ablation.

No old failed run is resumed. Both new arms share the official angle task,
phase-aware critic, rewards, reset and fixed budget. Only actor column55
gradient access differs. Evaluate on fresh development seed1008 with the
original contact/rotation/landing criteria; no automatic extension or retry.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import numpy as np
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
from metadata_actor_import import import_actor
from angle_feature_pilot import prepare,mask_angle_gradient
from inspect_roll_angle_inputs import neutralize_slot,gravity_phase
from roll_angle_observer import GravityPhase
from collect_roll_policy import config
from evaluate_roll_reset_pair import differences
from recover_completion_frontier import compare_fields

PLAN=ROOT/'configs/experiments/roll_angle_feature_pair.json'
OUT=ROOT/'records/reports/roll-angle-feature'


def cpu_check(model,digest):
    torch.set_num_threads(2);torch.manual_seed(0)
    source=read(ROOT/'records/reports/roll-action-memory/startup-recovery/control/progress.json')
    with np.load(Path(source['path'])/'forward-roll-trace.npz') as f:full=f['observations'].copy()
    phi=gravity_phase(full[:,:,3:6]);observer=GravityPhase()
    causal=np.asarray([observer.update(g) for g in full[:,:,3:6]])
    np.testing.assert_allclose(causal,phi,atol=2e-6,rtol=0)
    full[:,:,55]=phi.astype(np.float32)
    obs=full[::30].reshape(-1,61)[::4].copy()
    td=TensorDict({'actor':torch.from_numpy(obs)},batch_size=[len(obs)])
    actors={}
    for arm in ('control','candidate'):
        actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,
            hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
            distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'scalar'})
        import_actor(actor,model,digest);neutralize_slot(actor);actor.eval()
        if arm=='control':actor.mlp[0].weight.register_hook(mask_angle_gradient)
        for p in actor.parameters():p.requires_grad_(False)
        for p in actor.parameters():p.requires_grad_(True)
        actor(td).square().mean().backward();actors[arm]=actor
    control,candidate=actors['control'],actors['candidate']
    assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in control.state_dict().items())
    cg,ng=control.mlp[0].weight.grad,candidate.mlp[0].weight.grad
    assert not torch.count_nonzero(cg[:,55]).item() and torch.count_nonzero(ng[:,55]).item()>0
    keep=torch.arange(61)!=55
    torch.testing.assert_close(cg[:,keep],ng[:,keep],atol=0,rtol=0)
    assert torch.count_nonzero(cg[:,keep]).item()>0 and not torch.cuda.is_initialized()
    return dict(status='passed',hook_survives_actor_freeze_unfreeze=True,initial_states_exact=True,
        control_angle_gradient_exact_zero=True,candidate_angle_gradient_nonzero=int(torch.count_nonzero(ng[:,55])),
        other_input_gradients_exact=True,causal_phase_max_error_rad=float(abs(causal-phi).max()),
        real_probe_inputs=len(obs),optimizer_steps=0,physics_steps=0,cuda_initialized=False)


def freeze():
    assert os.isatty(1), 'Run CPU preflight in the existing visible task terminal'
    assert not PLAN.exists() and not OUT.exists()
    old=read(ROOT/'configs/experiments/roll_top_contact/control.json')
    for name,digest in old['source_sha256'].items():assert sha(ROOT/name)==digest,name
    audit=ROOT/'records/reports/roll-angle-input/source-contract.json'
    assert read(audit)['status']=='passed'
    draft=ROOT/'configs/experiments/roll_angle_feature_draft.json'
    assert read(draft)['status']=='draft-not-executable'
    fixture=cpu_check(ROOT/old['model'],old['model_sha256'])
    sources=dict(old['source_sha256'])
    for name in ('scripts/training/angle_feature_pilot.py','scripts/validation/angle_feature_pair.py',
        'scripts/validation/collect_roll_angle.py','scripts/validation/evaluate_roll_angle.py',
        'scripts/validation/verify_angle_feature_training.py','scripts/validation/roll_angle_observer.py',
        'scripts/validation/inspect_roll_angle_inputs.py','scripts/validation/recover_completion_frontier.py',
        'scripts/training/completion_frontier_pilot.py','scripts/validation/roll_model_screen.py'):
        sources[name]=sha(ROOT/name)
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');plans={};cfgs={};displays={}
    for arm in ('control','candidate'):
        p=deepcopy(old)
        p.update(change='angle-feature',task='Mjlab-Roulade-Angle-Flat-XgoDuck',arm=arm,stamp=stamp,
            source_sha256=sources,run_id=f'duck_roll_angle-feature_{arm}_s42_{stamp}',
            matching_control_run=f'duck_roll_angle-feature_control_s42_{stamp}',
            candidate_only_change='Actor angle column learnable; control zero-column gradient-masked',
            common_change='Both official Angle task, fresh phase-aware critic; angle normalization explicit0/1',
            eval_seed=1008,angle_audit_sha256=sha(audit),draft_sha256=sha(draft),
            extra_learned_policy_clip_common_step=700)
        cfgs[arm],displays[arm],_=prepare(p);plans[arm]=p
    diff=differences(asdict(cfgs['control']),asdict(cfgs['candidate']))
    assert len(diff)==1 and diff[0]['path']=='.agent.run_name',diff
    raw=read(ROOT/'configs/experiments/roll_reference.json')
    raw.update(seed=1008,source_sha256=sources,scope='Actor-angle access development pilot; no heldout or hardware claim')
    evaluation=yaml.load(yaml.dump(asdict(config(raw))),Loader=yaml.BaseLoader)
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=False);folder=ROOT/'configs/experiments/roll_angle_feature';folder.mkdir(exist_ok=False)
    paths={}
    for arm,p in plans.items():
        file=folder/f'{arm}.json';write(file,p);paths[arm]=str(file.relative_to(ROOT));(OUT/arm).mkdir()
    master=dict(status='frozen-before-learning',stage='S4-angle-feature-pilot',change='angle-feature-eval',
        output=str(OUT.relative_to(ROOT)),stamp=stamp,protocols=paths,
        protocol_sha256={k:sha(ROOT/v) for k,v in paths.items()},source_sha256=sources,
        training_source_sha256=old['training_source_sha256'],environment_source_sha256=old['environment_source_sha256'],
        evaluation_template=raw,maximum_learning_transitions=2457600,maximum_evaluation_transitions=43200,
        updates_each=50,critic_only_updates=14,actor_updates=36,
        continue_gate='candidate>=29/32 and candidate>=control+1 and candidate>=original',
        automatic_retry=False,automatic_extension=False,heldout_used=False,hardware_io=False,
        angle_audit_sha256=sha(audit),draft_sha256=sha(draft),
        physical_parameter_comparison='Same0physics instant; primary exact, inverse-weight derived two only8ULP and relative1e-6',
        angle_observer='Causal unwrap of observed projected gravity; zero at roll trigger',
        attribution='Actor access only, within shared phase-aware critic and common reparameterization; not exact source-critic comparison')
    write(PLAN,master)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
        training_configs={k:yaml.load(yaml.dump(asdict(c)),Loader=yaml.BaseLoader) for k,c in cfgs.items()},
        evaluation_configs={role:evaluation for role in ('original','control','candidate')},
        configuration_differences=diff,display=displays,cuda_initialized=False,gradient_and_observer_fixture=fixture))
    print(json.dumps(dict(status='passed',protocol_sha256=sha(PLAN),configuration_differences=diff,
        runs={k:p['run_id'] for k,p in plans.items()},fixture=fixture),indent=2),flush=True)


def child(script,*args):
    print('VISIBLE_FOREGROUND_STAGE',script,*args,flush=True)
    subprocess.run([sys.executable,'-B',str(ROOT/script),*[str(a) for a in args]],cwd=WORK,check=True)


def bind(master,role):
    p=deepcopy(master['evaluation_template']);p.update(role=role,master_sha256=sha(PLAN))
    if role!='original':
        checked=read(OUT/role/'training-verification.json');assert checked['status']=='passed'
        p.update(model=checked['model'],model_sha256=checked['model_sha256'])
    file=OUT/f'{role}-evaluation.json';assert not file.exists()
    write(file,dict(master,evaluations=[p],parent_master_sha256=sha(PLAN)))
    return file


def analyze():
    rows={r:read(OUT/f'{r}-analysis.json') for r in ('original','control','candidate')}
    assert all(r['status']=='passed' and r['observed_gravity_phase_verified'] for r in rows.values())
    values=[]
    for r in rows.values():
        folder=ROOT/r['evaluator']
        with np.load(folder/'forward-roll-trace.npz') as t:initial={k:t[k].copy() for k in ('initial_qpos','initial_qvel')}
        with np.load(folder/'expanded-model-initial.npz') as f:fields={k:f[k].copy() for k in f.files}
        values.append((initial,fields))
    checks=[]
    for initial,fields in values[1:]:
        for k in initial:np.testing.assert_array_equal(initial[k],values[0][0][k])
        checks.append(compare_fields(fields,values[0][1]))
    k={r:v['successes'] for r,v in rows.items()}
    advance=bool(k['candidate']>=29 and k['candidate']>=k['control']+1 and k['candidate']>=k['original'])
    pairs=list(zip(rows['control']['rows'],rows['candidate']['rows']))
    result=dict(status='passed',successes=k,episodes_each=32,candidate_continue=advance,
        decision='independent-confirmation-warranted' if advance else 'close-angle-feature-pilot',
        paired_candidate_wins=sum(not a['success'] and b['success'] for a,b in pairs),
        paired_candidate_losses=sum(a['success'] and not b['success'] for a,b in pairs),
        evaluation_initial_physical_states_exact=True,same_time_field_comparison=checks,
        training_seed=42,physical_seed=1008,heldout_used=False,protocol_sha256=sha(PLAN),
        scope='One training seed and one matched development seed; shared phase-aware critic in both arms')
    write(OUT/'analysis.json',result);print(json.dumps(result,indent=2),flush=True)


def run():
    assert os.isatty(1)
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']=='S4-angle-feature-pilot'
    master=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    for name,digest in master['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in master['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    for name,digest in master['environment_source_sha256'].items():assert sha(Path(name))==digest,name
    assert not (OUT/'progress.json').exists()
    state=dict(status='running',pid=os.getpid(),protocol_sha256=sha(PLAN),completed_stages=[]);write(OUT/'progress.json',state)
    try:
        child('scripts/validation/evaluate_roll_angle.py',bind(master,'original'),'--role','original')
        state['completed_stages'].append('original-evaluation');write(OUT/'progress.json',state)
        for arm in ('control','candidate'):
            file=ROOT/master['protocols'][arm];assert sha(file)==master['protocol_sha256'][arm]
            child('scripts/training/angle_feature_pilot.py',file)
            child('scripts/validation/verify_angle_feature_training.py',file)
            child('scripts/validation/evaluate_roll_angle.py',bind(master,arm),'--role',arm)
            state['completed_stages'].append(arm);write(OUT/'progress.json',state)
        analyze();state['status']='completed-review-required'
    except BaseException as error:state.update(status='stopped-review-required',error=repr(error));raise
    finally:write(OUT/'progress.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--analyze',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.analyze:analyze()
    else:run()
