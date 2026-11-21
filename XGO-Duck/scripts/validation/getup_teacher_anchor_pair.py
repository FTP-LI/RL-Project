"""Freeze and visibly execute one bounded successful-state anchor comparison.

Keep the admitted fixed critic and fresh actor Adam. Coefficient0 is the PPO
control; coefficient0.01 is the only candidate change. Candidate budget is
the completed control budget, never a chosen best checkpoint. Per-update
drift tripwires close each arm; only equal-budget, safe, functionally passing
results can become a development lead. Independent confirmation comes later.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,subprocess,sys
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import WORK,read,sha
from record_io import write
from getup_teacher_anchor_train import prepare as training_prepare,kinematic_stable
from getup_neck_home_order_study import prepare as evaluation_prepare,make_memory,startup as parent_eval_startup
from getup_late_feasibility_pair import decision,candidate_admitted,cpu_selection_checks

PLAN=ROOT/'configs/experiments/getup_teacher_anchor_pair.json'
OUT=ROOT/'records/reports/getup-teacher-anchor-pair'
STAGE='S3-teacher-anchor-execution'


def freeze():
    import mujoco
    from mjlab.scene import Scene
    from common_scene_contract import model_arrays,compare_arrays
    from evaluate_roll_reset_pair import differences
    from check_getup_neck_home_order import check as processor_check
    from getup_release_coverage import cpu_check
    assert not PLAN.exists() and not OUT.exists()
    value_path=ROOT/'records/reports/getup-anchor-value-check/scale-recovery/analysis.json'
    value=read(value_path)
    assert value['status']=='passed' and value['value_gate_passed'] and value['actor_and_critic_bitwise_unchanged']
    protocol=ROOT/'configs/experiments/getup_anchor_scale_recovery.json'
    assert value['protocol_sha256']==sha(protocol)
    base=read(protocol);value_record=ROOT/value['record']
    assert sha(value_record/'value-data.npz')==value['dataset_sha256']
    ppo_path=ROOT/'records/reports/getup-teacher-anchor/ppo-preflight.json';ppo=read(ppo_path)
    assert ppo['zero_coefficient_actor_critic_and_adam_bitwise_equal'] and ppo['positive_coefficient_changes_only_actor']
    assert ppo['ppo_sha256']==sha(ROOT/'scripts/training/getup_anchor_ppo.py')
    draft_path=ROOT/'configs/experiments/getup_teacher_anchor_draft.json';draft=read(draft_path)
    old=read(ROOT/'configs/experiments/getup_late_feasibility_pair.json')
    plan=deepcopy(old['training_arms']['control'])
    plan.update(seed=43,split=base['split'],split_seed=base['split_seed'],reward_scale=base['reward_scale'],
        probe_dataset=str((value_record/'value-data.npz').relative_to(ROOT)),probe_sha256=value['dataset_sha256'],
        anchor_data='records/reports/getup-teacher-anchor/anchor-data.npz',
        anchor_cells='records/reports/getup-teacher-anchor/cells.json',anchor_seed=54301,
        tripwire=deepcopy(draft['tripwire']),stage=STAGE,training_updates=50,
        auxiliary_weight=0.,scope='One bounded training-only mean-anchor PPO pilot; separate functional evaluation required')
    plan.pop('scale_recovery',None)
    plan['learner']['minibatch_seed']=5543
    assert plan['initial_std']==plan['learner']['initial_std']==.1
    preparation=read(ROOT/'records/reports/getup-teacher-anchor/preparation.json')
    assert sha(ROOT/plan['anchor_data'])==preparation['data_sha256']
    assert sha(ROOT/plan['anchor_cells'])==preparation['cells_sha256']
    warm=torch.load(ROOT/plan['critic_checkpoint'],map_location='cpu',weights_only=True)
    assert warm['reward_scale']==plan['reward_scale'] and sha(ROOT/plan['critic_checkpoint'])==plan['critic_sha256']
    cfg,_,meta=training_prepare(plan)
    model=Scene(cfg.env.scene,device='cpu').compile();fields=model_arrays(model)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+n) for n in meta['joint_names'].split(',')]
    addresses=model.jnt_qposadr[ids].tolist();ranges=model.jnt_range[ids].copy()
    np.testing.assert_allclose(ranges,plan['joint_limits'],atol=1e-7,rtol=0)
    with np.load(ROOT/'records/reports/getup-anchor-value-check/scale-recovery/cpu-scene-fields.npz') as f:
        compare_arrays(fields,{k:f[k] for k in f.files})
    with np.load(value_record/'value-data.npz') as f:
        watch_ids=[i for i in range(32) if i in base['split']['train_ids']]
        assert watch_ids and not set(watch_ids)&set(base['split']['validation_ids'])
        reference=float(kinematic_stable(torch.from_numpy(f['qpos'][349:,watch_ids]),torch.from_numpy(f['qvel'][349:,watch_ids]),
            addresses,torch.tensor(ranges)).float().mean())
    plan.update(qpos_addresses=addresses,watch_reference_stable_fraction=reference,
        watch_reference_worlds=len(watch_ids),watch_reference_indices=watch_ids,
        watch_stable_min=max(.5,reference-.25),forced_video_boundaries=[0,800],video_seconds=8.)
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');training={}
    for role,weight in [('control',0.),('candidate',.01)]:
        p=deepcopy(plan);p.update(arm=role,anchor_weight=weight,
            run_id=f'duck_getup_success-anchor_{role}_s43_{stamp}',
            matching_control=f'duck_getup_success-anchor_control_s43_{stamp}')
        training[role]=p
    assert not differences(asdict(training_prepare(training['control'])[0]),asdict(training_prepare(training['candidate'])[0]))
    seed=1030
    # Every evaluation is a new first episode; a draft mentioning this seed
    # does not consume it. Reject an existing actual run before freezing.
    for folder in ('records/evaluations','records/experiments'):
        assert not any(f'_s{seed}_' in x.name for x in (ROOT/folder).iterdir())
    template=deepcopy(old['arms']['original'])
    codes=np.random.default_rng(seed).permutation(np.repeat(np.arange(4),[32,32,128,128]))
    names=('prone','supine','left-side','right-side')
    groups={name:np.flatnonzero(codes==i).tolist() for i,name in enumerate(names)}
    template.update(seed=seed,split=dict(pose_codes=codes.tolist(),groups=groups),
        video_worlds={name:groups[name][0] for name in names},video_title='SUCCESS ANCHOR',progress_label='ANCHOR_EVAL',
        handoff_enabled=False,neck_home_enabled=False)
    arms={}
    for role in ('control','original','candidate'):
        p=deepcopy(template);p.update(arm=role,run_id=f'duck_getup_success-anchor-eval_{role}_s{seed}_{stamp}')
        arms[role]=p
    ecfg,_,_=evaluation_prepare(arms['original'])
    compare_arrays(fields,model_arrays(Scene(ecfg.env.scene,device='cpu').compile()))
    processor=processor_check(arms['original'],ecfg)
    numeric=cpu_check(dict(roles={'getup':template,'standing':dict(template,model=template['standing_model'],model_sha256=template['standing_model_sha256'])}))
    sources=deepcopy(base['source_sha256'])
    for name in ('scripts/training/getup_teacher_anchor_train.py','scripts/validation/getup_teacher_anchor_pair.py',
                 'scripts/validation/verify_getup_teacher_anchor_train.py','scripts/validation/verify_getup_teacher_anchor_eval.py'):
        sources[name]=sha(ROOT/name)
    assert len({Path(name).name for name in sources})==len(sources),'Collector snapshots require unique basenames'
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name
    for p in [*training.values(),*arms.values()]:p['source_sha256']=sources
    assert not torch.cuda.is_initialized()
    refs={str(path.relative_to(ROOT)):sha(path) for path in
        (value_path,protocol,ppo_path,draft_path,ROOT/plan['anchor_data'],ROOT/plan['anchor_cells'])}
    m=dict(status='frozen-before-learning',stage=STAGE,stamp=stamp,training_arms=training,arms=arms,
        source_sha256=sources,reference_sha256=refs,training_seed=43,evaluation_seed=seed,
        maximum_learning_transitions=2457600,maximum_evaluation_transitions=528000,
        heldout_used=False,hardware_io=False,automatic_retry=False,automatic_extension=False,
        candidate_budget_rule='Use control verified completed_updates; no later checkpoint choice. An earlier candidate stop is unmatched and cannot be selected.',
        selection_rule='Original and control selection use unchanged functional gates. Candidate must use equal actual updates, avoid drift tripwire, pass requested/actual/timely/final95percent gates and exceed control by2finalholds.',
        scope='One training-only successful-state mean anchor; independent development episodes only; fullS2-S6 remain open')
    OUT.mkdir();(OUT/'control').mkdir();(OUT/'candidate').mkdir()
    np.savez_compressed(OUT/'cpu-scene-fields.npz',**fields);write(PLAN,m)
    write(OUT/'preflight.json',dict(status='passed-cpu-only',protocol_sha256=sha(PLAN),source_count=len(sources),
        identical_training_configs=True,full_physical_fields_exact=True,processor=processor,model_import=numeric,
        selection=cpu_selection_checks(),watch_reference=reference,watch_min=plan['watch_stable_min'],
        checkpoint_scale_exact=True,cuda_initialized=False,
        training_config=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader)))
    print('SUCCESS_ANCHOR_PAIR_FROZEN',sha(PLAN),len(sources),'sources;max50/arm;perupdate drift stop',flush=True)


def checked(resolve_models=True):
    m=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in m['reference_sha256'].items():assert sha(ROOT/name)==digest,name
    for p in m['training_arms'].values():
        for key in ('model','critic_checkpoint','probe_dataset'):
            digest={'model':'model_sha256','critic_checkpoint':'critic_sha256','probe_dataset':'probe_sha256'}[key]
            assert sha(ROOT/p[key])==p[digest]
        for base,key in ((WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
            for name,digest in p[key].items():assert sha(base/name)==digest,name
    control_proof=OUT/'control/training-verification.json'
    if control_proof.exists():
        proof=read(control_proof);assert proof['status']=='passed'
        m['training_arms']['candidate']['training_updates']=proof['rollout_updates']
    if resolve_models:
        for role in ('control','candidate'):
            artifact=ROOT/'artifacts'/m['training_arms'][role]['run_id']
            if not artifact.exists():continue
            proof=read(OUT/role/'training-verification.json');assert proof['status']=='passed'
            assert sha(artifact/'policy.onnx')==proof['model_sha256']
            assert sha(ROOT/proof['record']/'model-final.pt')==proof['checkpoint_sha256']
            m['arms'][role].update(model=str((artifact/'policy.onnx').relative_to(ROOT)),model_sha256=proof['model_sha256'])
    return m


def evaluate(role):
    import collect_fixed_recovery as collector
    from common_scene_contract import model_arrays,compare_arrays
    m=checked();p=m['arms'][role]
    if role!='original':assert p['model']!=m['arms']['original']['model']
    original=collector.actor_model;proofs=[]
    def preserving_model(plan,device):
        cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state(device).clone()
        with torch.random.fork_rng(devices=[0]):result=original(plan,device)
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state(device))
        proofs.append(dict(model_sha256=plan['model_sha256'],cpu_cuda_rng_unchanged=True))
        return result
    def startup(env,obs,actors,plan,record):
        parent_eval_startup(env,obs,actors,plan,record)
        assert len(proofs)==2 and all(x['cpu_cuda_rng_unchanged'] for x in proofs)
        with np.load(OUT/'cpu-scene-fields.npz') as f:compare_arrays(model_arrays(env.sim.mj_model),{k:f[k] for k in f.files})
        proof=read(record/'startup.json');proof.update(both_model_import_rng_unchanged=True,new_frozen_cpu_scene_exact=True)
        write(record/'startup.json',proof);write(record/'model-import-rng.json',proofs)
    collector.actor_model=preserving_model
    collector.collect(role,m,plan_path=PLAN,out=OUT,prepare=evaluation_prepare,make_memory=make_memory,startup_check=startup)


def project_state(**values):
    p=read(ROOT/'project.json');p['staged_training_request'].update(values);write(ROOT/'project.json',p)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    for name in ('train','verify-train'):parser.add_argument('--'+name,choices=('control','candidate'))
    for name in ('evaluate','verify-eval'):parser.add_argument('--'+name,choices=('control','original','candidate'))
    a=parser.parse_args()
    if a.freeze:freeze();return
    if a.train:
        from getup_teacher_anchor_train import run
        run(checked(False)['training_arms'][a.train],PLAN,OUT);return
    if a.verify_train:
        from verify_getup_teacher_anchor_train import verify
        verify(checked(False)['training_arms'][a.verify_train],OUT);return
    if a.evaluate:evaluate(a.evaluate);return
    if a.verify_eval:
        from verify_getup_teacher_anchor_eval import verify
        verify(a.verify_eval);return
    assert os.isatty(1),'Use the original visible VS Code shared terminal'
    checked(False);assert not (OUT/'coordinator.json').exists(),'No retry or extension'
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']==STAGE
    write(OUT/'coordinator.json',dict(status='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat()))
    def step(*args):subprocess.run([sys.executable,'-B',str(Path(__file__)),*args],cwd=WORK,check=True)
    try:
        project_state(learning_running=True,simulation_running=True)
        step('--train','control');step('--verify-train','control');project_state(learning_running=False)
        for role in ('control','original'):step('--evaluate',role);step('--verify-eval',role)
        control,official=[read(OUT/f'{role}-analysis.json') for role in ('control','original')]
        choice=decision(control,official);write(OUT/'selection.json',choice)
        if choice['action']=='candidate':
            project_state(learning_running=True);step('--train','candidate');step('--verify-train','candidate')
            project_state(learning_running=False);step('--evaluate','candidate');step('--verify-eval','candidate')
            cp=read(OUT/'control/training-verification.json');ap=read(OUT/'candidate/training-verification.json')
            matched=cp['rollout_updates']==ap['rollout_updates']
            admitted=matched and not ap['drift_tripwire_triggered'] and candidate_admitted(control,read(OUT/'candidate-analysis.json'))
            choice=dict(action='select' if admitted else 'close',selected='candidate' if admitted else None,
                candidate_admitted=admitted,actual_budgets_matched=matched,
                required_successes=choice['required_successes'],reason='Frozen functional, relative, equal-budget and drift gates')
            write(OUT/'selection.json',choice)
        proofs={role:read(OUT/f'{role}-analysis.json') for role in ('control','original','candidate') if (OUT/f'{role}-analysis.json').exists()}
        training={role:read(OUT/role/'training-verification.json') for role in ('control','candidate') if (OUT/role/'training-verification.json').exists()}
        result=dict(status='passed-integrity',selection=choice,
            timely_counts={role:p['overall']['timely_holds'] for role,p in proofs.items()},
            final_hold_counts={role:p['end_hold']['overall']['timely_holds'] for role,p in proofs.items()},
            actual_learning_transitions=sum(p['actual_control_transitions'] for p in training.values()),
            actual_evaluation_transitions=len(proofs)*176000,protocol_sha256=sha(PLAN),
            heldout_used=False,hardware_io=False,automatic_extension=False,scope='Bounded development comparison; no reliability or fullgoal completion claim')
        write(OUT/'analysis.json',result)
        write(OUT/'coordinator.json',dict(status='completed-review-required',pid=os.getpid(),finished_at=datetime.now(timezone.utc).isoformat()))
        print('SUCCESS_ANCHOR_PAIR_CLOSED',json.dumps(result),flush=True)
    except BaseException as error:
        write(OUT/'coordinator.json',dict(status='stopped-review-required',pid=os.getpid(),error=repr(error)));raise
    finally:project_state(learning_running=False,simulation_running=False)


if __name__=='__main__':main()
