"""Freeze and visibly coordinate one new bounded recovery learning pair.

Control and candidate differ only in the late feasibility actor-loss weight.
Use the admitted matched-interface critic, reachable selection logic and a
fresh development seed. Each evaluation is320first11s episodes; preserve
original timing/strict scores and the final3s holding gate. No hardware,
resampling, closed-protocol restart, automatic retry or budget extension.
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
from getup_late_feasibility import prepare as training_prepare,kinematic_stable,masks
from getup_neck_home_order_study import prepare as evaluation_prepare,make_memory,startup as evaluation_startup

PLAN=ROOT/'configs/experiments/getup_late_feasibility_pair.json'
OUT=ROOT/'records/reports/getup-late-feasibility'


def covered(proof):
    return proof['other_release_episodes']==0 and all(x['coverage_passed'] for x in proof['actual'].values())


def decision(control,official):
    assert control['status']==official['status']=='passed'
    if not covered(control) or not covered(official):return dict(action='close',reason='Insufficient actual release coverage; no resampling')
    if official['all_motion_gates_passed']:return dict(action='select',selected='original',reason='Simplest fixed policy passes')
    if control['all_motion_gates_passed']:return dict(action='select',selected='control',reason='Common PPO control passes all gates')
    score=control['end_hold']['overall'];needed=score['timely_holds']+2
    if needed>score['episodes']:return dict(action='close',reason='Candidate improvement requirement is unreachable',required_successes=needed)
    return dict(action='candidate',required_successes=needed,reason='Covered control lacks functional gates; candidate budget is feasible')


def candidate_admitted(control,candidate):
    return bool(covered(candidate) and candidate['all_motion_gates_passed'] and
        candidate['end_hold']['overall']['timely_holds']>=control['end_hold']['overall']['timely_holds']+2)


def cpu_selection_checks():
    def proof(k,passed=False):
        return dict(status='passed',other_release_episodes=0,actual={name:dict(coverage_passed=True) for name in range(4)},
            all_motion_gates_passed=passed,end_hold=dict(overall=dict(episodes=320,timely_holds=k)))
    assert decision(proof(300),proof(304,True))['selected']=='original'
    assert decision(proof(304,True),proof(300))['selected']=='control'
    assert decision(proof(318),proof(300))['required_successes']==320
    assert decision(proof(319),proof(300))['action']=='close'
    bad=proof(300);bad['actual'][0]['coverage_passed']=False
    assert decision(bad,proof(300))['action']=='close'
    assert candidate_admitted(proof(318),proof(320,True))
    assert not candidate_admitted(proof(318),proof(319,True))
    assert not candidate_admitted(proof(300),proof(320,False))
    a,b=masks(torch.tensor([0,99,100,249,250,399,0]))
    assert a.tolist()==[False,False,True,True,True,True,False]
    assert b.tolist()==[False,False,False,False,True,True,False]
    return dict(status='passed',reachable_ceiling_checked=True,insufficient_coverage_stops=True,
        original_and_final_functional_gates_required=True,freeze_release_late_and_reset_masks=True,physical_steps=0)


def freeze():
    import mujoco
    from mjlab.scene import Scene
    from common_scene_contract import model_arrays,compare_arrays
    from evaluate_roll_reset_pair import differences
    from check_getup_neck_home_order import check as processor_check
    from getup_release_coverage import cpu_check
    assert not PLAN.exists() and not OUT.exists(),'Keep frozen attempts immutable'
    admission=read(ROOT/'records/reports/getup-interface-value/readback-recovery/analysis.json')
    assert admission['status']=='passed' and admission['value_gate_passed'] and admission['actor_bitwise_unchanged']
    record=ROOT/admission['record'];base=read(record/'protocol.json')
    assert sha(record/'critic-final.pt')==admission['critic_sha256'] and sha(record/'value-data.npz')==admission['dataset_sha256']
    draft=read(ROOT/'configs/experiments/getup_late_feasibility_draft.json')
    aux=read(ROOT/'records/reports/getup-target-feasibility/auxiliary-preflight.json')
    assert aux['status']=='passed-cpu-only' and aux['positive_auxiliary_changes_only_actor']
    for name,digest in aux['source_sha256'].items():assert sha(ROOT/name)==digest,name
    plan=dict(base);plan.update(critic_checkpoint=str((record/'critic-final.pt').relative_to(ROOT)),
        critic_sha256=sha(record/'critic-final.pt'),probe_dataset=str((record/'value-data.npz').relative_to(ROOT)),
        probe_sha256=sha(record/'value-data.npz'),reward_scale=admission['reward_scale'],
        learner=dict(actor_lr=2e-5,critic_lr=3e-4,minibatch_seed=5542,initial_std=.10,gamma=.99,
            **{'lambda':.95},epochs=5,minibatches=4,clip=.2,gradient_norm=1.,target_kl=.02,severe_kl_stop=.1),
        stage='S3-late-feasibility-learning',training_updates=50,control_steps=1200,
        maximum_control_transitions=1228800,learning_updates=50,
        scope='One bounded50-update whole-body feasibility pilot, matched source/critic; no final reliability or hardware claim')
    cfg,_,meta=training_prepare(plan);model=Scene(cfg.env.scene,device='cpu').compile();fields=model_arrays(model)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+n) for n in meta['joint_names'].split(',')]
    addresses=model.jnt_qposadr[ids].tolist();ranges=model.jnt_range[ids].copy()
    np.testing.assert_allclose(ranges,plan['joint_limits'],atol=1e-7,rtol=0)
    with np.load(ROOT/'records/reports/getup-interface-value/readback-recovery/cpu-scene-fields.npz') as f:
        compare_arrays(fields,{k:f[k] for k in f.files})
    with np.load(record/'value-data.npz') as f:
        watch_ids=[i for i in range(32) if i in base['split']['train_ids']]
        assert watch_ids and not set(watch_ids)&set(base['split']['validation_ids'])
        reference=float(kinematic_stable(torch.from_numpy(f['qpos'][349:,watch_ids]),torch.from_numpy(f['qvel'][349:,watch_ids]),
            addresses,torch.tensor(ranges)).float().mean())
    plan.update(qpos_addresses=addresses,watch_reference_stable_fraction=reference,watch_reference_worlds=len(watch_ids),
        watch_reference_indices=watch_ids,watch_stable_min=max(.5,reference-.25),
        watch_rule='All-world kinematic stable fraction at age>=7s below the saved train-world fixed-actor reference minus0.25 (floor0.50) for25consecutive controls; diagnostic not acceptance',
        forced_video_boundaries=[0,800],video_seconds=8.)
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');training={}
    for role,weight in [('control',0.),('candidate',1.)]:
        p=deepcopy(plan);p.update(arm=role,auxiliary_weight=weight,
            run_id=f'duck_getup_late-feasibility_{role}_s42_{stamp}',matching_control=f'duck_getup_late-feasibility_control_s42_{stamp}')
        training[role]=p
    assert not differences(asdict(training_prepare(training['control'])[0]),asdict(training_prepare(training['candidate'])[0]))
    template=deepcopy(read(ROOT/'configs/experiments/getup_neck_home_order.json')['arms']['control'])
    codes=np.random.default_rng(1016).permutation(np.repeat(np.arange(4),[32,32,128,128]))
    names=('prone','supine','left-side','right-side');groups={name:np.flatnonzero(codes==i).tolist() for i,name in enumerate(names)}
    template.update(seed=1016,split=dict(pose_codes=codes.tolist(),groups=groups),
        video_worlds={name:groups[name][0] for name in names},video_title='LATE FEASIBILITY',progress_label='LATE_FEASIBILITY_EVAL',
        handoff_enabled=False,neck_home_enabled=False,scope='Fresh first-episode full-contact recovery evaluation; no fullS5 or hardware acceptance')
    arms={}
    for role in ('original','control','candidate'):
        p=deepcopy(template);p.update(arm=role,run_id=f'duck_getup_late-feasibility-eval_{role}_s1016_{stamp}')
        arms[role]=p
    eval_cfg,_,_=evaluation_prepare(arms['original'])
    compare_arrays(fields,model_arrays(Scene(eval_cfg.env.scene,device='cpu').compile()))
    processor=processor_check(arms['original'],eval_cfg)
    numeric=cpu_check(dict(roles={'getup':template,'standing':dict(template,model=template['standing_model'],model_sha256=template['standing_model_sha256'])}))
    selection=cpu_selection_checks()
    sources=deepcopy(base['source_sha256'])
    for name in ('scripts/training/getup_feasibility_loss.py','scripts/training/getup_late_feasibility.py',
        'scripts/validation/check_getup_feasibility_loss.py','scripts/validation/verify_getup_late_feasibility.py',
        'scripts/validation/verify_late_feasibility_evaluation.py','scripts/validation/getup_late_feasibility_pair.py'):
        sources[name]=sha(ROOT/name)
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name
    for name,digest in base['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    for p in [*training.values(),*arms.values()]:p['source_sha256']=sources
    assert not torch.cuda.is_initialized()
    m=dict(status='frozen-before-learning',stamp=stamp,training_arms=training,arms=arms,
        source_sha256=sources,draft_sha256=sha(ROOT/'configs/experiments/getup_late_feasibility_draft.json'),
        value_admission_sha256=sha(ROOT/'records/reports/getup-interface-value/readback-recovery/analysis.json'),
        maximum_learning_transitions=2457600,maximum_evaluation_transitions=528000,
        training_seed=42,evaluation_seed=1016,heldout_used=False,hardware_io=False,automatic_retry=False,automatic_extension=False,
        scope='Matched common-interface PPO with one late auxiliary objective; fixed first-episode development evaluation only',
        selection_rule=draft['selection'],stop_rule=draft['stops'],full_S2_S6_scope_preserved=True)
    OUT.mkdir();(OUT/'control').mkdir();(OUT/'candidate').mkdir();np.savez_compressed(OUT/'cpu-scene-fields.npz',**fields)
    write(PLAN,m);write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),source_count=len(sources),
        identical_training_configs=True,full_physical_fields_exact=True,processor=processor,model_import=numeric,
        selection=selection,watch_reference=reference,watch_min=plan['watch_stable_min'],cuda_initialized=False,
        training_config=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader)))
    print('LATE_FEASIBILITY_PAIR_FROZEN',sha(PLAN),len(sources),'sources;watch minimum',plan['watch_stable_min'],flush=True)


def checked(resolve_models=True):
    m=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    for name,digest in m['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for p in m['training_arms'].values():
        assert sha(ROOT/p['model'])==p['model_sha256']
        assert sha(ROOT/p['critic_checkpoint'])==p['critic_sha256'] and sha(ROOT/p['probe_dataset'])==p['probe_sha256']
        for base,key in ((WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
            for name,digest in p[key].items():assert sha(base/name)==digest,name
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
    from collect_fixed_recovery import collect
    m=checked();p=m['arms'][role]
    if role!='original':assert p['model']!=m['arms']['original']['model'],'No official fallback for a missing learned model'
    collect(role,m,plan_path=PLAN,out=OUT,prepare=evaluation_prepare,make_memory=make_memory,startup_check=evaluation_startup)


def project_state(**values):
    project=read(ROOT/'project.json');project['staged_training_request'].update(values);write(ROOT/'project.json',project)


def summarize():
    m=checked();proofs={role:read(OUT/f'{role}-analysis.json') for role in ('original','control','candidate') if (OUT/f'{role}-analysis.json').exists()}
    choice=read(OUT/'selection.json');counts={role:p['overall']['timely_holds'] for role,p in proofs.items()}
    trained={role:read(OUT/role/'training-verification.json') for role in ('control','candidate') if (OUT/role/'training-verification.json').exists()}
    result=dict(status='passed',counts=counts,final_hold_counts={role:p['end_hold']['overall']['timely_holds'] for role,p in proofs.items()},
        selected_development_arm=choice.get('selected'),development_lead=choice.get('selected') is not None,
        selection=choice,roles=proofs,actual_learning_transitions=sum(p['actual_control_transitions'] for p in trained.values()),
        actual_evaluation_transitions=len(proofs)*176000,protocol_sha256=sha(PLAN),heldout_used=False,hardware_io=False,
        automatic_extension=False,scope='Training/data integrity plus frozen development gates; independent confirmation and fullS2-S6 remain pending')
    write(OUT/'analysis.json',result);print('LATE_FEASIBILITY_RESULT',json.dumps({k:v for k,v in result.items() if k!='roles'}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--train',choices=('control','candidate'));parser.add_argument('--verify-train',choices=('control','candidate'))
    parser.add_argument('--evaluate',choices=('original','control','candidate'));parser.add_argument('--verify-eval',choices=('original','control','candidate'))
    args=parser.parse_args()
    if args.freeze:freeze();return
    if args.train:
        from getup_late_feasibility import run
        run(checked(False)['training_arms'][args.train],PLAN,OUT);return
    if args.verify_train:
        from verify_getup_late_feasibility import verify
        verify(checked(False)['training_arms'][args.verify_train],OUT);return
    if args.evaluate:evaluate(args.evaluate);return
    if args.verify_eval:
        from verify_late_feasibility_evaluation import verify
        verify(args.verify_eval);return
    assert os.isatty(1),'Use the original visible VS Code shared terminal'
    checked(False);assert not (OUT/'coordinator.json').exists(),'No retry or extension'
    write(OUT/'coordinator.json',dict(status='running',pid=os.getpid()))
    def step(*arguments):subprocess.run([sys.executable,'-B',str(Path(__file__)),*arguments],cwd=WORK,check=True)
    try:
        project_state(learning_running=True,simulation_running=True,current_evidence='records/reports/getup-late-feasibility/control/progress.json')
        step('--train','control');step('--verify-train','control')
        project_state(learning_running=False)
        for role in ('control','original'):step('--evaluate',role);step('--verify-eval',role)
        control,official=[read(OUT/f'{role}-analysis.json') for role in ('control','original')]
        choice=decision(control,official);write(OUT/'selection.json',choice)
        if choice['action']=='candidate':
            project_state(learning_running=True)
            step('--train','candidate');step('--verify-train','candidate');project_state(learning_running=False)
            step('--evaluate','candidate');step('--verify-eval','candidate')
            candidate=read(OUT/'candidate-analysis.json');admitted=candidate_admitted(control,candidate)
            write(OUT/'selection.json',dict(action='select' if admitted else 'close',selected='candidate' if admitted else None,
                candidate_admitted=admitted,required_successes=choice['required_successes'],reason='Frozen functional and relative gates applied'))
        summarize();write(OUT/'coordinator.json',dict(status='completed-review-required',pid=os.getpid()))
    except BaseException as error:
        write(OUT/'coordinator.json',dict(status='stopped-review-required',pid=os.getpid(),error=repr(error)));raise
    finally:project_state(learning_running=False,simulation_running=False)


if __name__=='__main__':main()
