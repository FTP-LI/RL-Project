"""Freeze and visibly execute independent full-command confirmation stages.

Fix all controller parameters before two development seeds. Consume the one
heldout seed only if both development runs pass every predeclared condition.
First-episode functional success includes pose and foot-center proxies; old
instantaneous results stay visible. No learning or parameter adaptation.
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
from grouped_motion_collection import collect,prepare

PLAN=ROOT/'configs/experiments/seven_command_confirmation.json'
OUT=ROOT/'records/reports/seven-command'


def freeze():
    import torch
    assert not PLAN.exists()
    decision=read(ROOT/'records/reports/turn-feedback/analysis.json')
    assert decision['turn_feedback_decision']['candidate_continue'] and decision['strafe_screen_passed']
    assert read(ROOT/'records/reports/straight-parallel/analysis.json')['functional_development_passed']
    base=read(ROOT/'configs/experiments/straight_parallel_confirmation.json')
    original=read(ROOT/'configs/experiments/walking_reference_suite.json')['criteria']
    seeds=[1007,1008,2001];plans={};preflights={}
    sources=['seven_command_confirmation.py','verify_seven_command_confirmation.py','walking_command_controller.py',
        'gyro_turn_controller.py','grouped_motion_collection.py','batched_velocity_confirmation.py',
        'command_input_mapping.py','command_response_probe.py','evaluation_metrics.py','planar_block_metrics.py','s1_reference_inputs.py']
    hashes={'scripts/validation/'+f:sha(ROOT/'scripts/validation'/f) for f in sources}
    hashes.update({'scripts/training/'+f:sha(ROOT/'scripts/training'/f) for f in ('record_io.py','training_watch.py','live_observer.py')})
    for seed in seeds:
        p=deepcopy(base);p.update(stage='S2-seven-command-confirmation',seed=seed,assignment_seed=70000+seed,
            frozen_at=datetime.now(timezone.utc).isoformat(),parallel_worlds=224,episodes_per_command=32,
            commands=original['commands'],maximum_control_steps=1000,maximum_transitions=224000,
            controller_module='walking_command_controller',heldout_used=seed==2001,
            controllers={k:'gyro-feedback' for k in original['commands']},
            controller_descriptions={k:'gyro feedback' if k.startswith('turn-') else 'fixed mapping' for k in original['commands']},
            source_sha256=hashes,
            scope='Full seven-command fixed-controller functional simulation confirmation; no weight learning or hardware claim')
        p['criteria']['linear_error']=original['linear_xy_l2_error_max']
        p['criteria']['signed_speed']={k:original['minimum_signed_command_axis_speed'].get(k,0.) for k in original['commands']}
        p['criteria']['episode_success_fraction_min']=.95
        p['criteria']['posture']=dict(nominal_height_m=.139439,height_tolerance_m=.04,tilt_max_deg=25.,
            minimum_signed_site_separation_m=.015,maximum_bad_sample_fraction=.01,
            scope='After first second, trunk upright and signed foot-site-center proxy; not mesh clearance')
        perm=np.random.default_rng(p['assignment_seed']).permutation(224)
        p['groups']={k:perm[i*32:(i+1)*32].tolist() for i,k in enumerate(p['commands'])}
        actual=yaml.load(yaml.dump(asdict(prepare(p))),Loader=yaml.BaseLoader)
        expected=deepcopy(read(ROOT/'records/reports/straight-parallel/preflight.json')['config'])
        expected['seed']=str(seed);expected['scene']['num_envs']='224';assert actual==expected
        assert sorted(sum(p['groups'].values(),[]))==list(range(224))
        plans[seed]=p;preflights[seed]=actual
    from walking_command_controller import create_controller
    c=create_controller(224);obs=np.zeros((224,61),np.float32);zero=np.zeros((224,3),np.float32)
    assert np.array_equal(c.map(zero,obs),zero)
    for name,digest in base['training_source_sha256'].items():assert sha(WORK/name)==digest
    assert sha(ROOT/base['model'])==base['model_sha256'] and not torch.cuda.is_initialized()
    protocol_dir=ROOT/'configs/experiments/seven_command';protocol_dir.mkdir(exist_ok=True)
    paths={}
    for seed,p in plans.items():
        file=protocol_dir/f's{seed}.json';assert not file.exists();write(file,p);paths[str(seed)]=str(file.relative_to(ROOT))
        out=OUT/f's{seed}';out.mkdir(exist_ok=True)
        write(out/'preflight.json',dict(status='passed',protocol_sha256=sha(file),config=preflights[seed],
            cuda_initialized=False,only_declared_config_changes=True,group_coverage_checked=True))
    master=dict(status='frozen-before-simulation',frozen_at=datetime.now(timezone.utc).isoformat(),
        stage='S2-seven-command-confirmation',ordered_seeds=seeds,development_seeds=seeds[:2],heldout_seed=2001,
        protocols=paths,protocol_sha256={k:sha(ROOT/v) for k,v in paths.items()},
        maximum_transitions=672000,maximum_shared_control_steps=3000,learning_updates=0,
        source_sha256=hashes,automatic_retry=False,automatic_extension=False,
        stop_rule='Stop after any integrity or functional failure; heldout requires both development seeds passing all seven conditions',
        success_rule='Every command >=95% first-episode individual functional successes for each independent seed; report Wilson intervals',
        source_decision_sha256=sha(ROOT/'records/reports/turn-feedback/analysis.json'),
        prior_posture_review_sha256=sha(OUT/'prior-posture-review.json'),
        scope='Fixed official weights plus frozen controller; independent simulation seeds, not three newly trained models')
    write(PLAN,master);print(json.dumps(dict(status='passed',protocol_sha256=sha(PLAN),seeds=seeds,worlds=224),indent=2))


def run_seed(master,seed):
    key=str(seed);file=ROOT/master['protocols'][key];assert sha(file)==master['protocol_sha256'][key]
    plan=read(file);pre=read(OUT/f's{seed}/preflight.json');assert pre['protocol_sha256']==sha(file)
    out=OUT/f's{seed}';assert not (out/'progress.json').exists(),'Existing attempt retained; no automatic retry'
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    assert sha(ROOT/plan['model'])==plan['model_sha256']
    if plan['heldout_used']:
        for previous in master['development_seeds']:assert read(OUT/f's{previous}/analysis.json')['functional_development_passed']
    run_id=f'duck_walk_seven-command_confirmation_s{seed}_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=ROOT/'records/evaluations'/run_id;folder.mkdir();(folder/'videos').mkdir()
    state=dict(status='running',run_id=run_id,path=str(folder),pid=os.getpid(),seed=seed,num_envs=32,parallel_worlds=224,
        episode_seconds=20,conditions=[],model=str(ROOT/plan['model']),model_sha256=plan['model_sha256'],
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(file),learning_updates=0,
        heldout_used=plan['heldout_used'],hardware_io=False,formal_acceptance_eligible=True,scope=plan['scope'])
    def save():write(folder/'evaluation.json',state);write(out/'progress.json',state)
    save();shutil.copy2(file,folder/'protocol.json')
    for name in plan['source_sha256']:shutil.copy2(ROOT/name,folder/Path(name).name)
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=run_id,pid=os.getpid()))
    if plan['heldout_used']:write(OUT/'heldout-consumption.json',dict(started_at=state['started_at'],seed=seed,run_id=run_id,protocol_sha256=sha(file),retry_allowed=False))
    groups={k:np.asarray(v) for k,v in plan['groups'].items()};desired=np.zeros((224,3),np.float32)
    for name,ids in groups.items():desired[ids]=plan['commands'][name]
    from evaluate_standup_reference import Tee
    oldout,olderr=sys.stdout,sys.stderr
    with (folder/'execution.log').open('a',buffering=1) as log:
        sys.stdout,sys.stderr=Tee(oldout,log),Tee(olderr,log)
        try:
            print('SEVEN_COMMAND_STAGE',seed,'heldout',plan['heldout_used'],flush=True)
            collect(plan,pre,folder,state,save,desired,groups,224,1000)
            from verify_seven_command_confirmation import verify
            analysis=verify(folder)
            state['status']='collected-and-verified';write(out/'progress.json',state)
            assert analysis['functional_development_passed'],'Functional gate failed: stop, retain data, do not consume next seed'
            return analysis
        except BaseException as error:
            if state['status']=='running':state.update(status='failed',error=repr(error));save()
            raise
        finally:sys.stdout,sys.stderr=oldout,olderr


def run():
    assert os.isatty(1),'Use the single visible VS Code task terminal'
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']=='S2-seven-command-confirmation'
    master=read(PLAN);assert not (OUT/'suite-progress.json').exists()
    progress=dict(status='running',protocol_sha256=sha(PLAN),completed_seeds=[],pid=os.getpid(),full_goal_complete=False)
    write(OUT/'suite-progress.json',progress)
    try:
        for seed in master['ordered_seeds']:
            run_seed(master,seed);progress['completed_seeds'].append(seed);write(OUT/'suite-progress.json',progress)
        progress['status']='all-seven-functional-confirmed';write(OUT/'suite-progress.json',progress)
        print('SEVEN_COMMAND_CONFIRMATION_COMPLETE: full goal still requires recovery, roll and common-plant transitions.',flush=True)
    except BaseException as error:
        progress.update(status='stopped-review-required',error=repr(error));write(OUT/'suite-progress.json',progress);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    freeze() if args.freeze else run()
