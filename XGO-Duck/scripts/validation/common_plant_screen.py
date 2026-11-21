"""Bounded fixed-skill regression under one nominal physical recipe.

Freeze every interface and metric before simulation. First construct all
three actual GPU scenes without stepping them; only if their contracts pass,
run seven walking commands, four recovery strata and one forward-roll group.
All original skill criteria remain. A behavioral failure is a diagnostic
result, while an integrity/capacity failure stops the whole screen. No learning,
automatic extension, heldout selection or continuous-switch claim.
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
import grouped_motion_collection as walk_collection
import evaluate_standup_reference as getup_collection
import collect_roll_policy as roll_collection
from common_scene_contract import adopt_physics,physical_description,model_arrays,compare_arrays,install_guard
from inspect_common_skill_plant import digest,stable

PLAN=ROOT/'configs/experiments/common_plant_screen.json'
OUT=ROOT/'records/reports/common-plant-screen'
ORIGINAL_PREPARE={'walk':walk_collection.prepare,'getup':getup_collection.config,'roll':roll_collection.config}


def prepare(skill,plan):return adopt_physics(ORIGINAL_PREPARE[skill](plan),skill)


def freeze():
    from mjlab.scene import Scene
    assert os.isatty(1) and not PLAN.exists() and not OUT.exists()
    audit=ROOT/'records/reports/common-plant/source-contract.json';assert read(audit)['status']=='passed'
    from common_skill_plant import __file__ as helper
    assert sha(Path(helper))==read(audit)['helper_sha256']
    templates={'walk':read(ROOT/'configs/experiments/seven_command/s1007.json'),
        'getup':read(ROOT/'configs/experiments/standup_reference.json'),'roll':read(ROOT/'configs/experiments/roll_reference.json')}
    sources={};training_sources={}
    for p in templates.values():
        for name,digest_ in p['source_sha256'].items():assert sha(ROOT/name)==digest_,name
        sources.update(p['source_sha256']);training_sources.update(p['training_source_sha256'])
    for name in ('common_plant_screen.py','common_scene_contract.py','common_skill_plant.py',
        'inspect_common_skill_plant.py','verify_common_plant_screen.py','verify_standup_reference.py',
        'collect_roll_policy.py','roll_model_screen.py'):
        sources['scripts/validation/'+name]=sha(ROOT/'scripts/validation'/name)
    sources['scripts/training/metadata_actor_import.py']=sha(ROOT/'scripts/training/metadata_actor_import.py')
    for name,digest_ in training_sources.items():assert sha(WORK/name)==digest_,name
    for name,digest_ in read(audit)['source_sha256'].items():
        assert sha(WORK/name)==digest_,name;training_sources[name]=digest_
    plans={};configs={};models={};fingerprints={};stamp=datetime.now().strftime('%Y%m%dT%H%M%S')
    for skill,old in templates.items():
        p=deepcopy(old);p.update(skill=skill,seed=1009,stage='S5-common-plant-nominal-screen',
            source_sha256=sources,training_source_sha256=training_sources,heldout_used=False,learning_updates=0,
            scope='Nominal common-plant fixed-skill development screen; no learning, continuous switch or hardware claim')
        if skill=='walk':
            p['assignment_seed']=71009;perm=np.random.default_rng(p['assignment_seed']).permutation(224)
            p['groups']={k:perm[i*32:(i+1)*32].tolist() for i,k in enumerate(p['commands'])}
        if skill=='roll':p['role']='reference'
        cfg=prepare(skill,p)
        assert cfg.sim.nconmax==200 and cfg.sim.mujoco.timestep==.004 and cfg.decimation==5
        assert not any(k in cfg.events for k in ('encoder_bias','randomize_com','randomize_head_com',
            'randomize_mass_inertia','randomize_joint_friction','randomize_armature','push_robot'))
        fingerprints[skill]=digest(physical_description(cfg))
        configs[skill]=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader);plans[skill]=p
        scene=Scene(cfg.scene,device='cpu');model=scene.compile();models[skill]=model_arrays(model)
    assert len(set(fingerprints.values()))==1
    for skill in ('getup','roll'):compare_arrays(models[skill],models['walk'])
    assert plans['getup']['criteria']['max_seconds']==11 and plans['getup']['criteria']['freeze_seconds']==2
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=False);directory=ROOT/'configs/experiments/common_plant';directory.mkdir(exist_ok=False)
    paths={};config_paths={}
    for skill,p in plans.items():
        path=directory/f'{skill}.json';write(path,p);paths[skill]=str(path.relative_to(ROOT))
        (OUT/skill).mkdir();(OUT/'startup'/skill).mkdir(parents=True)
        np.savez_compressed(OUT/f'{skill}-cpu-scene-fields.npz',**models[skill])
        write(OUT/f'{skill}-preflight.json',dict(status='passed',protocol_sha256=sha(path),config=configs[skill],
            physical_config_sha256=fingerprints[skill],scene_fields_sha256=sha(OUT/f'{skill}-cpu-scene-fields.npz')))
    (OUT/'walk/s1009').mkdir()
    shutil.copy2(OUT/'walk-preflight.json',OUT/'walk/s1009/preflight.json')
    roll_master=dict(stage='S5-common-plant-nominal-screen',change='common-plant',output=str((OUT/'roll').relative_to(ROOT)),
        evaluations=[plans['roll']],heldout_used=False,source_sha256=sources,training_source_sha256=training_sources)
    write(OUT/'roll/evaluation-master.json',roll_master)
    write(OUT/'roll/preflight.json',dict(status='passed',protocol_sha256=sha(OUT/'roll/evaluation-master.json'),
        evaluation_configs={'reference':configs['roll']}))
    master=dict(status='frozen-before-simulation',stage='S5-common-plant-nominal-screen',stamp=stamp,
        protocols=paths,protocol_sha256={k:sha(ROOT/v) for k,v in paths.items()},source_sha256=sources,
        training_source_sha256=training_sources,audit_sha256=sha(audit),output=str(OUT.relative_to(ROOT)),
        physical_config_sha256=fingerprints['walk'],maximum_episodes=384,maximum_control_transitions=308800,
        budgets={'walk':224000,'getup':70400,'roll':14400},learning_updates=0,physical_seed=1009,heldout_used=False,
        startup_checks_before_any_physics=True,automatic_retry=False,automatic_extension=False,hardware_io=False,
        stop_rule='Integrity/nonfinite/contact capacity stops all; retain behavioral failures and complete other fixed diagnostics within original budgets',
        continue_gate='All seven walking conditions and four getup strata pass original gates; roll passes unchanged>=90percent gate before continuous-switch acceptance',
        scope='One nominal common-plant development seed only; per-skill first episodes, not continuous transitions')
    write(PLAN,master);write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
        full_cpu_scene_fields_exact=True,physical_fingerprints=fingerprints,cuda_initialized=False,
        maximum_control_transitions=308800,source_getup_total_seconds=11,source_freeze_included=True))
    print(json.dumps(dict(status='passed',protocol_sha256=sha(PLAN),cpu_complete_scene_fields_equal=True,
        maximum_episodes=384,maximum_control_transitions=308800,cuda_initialized=False),indent=2),flush=True)


def checked_plan(skill):
    m=read(PLAN);path=ROOT/m['protocols'][skill];assert sha(path)==m['protocol_sha256'][skill]
    p=read(path)
    for name,digest_ in p['source_sha256'].items():assert sha(ROOT/name)==digest_,name
    for name,digest_ in p['training_source_sha256'].items():assert sha(WORK/name)==digest_,name
    assert sha(ROOT/p['model'])==p['model_sha256']
    return p,path


def startup(skill):
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends(allow_tf32=False)
    p,_=checked_plan(skill);folder=OUT/'startup'/skill
    assert not (folder/'runtime-contract.json').exists()
    cls=install_guard(p,folder,OUT/'startup','startup')
    env=cls(cfg=prepare(skill,p),device='cuda:0',render_mode=None if skill=='walk' else 'rgb_array')
    try:env.seed(p['seed']);env.reset();assert env.common_step_counter==0
    finally:env.close()
    assert read(folder/'runtime-contract.json')['control_steps']==0
    print('COMMON_STARTUP_PASSED',skill,'physics_steps=0',flush=True)


def setup_record(skill,p,path):
    run_id=f'duck_{skill}_common-plant_reference_s1009_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=ROOT/'records/evaluations'/run_id;folder.mkdir();(folder/'videos').mkdir()
    write(folder/'protocol.json',p)
    for name in p['source_sha256']:shutil.copy2(ROOT/name,folder/Path(name).name)
    state=dict(status='running',run_id=run_id,path=str(folder),pid=os.getpid(),seed=p['seed'],conditions=[],
        model=str(ROOT/p['model']),model_sha256=p['model_sha256'],protocol_sha256=sha(path),
        learning_updates=0,started_at=datetime.now(timezone.utc).isoformat(),heldout_used=False,hardware_io=False,scope=p['scope'])
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=run_id,pid=os.getpid()))
    def save():write(folder/'evaluation.json',state);write(OUT/skill/'progress.json',state)
    save();return folder,state,save


def evaluate(skill):
    from evaluate_standup_reference import Tee
    p,path=checked_plan(skill)
    assert not (OUT/skill/'progress.json').exists()
    if skill=='roll':
        import roll_model_screen as base
        original=roll_collection.evaluate
        roll_collection.config=lambda plan:prepare('roll',plan)
        def guarded(plan,folder,state):
            install_guard(plan,folder,OUT/'startup','rollout');return original(plan,folder,state)
        base.evaluate=guarded
        base.run_one(OUT/'roll/evaluation-master.json','reference')
        return
    folder,state,save=setup_record(skill,p,path)
    install_guard(p,folder,OUT/'startup','rollout')
    with (folder/'execution.log').open('a',buffering=1) as log:
        oldout,olderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(oldout,log),Tee(olderr,log)
        try:
            if skill=='walk':
                state.update(num_envs=32,parallel_worlds=224,episode_seconds=20,formal_acceptance_eligible=True)
                desired=np.zeros((224,3),np.float32);groups={k:np.asarray(v) for k,v in p['groups'].items()}
                for name,ids in groups.items():desired[ids]=p['commands'][name]
                walk_collection.prepare=lambda plan:prepare('walk',plan)
                walk_collection.collect(p,read(OUT/'walk-preflight.json'),folder,state,save,desired,groups,224,1000)
                import verify_seven_command_confirmation as verifier
                verifier.OUT=OUT/'walk';verifier.verify(folder)
            else:
                getup_collection.config=lambda plan:prepare('getup',plan);getup_collection.OUT=OUT/'getup'
                getup_collection.evaluate(p,folder,state);state['status']='completed';save()
                import verify_standup_reference as verifier
                verifier.OUT=OUT/'getup';verifier.main()
            state['status']='completed';state['ended_at']=datetime.now(timezone.utc).isoformat();save()
        except BaseException as error:
            state.update(status='failed',error=repr(error),ended_at=datetime.now(timezone.utc).isoformat());save();raise
        finally:sys.stdout,sys.stderr=oldout,olderr


def child(*args):
    print('VISIBLE_COMMON_STAGE',*map(str,args),flush=True)
    subprocess.run([sys.executable,'-B',str(Path(__file__)),*map(str,args)],cwd=WORK,check=True)


def run():
    assert os.isatty(1)
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']=='S5-common-plant-nominal-screen'
    master=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    assert not (OUT/'progress.json').exists()
    state=dict(status='running',pid=os.getpid(),protocol_sha256=sha(PLAN),completed_stages=[]);write(OUT/'progress.json',state)
    try:
        for skill in ('walk','getup','roll'):
            child('--startup',skill);state['completed_stages'].append('startup-'+skill);write(OUT/'progress.json',state)
        for skill in ('walk','getup','roll'):
            child('--evaluate',skill)
            subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validation/verify_common_plant_screen.py'),skill],cwd=WORK,check=True)
            state['completed_stages'].append(skill);write(OUT/'progress.json',state)
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validation/verify_common_plant_screen.py'),'--analyze'],cwd=WORK,check=True)
        state['status']='completed-review-required'
    except BaseException as error:state.update(status='stopped-review-required',error=repr(error));raise
    finally:write(OUT/'progress.json',state)


if __name__=='__main__':
    from evaluate_standup_reference import Tee
    import traceback
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--startup',choices=['walk','getup','roll']);parser.add_argument('--evaluate',choices=['walk','getup','roll']);a=parser.parse_args()
    operation='freeze' if a.freeze else ('startup-'+a.startup if a.startup else 'evaluate-'+a.evaluate if a.evaluate else 'run')
    logdir=ROOT/'records/reports/common-plant'
    logpath=logdir/f'screen-{operation}-{datetime.now().strftime("%Y%m%dT%H%M%S")}-{os.getpid()}.log'
    event=dict(operation=operation,argv=sys.argv,started_at=datetime.now(timezone.utc).isoformat(),
        foreground_tty=os.isatty(1),script_sha256=sha(Path(__file__)),log=str(logpath.relative_to(ROOT)))
    with logpath.open('x',buffering=1) as log:
        oldout,olderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(oldout,log),Tee(olderr,log)
        try:
            print('COMMON_VISIBLE_COMMAND',json.dumps(event),flush=True)
            if a.freeze:freeze()
            elif a.startup:startup(a.startup)
            elif a.evaluate:evaluate(a.evaluate)
            else:run()
            event['exit_code']=0
        except BaseException as error:
            event.update(exit_code=1,error=repr(error));traceback.print_exc();raise
        finally:
            event['ended_at']=datetime.now(timezone.utc).isoformat()
            with (logdir/'screen-commands.jsonl').open('a') as ledger:ledger.write(json.dumps(event)+'\n')
            sys.stdout,sys.stderr=oldout,olderr
