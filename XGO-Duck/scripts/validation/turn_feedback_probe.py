"""Prospective randomized-world turn comparison plus missing strafe screen.

Why: static mapping's left yaw MAE exceeded .2 rad/s, with positive bias and
oscillation. Compare one fixed feedback law using the policy's delayed/noisy
gyro, retaining the raw yaw gate. Static strafe mapping is a separate screen.
No learning, solver parity claim, parameter search or automatic extension.
"""
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

PLAN=ROOT/'configs/experiments/turn_feedback_probe.json'
OUT=ROOT/'records/reports/turn-feedback'


def check_controller():
    from gyro_turn_controller import Controller
    from command_input_mapping import transform
    desired=np.array([[0,0,.4],[0,0,.4],[.15,0,0],[0,0,0]],np.float32)
    obs=np.zeros((4,61),np.float32);c=Controller(['static','gyro-feedback','gyro-feedback','gyro-feedback'])
    first=c.map(desired,obs);np.testing.assert_array_equal(first[[0,2,3]],transform(desired)[[0,2,3]])
    np.testing.assert_allclose(first[1,2],.9,atol=1e-6)
    obs[:,2]=.8;second=c.map(desired,obs)
    np.testing.assert_allclose(second[1,2],.6+.75*(.4-.8*c.alpha),atol=1e-6)
    before=c.ema.copy();c.reset(np.array([1]));assert c.ema[0]==before[0] and not c.ready[1]
    third=c.map(desired,obs);np.testing.assert_allclose(third[1,2],.3,atol=1e-6)
    obs[:,2]=-100.;c.reset(np.arange(4));assert c.map(desired,obs)[1,2]==1.
    obs[0,2]=np.nan
    try:c.map(desired,obs)
    except AssertionError:pass
    else:raise AssertionError('Nonfinite feedback input accepted')
    return dict(ema_sequence=True,selective_reset=True,saturation=True,static_zero_linear_identity=True,nonfinite_rejected=True)


def freeze():
    import torch
    assert not PLAN.exists()
    base=read(ROOT/'configs/experiments/straight_parallel_confirmation.json')
    evidence=read(ROOT/'records/reports/turn-feedback/prior-yaw-analysis.json')
    assert evidence['source_status']=='failed'
    original=read(ROOT/'configs/experiments/walking_reference_suite.json')['criteria']
    names=['turn-left-control','turn-right-control','turn-left-feedback','turn-right-feedback','strafe-left','strafe-right']
    commands={k:original['commands'][k.removesuffix('-control').removesuffix('-feedback')] for k in names}
    perm=np.random.default_rng(71006).permutation(96)
    plan=dict(status='frozen-before-simulation',frozen_at=datetime.now(timezone.utc).isoformat(),stage='S2b-turn-strafe-screen',
        task=base['task'],model=base['model'],model_sha256=base['model_sha256'],seed=1006,assignment_seed=71006,
        parallel_worlds=96,episodes_per_command=16,seconds=12,maximum_control_steps=600,maximum_transitions=57600,
        commands=commands,groups={k:perm[i*16:(i+1)*16].tolist() for i,k in enumerate(names)},
        controllers={k:'gyro-feedback' if k.endswith('-feedback') else 'static' for k in names},
        controller_descriptions={k:'gyro feedback' if k.endswith('-feedback') else 'fixed mapping' for k in names},
        controller_module='gyro_turn_controller',feedback=dict(gain=.75,time_constant_seconds=.2,maximum_abs_yaw_input=1.),
        criteria=dict(base['criteria']),joint_names=base['joint_names'],joint_limits=base['joint_limits'],
        video_seconds=3.,video_fps=25,learning_updates=0,heldout_used=False,automatic_retry=False,automatic_extension=False,
        comparison='Independent randomly assigned worlds, not paired physical samples or pointwise trajectory identity',
        continue_gate=dict(left_yaw_mae_relative_improvement_min=.10,right_yaw_mae_ratio_max=1.05,both_turn_functional_pass=True),
        scope='Bounded S2b gyro comparison and S2c static strafe screen only; no complete gait, heldout or hardware claim')
    for key,original_key in (('linear_error','linear_xy_l2_error_max'),('signed_speed','minimum_signed_command_axis_speed')):
        plan['criteria'][key]={k:original[original_key][k.removesuffix('-control').removesuffix('-feedback')] for k in names}
    cfg=prepare(plan);actual=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader)
    expected=read(ROOT/'records/reports/straight-parallel/preflight.json')['config']
    expected['seed']='1006';expected['episode_length_s']='12.0';assert actual==expected
    tests=check_controller();assert sorted(sum(plan['groups'].values(),[]))==list(range(96))
    sources=['turn_feedback_probe.py','verify_turn_feedback_probe.py','gyro_turn_controller.py','grouped_motion_collection.py',
        'batched_velocity_confirmation.py','command_input_mapping.py','command_response_probe.py','evaluation_metrics.py',
        'planar_block_metrics.py','s1_reference_inputs.py']
    plan['source_sha256']={'scripts/validation/'+f:sha(ROOT/'scripts/validation'/f) for f in sources}
    plan['source_sha256'].update({'scripts/training/'+f:sha(ROOT/'scripts/training'/f) for f in ('record_io.py','training_watch.py','live_observer.py')})
    plan['training_source_sha256']=base['training_source_sha256']
    for name,digest in plan['training_source_sha256'].items():assert sha(WORK/name)==digest
    assert sha(ROOT/plan['model'])==plan['model_sha256'] and not torch.cuda.is_initialized()
    write(PLAN,plan);write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),config=actual,
        controller_checks=tests,cuda_initialized=False,only_declared_config_changes=True))
    print(json.dumps(dict(status='passed',protocol_sha256=sha(PLAN),controller_checks=tests),indent=2))


def run():
    assert os.isatty(1),'Use the single visible VS Code task terminal'
    plan=read(PLAN);pre=read(OUT/'preflight.json');assert pre['protocol_sha256']==sha(PLAN)
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']==plan['stage']
    assert not (OUT/'progress.json').exists()
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    assert sha(ROOT/plan['model'])==plan['model_sha256']
    run_id='duck_walk_gyro-turn-screen_comparison_s1006_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=ROOT/'records/evaluations'/run_id;folder.mkdir();(folder/'videos').mkdir()
    from evaluate_standup_reference import Tee
    log=(folder/'execution.log').open('a',buffering=1);sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    state=dict(status='running',run_id=run_id,path=str(folder),pid=os.getpid(),seed=1006,num_envs=16,parallel_worlds=96,
        episode_seconds=12,conditions=[],model=str(ROOT/plan['model']),model_sha256=plan['model_sha256'],
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PLAN),learning_updates=0,
        hardware_io=False,formal_acceptance_eligible=False,scope=plan['scope'])
    def save():write(folder/'evaluation.json',state);write(OUT/'progress.json',state)
    save();shutil.copy2(PLAN,folder/'protocol.json')
    for name in plan['source_sha256']:shutil.copy2(ROOT/name,folder/Path(name).name)
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=run_id,pid=os.getpid()))
    groups={k:np.asarray(v) for k,v in plan['groups'].items()};desired=np.zeros((96,3),np.float32)
    for name,ids in groups.items():desired[ids]=plan['commands'][name]
    try:collect(plan,pre,folder,state,save,desired,groups,96,600)
    except BaseException as error:
        if state['status']=='running':state.update(status='failed',error=repr(error));save()
        raise
    from verify_turn_feedback_probe import verify
    verify(folder);state['status']='collected-and-verified';write(OUT/'progress.json',state)
    print('TURN_STRAFE_SCREEN_COMPLETE: analyze before any further execution.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    freeze() if args.freeze else run()
