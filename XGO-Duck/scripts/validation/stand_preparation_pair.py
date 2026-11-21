"""Prospective zero-learning standing-to-roll preparation diagnostic.

Keep source initial noise and roll criteria. Prepared arm spends one real
second using the official zero-command walking policy, retains its physical
state, then selects the unchanged roll model with fresh raw-output history.
Every proposal and preparation failure remains in the denominator.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import numpy as np
import yaml
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
from prepared_roll_collection import config,evaluate
from evaluate_roll_reset_pair import differences
from verify_roll_reference import independent

PLAN=ROOT/'configs/experiments/roll_stand_preparation_pair.json'
OUT=ROOT/'records/reports/roll-stand-preparation'


def freeze():
    import torch
    import onnx
    import onnxruntime as ort
    import mujoco
    assert not PLAN.exists()
    closed=ROOT/'records/reports/roll-completion-frontier/startup-recovery/analysis.json'
    assert not read(closed)['candidate_continue']
    base=read(ROOT/'configs/experiments/roll_reference.json')
    walk=read(ROOT/'configs/experiments/seven_command/s1007.json')
    sources=dict(base['source_sha256'])
    for name in ('stand_preparation_pair.py','prepared_roll_collection.py','roll_standing_preparation.py','s1_reference_inputs.py'):
        path=ROOT/'scripts/validation'/name;sources[str(path.relative_to(ROOT))]=sha(path)
    for name,digest in base['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in base['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    for path,digest in ((base['model'],base['model_sha256']),(walk['model'],walk['model_sha256'])):assert sha(ROOT/path)==digest
    meta={role:{x.key:x.value for x in onnx.load(ROOT/model).metadata_props} for role,model in (('roll',base['model']),('walk',walk['model']))}
    assert meta['roll']['joint_names']==meta['walk']['joint_names']
    assert all(float(m['action_scale'])==1. for m in meta.values())
    names=meta['roll']['joint_names'].split(',');stamp=datetime.now().strftime('%Y%m%dT%H%M%S')
    items=[];cfgs={}
    for role,seconds in (('control',0.),('prepared',1.)):
        p=deepcopy(base);p.update(role=role,seed=1007,prepare_seconds=seconds,
            preparation_model=walk['model'],preparation_model_sha256=walk['model_sha256'],
            joint_names=names,source_sha256=sources,run_id=f'duck_roll_stand-preparation_{role}_s1007_{stamp}',
            scope='Development standing-to-roll preparation; no learning, heldout, common-plant locomotion or hardware acceptance')
        cfgs[role]=config(p);items.append(p)
    diff=differences(asdict(cfgs['control']),asdict(cfgs['prepared']))
    assert len(diff)==1 and diff[0]['path']=='.episode_length_s',diff
    assert float(cfgs['control'].episode_length_s)==9. and float(cfgs['prepared'].episode_length_s)==10.
    model=cfgs['control'].scene.entities['robot'].spec_fn().compile()
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names];assert min(ids)>=0
    limits=model.jnt_range[ids].copy()
    for p in items:p['joint_limits']=limits.tolist()
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/walk['model']),sess_options=options,providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
    old=read(ROOT/'records/reports/roll-adaptation-screen/original-analysis.json')
    with np.load(ROOT/old['evaluator']/'forward-roll-trace.npz') as t:obs=t['observations'][0].copy()
    obs[:,6:20]+=np.asarray(meta['roll']['default_joint_pos'].split(','),np.float32)-np.asarray(meta['walk']['default_joint_pos'].split(','),np.float32)
    obs[:,34:48]=0;obs[:,48:]=0
    actions=np.concatenate([session.run(None,{'obs':row[None]})[0] for row in obs]);assert np.isfinite(actions).all()
    tree=ast.parse((ROOT/'scripts/validation/prepared_roll_collection.py').read_text())
    resets=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='env' and n.func.attr=='reset']
    assert len(resets)==1
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=False)
    master=dict(status='frozen-before-simulation',stage='S4-roll-preparation',change='stand-preparation',output=str(OUT.relative_to(ROOT)),
        evaluations=items,source_sha256=sources,training_source_sha256=base['training_source_sha256'],
        maximum_transitions=30400,maximum_control_steps=950,learning_updates=0,heldout_used=False,
        continue_gate='prepared>=29/32 and prepared>=control; compare counts and intervals without claiming a training improvement',
        preparation_gate=dict(max_tilt_deg=45.,min_height_m=.08,actual_joint_tolerance_deg=2.,trigger_hold_seconds=.2),
        physical_reset_at_switch=False,roll_history_reset_at_switch=True,actuator_history_reset_at_switch=False,
        roll_deadline_after_trigger_seconds=7.,roll_hold_seconds=2.,automatic_extension=False,automatic_retry=False,
        preceding_failed_learning_sha256=sha(closed),stamp=stamp)
    write(PLAN,master)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
        evaluation_configs={k:yaml.load(yaml.dump(asdict(v)),Loader=yaml.BaseLoader) for k,v in cfgs.items()},
        configuration_differences=diff,walking_input_samples=len(obs),walking_actions_finite=True,
        joint_order_and_model_homes_checked=True,only_initial_explicit_reset_lines=resets,cuda_initialized=False))
    print(json.dumps(dict(status='passed',protocol_sha256=sha(PLAN),configuration_differences=diff,
                          runs=[p['run_id'] for p in items],walking_input_samples=len(obs)),indent=2))


def verify(folder):
    import imageio_ffmpeg
    import onnxruntime as ort
    folder=Path(folder);p=read(folder/'protocol.json');r=read(folder/'evaluation.json')
    assert r['status']=='completed' and r['protocol_sha256']==sha(folder/'protocol.json')
    assert p['master_sha256']==sha(PLAN) and sha(ROOT/p['model'])==p['model_sha256']==r['model_sha256']
    assert sha(ROOT/p['preparation_model'])==p['preparation_model_sha256']
    assert yaml.load((folder/'env.yaml').read_text(),Loader=yaml.BaseLoader)==read(OUT/'preflight.json')['evaluation_configs'][p['role']]
    for name,digest in p['source_sha256'].items():assert sha(folder/Path(name).name)==digest,name
    assert sha(folder/'expanded-model-initial.npz')==r['expanded_model_sha256']
    assert sha(folder/'reset-physical-initial.npz')==r['reset_initial_sha256']
    prep=read(folder/'standing-preparation.json');steps=round(p['prepare_seconds']/.02)
    assert prep['control_steps']==steps and prep['end_counter']-prep['start_counter']==steps
    assert r['physical_counter_at_roll_trigger']==prep['end_counter']
    assert r['physical_counter_at_end']-r['physical_counter_at_roll_trigger']==450
    assert not prep['physical_reset_at_handoff']
    assert sha(folder/'standing-preparation.npz')==prep['preparation_trace_sha256']==r['preparation_trace_sha256']
    with np.load(folder/'standing-preparation.npz') as f:s={k:f[k].copy() for k in f.files}
    with np.load(folder/'reset-physical-initial.npz') as f:
        np.testing.assert_array_equal(f['qpos'],s['qpos'][0]);np.testing.assert_array_equal(f['qvel'],s['qvel'][0])
    np.testing.assert_allclose(s['time'],np.arange(steps+1)*.02-p['prepare_seconds'],atol=1e-7,rtol=0)
    limits=np.asarray(r['joint_limits']);np.testing.assert_allclose(limits,p['joint_limits'],atol=1e-12,rtol=0)
    positions=s['qpos'][:,:,r['qpos_joint_addresses']];tol=np.deg2rad(2.)
    bounded=((positions>=limits[:,0]-tol)&(positions<=limits[:,1]+tol)).all(axis=2)
    np.testing.assert_array_equal(s['joint_envelope_ok'],bounded)
    q=s['quaternion'];np.testing.assert_allclose(np.linalg.norm(q,axis=2),1,atol=2e-4,rtol=0)
    tilt=np.arccos(np.clip(1-2*(q[:,:,1]**2+q[:,:,2]**2),-1,1));c=p['criteria']
    standing=(abs(s['height']-c['nominal_height_m'])<=c['height_tolerance_m'])&(tilt<=np.deg2rad(c['tilt_max_deg']))
    standing&=(np.linalg.norm(s['angular_velocity'],axis=2)<=c['angular_speed_max_rad_s'])&s['foot_contact'].all(axis=2)&bounded
    np.testing.assert_array_equal(standing,s['standing_ok'])
    valid=bounded&~s['terminated']&~s['truncated']
    if steps:valid&=(s['height']>=.08)&(tilt<=np.deg2rad(45.))
    np.testing.assert_array_equal(s['active'],np.logical_and.accumulate(valid,axis=0))
    alive=s['active'][-1].copy()
    if steps:alive&=standing[-10:].all(axis=0)
    np.testing.assert_array_equal(alive,prep['alive'])
    with np.load(folder/'forward-roll-trace.npz') as f:t={k:f[k].copy() for k in f.files}
    item=r['conditions'][0];assert sha(folder/'forward-roll-trace.npz')==item['trace_sha256']
    np.testing.assert_array_equal(t['initial_qpos'],s['qpos'][-1]);np.testing.assert_array_equal(t['initial_qvel'],s['qvel'][-1])
    np.testing.assert_allclose(t['joint_positions'][0],positions[-1],atol=1e-7,rtol=0)
    np.testing.assert_array_equal(t['active'][0],alive)
    np.testing.assert_allclose(t['observations'][0,:,34:48],0.,atol=1e-7,rtol=0)
    np.testing.assert_allclose(t['observations'][1:,:,34:48][t['active'][1:]],t['raw_policy_actions'][:-1][t['active'][1:]],atol=1e-7,rtol=0)
    np.testing.assert_allclose(t['observations'][:,:,48:][t['active']],0.,atol=1e-6,rtol=0)
    np.testing.assert_allclose(t['applied_targets'][1:],np.asarray(r['policy_home'])+t['raw_policy_actions'][1:],atol=1e-6,rtol=0)
    rows=independent(t,c,limits,tol)
    for observed,saved in zip(rows,item['episodes']):
        for key,value in observed.items():
            wanted=saved['roll_success'] if key=='success' else saved[key]
            if value is None or isinstance(value,(bool,int)):assert value==wanted,(key,value,wanted)
            else:np.testing.assert_allclose(value,wanted,atol=2e-4,rtol=0)
        observed['roll_success']=observed['success'];observed['preparation_passed']=bool(alive[observed['env_index']])
        observed['success']=bool(observed['success'] and observed['preparation_passed'])
        assert observed['success']==saved['success']
    options=ort.SessionOptions();options.intra_op_num_threads=1;replays={};maximum=0.
    for skill,model,obs,actions in (('roll',p['model'],t['observations'],t['raw_policy_actions']),
                                  ('stand',p['preparation_model'],s['observations'],s['raw_actions'])):
        session=ort.InferenceSession(str(ROOT/model),sess_options=options,providers=['CPUExecutionProvider'])
        indices=range(1,len(obs),30 if skill=='roll' else 5);count=0
        for step in indices:
            active=t['active'][step] if skill=='roll' else s['active'][step-1]
            for i in np.flatnonzero(active):
                action=session.run(None,{'obs':obs[step,i:i+1]})[0][0]
                np.testing.assert_allclose(action,actions[step,i],atol=2e-5,rtol=2e-5)
                maximum=max(maximum,float(abs(action-actions[step,i]).max()));count+=1
        replays[skill]=count
        if skill=='stand' and steps:
            np.testing.assert_allclose(s['observations'][1:,:,34:48],s['raw_actions'][:-1],atol=1e-7,rtol=0)
            home=np.asarray(session.get_modelmeta().custom_metadata_map['default_joint_pos'].split(','),np.float32)
            np.testing.assert_allclose(s['targets'][1:],home+s['raw_actions'][1:],atol=1e-6,rtol=0)
    video=folder/item['video']['path'];assert sha(video)==item['video']['sha256']
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);frames=sum(1 for _ in reader)
    assert meta['fps']==25 and meta['size']==(960,720) and frames==len(item['video']['frame_times_seconds'])
    assert item['video']['frame_times_seconds'][:prep['prefix_frames']]==prep['prefix_times_seconds']
    k=sum(x['success'] for x in rows);n=len(rows);assert n==32 and k==item['successes']
    z=1.959963984540054;f=k/n;den=1+z*z/n;mid=(f+z*z/(2*n))/den;half=z*np.sqrt(f*(1-f)/n+z*z/(4*n*n))/den
    result=dict(status='passed',role=p['role'],episodes=n,successes=k,roll_successes=sum(x['roll_success'] for x in rows),
        preparation_passed=int(alive.sum()),numeric_gate_passed=f>=.9,wilson95=[mid-half,mid+half],rows=rows,
        evaluator=str(folder.relative_to(ROOT)),protocol_sha256=sha(folder/'protocol.json'),master_sha256=sha(PLAN),
        trace_sha256=sha(folder/'forward-roll-trace.npz'),preparation_trace_sha256=sha(folder/'standing-preparation.npz'),
        video_sha256=sha(video),decoded_frames=frames,replayed_inputs=replays,maximum_inference_error=maximum,
        physical_reset_at_handoff=False,physical_boundary_exact=True,learning_updates=0,heldout_used=False)
    write(folder/'verification.json',result);write(OUT/f'{p["role"]}-analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));return result


def run_one(role):
    from evaluate_standup_reference import Tee
    assert os.isatty(1),'Use the single visible task terminal'
    master=read(PLAN);p=deepcopy(next(x for x in master['evaluations'] if x['role']==role));p['master_sha256']=sha(PLAN)
    for name,digest in p['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in p['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    for model,digest in ((p['model'],p['model_sha256']),(p['preparation_model'],p['preparation_model_sha256'])):assert sha(ROOT/model)==digest
    folder=ROOT/'records/evaluations'/p['run_id'];folder.mkdir();(folder/'videos').mkdir()
    assert not (OUT/f'{role}-progress.json').exists()
    write(folder/'protocol.json',p)
    for name in p['source_sha256']:shutil.copy2(ROOT/name,folder/Path(name).name)
    state=dict(status='running',run_id=p['run_id'],path=str(folder),pid=os.getpid(),conditions=[],
        role=role,model=str(ROOT/p['model']),model_sha256=p['model_sha256'],protocol_sha256=sha(folder/'protocol.json'),
        learning_updates=0,started_at=datetime.now(timezone.utc).isoformat())
    def save():write(folder/'evaluation.json',state);write(OUT/f'{role}-progress.json',state)
    save();write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=p['run_id'],pid=os.getpid()))
    with (folder/'execution.log').open('a',buffering=1) as log:
        oldout,olderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(oldout,log),Tee(olderr,log)
        try:evaluate(p,folder,state);state['status']='completed'
        except BaseException as error:state.update(status='failed',error=repr(error));raise
        finally:state['ended_at']=datetime.now(timezone.utc).isoformat();save();sys.stdout,sys.stderr=oldout,olderr
    verify(folder)


def analyze():
    rows={role:read(OUT/f'{role}-analysis.json') for role in ('control','prepared')}
    assert all(r['status']=='passed' for r in rows.values())
    folders={role:ROOT/r['evaluator'] for role,r in rows.items()};resets={};fields={}
    for role,folder in folders.items():
        with np.load(folder/'reset-physical-initial.npz') as f:resets[role]={k:f[k].copy() for k in f.files}
        with np.load(folder/'expanded-model-initial.npz') as f:fields[role]={k:f[k].copy() for k in f.files}
    for key in resets['control']:np.testing.assert_array_equal(resets['control'][key],resets['prepared'][key])
    assert set(fields['control'])==set(fields['prepared'])
    comparison=[]
    for key,a in fields['control'].items():
        b=fields['prepared'][key]
        if key in ('body_invweight0','dof_invweight0'):
            assert a.dtype==b.dtype==np.float32 and np.all(a>=0) and np.all(b>=0)
            ulp=abs(a.view(np.int32).astype(np.int64)-b.view(np.int32).astype(np.int64))
            rel=abs(a.astype(float)-b.astype(float))/np.maximum(abs(a.astype(float)),1e-30)
            assert ulp.max(initial=0)<=8 and rel.max(initial=0)<=1e-6
            comparison.append(dict(field=key,max_ulp=int(ulp.max(initial=0)),max_relative=float(rel.max(initial=0))))
        else:np.testing.assert_array_equal(a,b);comparison.append(dict(field=key,exact=True))
    k={role:r['successes'] for role,r in rows.items()}
    advance=bool(k['prepared']>=29 and k['prepared']>=k['control'])
    result=dict(status='passed',successes=k,episodes_each=32,preparation_continue=advance,
        decision='independent-interface-confirmation-warranted' if advance else 'close-standing-preparation-screen',
        captured_reset_qpos_qvel_exact=True,parameter_comparison=comparison,
        preparation_failures_retained=True,physical_reset_at_handoff=False,physical_boundaries_exact=True,
        post_trigger_roll_criteria_unchanged=True,total_observation_seconds={'control':9.,'prepared':10.},
        learning_updates=0,heldout_used=False,protocol_sha256=sha(PLAN))
    write(OUT/'analysis.json',result);print(json.dumps(result,indent=2));return result


def run():
    assert os.isatty(1)
    master=read(PLAN);assert sha(PLAN)==read(OUT/'preflight.json')['protocol_sha256']
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']==master['stage']
    assert not (OUT/'progress.json').exists()
    state=dict(status='running',pid=os.getpid(),protocol_sha256=sha(PLAN),completed_roles=[]);write(OUT/'progress.json',state)
    try:
        for p in master['evaluations']:
            subprocess.run([sys.executable,'-B',str(Path(__file__)),'--protocol',str(PLAN),'--role',p['role']],cwd=WORK,check=True)
            state['completed_roles'].append(p['role']);write(OUT/'progress.json',state)
        analyze();state['status']='completed-review-required'
    except BaseException as error:state.update(status='stopped-review-required',error=repr(error));raise
    finally:write(OUT/'progress.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--protocol');parser.add_argument('--role');parser.add_argument('--verify-folder');parser.add_argument('--analyze',action='store_true');args=parser.parse_args()
    if args.protocol:PLAN=Path(args.protocol);OUT=ROOT/read(PLAN)['output']
    if args.freeze:freeze()
    elif args.role:run_one(args.role)
    elif args.verify_folder:verify(args.verify_folder)
    elif args.analyze:analyze()
    else:run()
