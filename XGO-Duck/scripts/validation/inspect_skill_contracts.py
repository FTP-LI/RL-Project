"""Offline source/model audit before dynamic skill-switch evaluation.

Read ONNX metadata and execute only the archived pure recovery state machine.
Never import Controller, bridge, wire, board APIs, or infer a robot command.
An audit pass means the differences are reproducibly measured; it does not
establish dynamic transition safety or deployment readiness.
"""
from datetime import datetime,timezone
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import numpy as np
import onnx

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'official_resources/xgoduck_runtime_arduino/python'
OUT=ROOT/'records/reports/skill-interfaces'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def pure_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def constants(path,names):
    values={}
    for node in ast.parse(path.read_text(encoding='utf-8-sig')).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in names:values[target.id]=ast.literal_eval(node.value)
    assert set(values)==set(names),(names,values)
    return values


def load_calls():
    tree=ast.parse((ARCHIVE/'controller.py').read_text(encoding='utf-8-sig'))
    result=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or node.func.attr!='load':continue
        owner=node.func.value
        if isinstance(owner,ast.Attribute) and isinstance(owner.value,ast.Name) and owner.value.id=='self':
            result.append(dict(policy=owner.attr,model=None if not node.args else ast.literal_eval(node.args[0]),
                use_model_home=next((ast.literal_eval(k.value) for k in node.keywords if k.arg=='use_model_home'),False),line=node.lineno))
    return result


def main():
    # These two modules are mathematical helpers with no device access.
    core=pure_module(ARCHIVE/'rl_core.py','_xgo_offline_core')
    recovery=pure_module(ARCHIVE/'recovery.py','_xgo_offline_recovery')
    controller=constants(ARCHIVE/'controller.py',['CONTROL_PERIOD','ROULADE_S','ROULADE_ALPHA'])
    imports=read(ROOT/'configs/experiments/official_actor_bootstrap.json')
    paths=[('walk-archived-runtime',ARCHIVE/'xgoduck_walk.onnx'),
           ('walk-fixed-reference',ROOT/imports['official_model']),
           ('getup',ARCHIVE/'xgoduck_getup.onnx'),('roll',ARCHIVE/'xgoduck_roulade.onnx')]
    models=[]
    for role,path in paths:
        model=onnx.load(str(path),load_external_data=False);meta={m.key:m.value for m in model.metadata_props}
        names=meta['joint_names'].split(',');home=np.asarray(meta['default_joint_pos'].split(','),float)
        assert len(names)==14 and home.shape==(14,) and np.isfinite(home).all()
        np.testing.assert_allclose(float(meta['action_scale']),1.,atol=0,rtol=0)
        deg=np.rad2deg(home);offset=deg-core.HOME_DEG
        models.append(dict(role=role,path=str(path.relative_to(ROOT)),sha256=sha(path),joint_names=names,
            model_home_deg=deg.tolist(),runtime_home_deg=core.HOME_DEG.tolist(),
            model_minus_runtime_home_deg=offset.tolist(),max_home_offset_deg=float(abs(offset).max()),
            offset_joints=[dict(joint=names[i],degrees=float(offset[i])) for i in range(14) if abs(offset[i])>1e-4],
            action_scale=float(meta['action_scale']),command_metadata=meta.get('command_names')))
    assert all(m['joint_names']==models[0]['joint_names'] for m in models)
    calls=load_calls()
    assert {c['policy'] for c in calls if not c['use_model_home']}>= {'policy','getup_policy','roulade_policy'}
    machine=recovery.Recovery();events=[]
    gravity=[np.sin(np.deg2rad(60)),0.,-np.cos(np.deg2rad(60))]
    for step in range(180):
        now=step*.02;g=gravity if machine.phase!='getup' else [0.,0.,-1.]
        if machine.update(g,now):events.append(dict(time_seconds=now,step=step,**machine.status()))
        if machine.phase=='walk' and events:break
    assert [e['phase'] for e in events]==['default_pose','getup','walk']
    evaluator=read(ROOT/'configs/experiments/standup_reference.json')
    rollplan=read(ROOT/'configs/experiments/roll_reference.json')
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        models=models,policy_load_calls=calls,
        runtime=dict(**controller,default_alpha=core.ACTION_ALPHA,head_alpha=core.ACTION_ALPHA_HEAD,
            getup_default_hold_seconds=machine.HOLD_SECONDS,fall_angle_deg=machine.FALL_DEG,
            fall_sample_duration_seconds=machine.FALL_SECONDS,standing_angle_deg=machine.STAND_DEG,
            standing_sample_duration_seconds=machine.STAND_SECONDS,
            standing_gate_inputs=['gravity tilt','valid feedback','sample count'],
            explicit_height_or_foot_contact_gate=False,reset_action_history='zeros on policy switch',
            action_history='raw output before exponential smoothing',synthetic_recovery_events=events),
        simulation=dict(getup_freeze_seconds=evaluator['criteria']['freeze_seconds'],
            getup_deadline_seconds=evaluator['criteria']['deadline_seconds'],
            getup_hold_seconds=evaluator['criteria']['hold_seconds'],
            roll_deadline_seconds=rollplan['criteria']['deadline_seconds'],
            roll_hold_seconds=rollplan['criteria']['hold_seconds'],
            uses_each_model_metadata_home=True,external_exponential_smoothing=False),
        findings=['Getup and roll use common runtime HOME while model-aligned simulation uses their metadata HOME.',
                  'Runtime getup default hold is 1 s; task-based reference freeze is 2 s.',
                  'Runtime roll returns after 1.9 s; reference allows a maneuver/hold onset by 7 s plus 2 s hold.',
                  'Runtime standing return uses tilt and a duration, without simulated height or foot-contact checks.',
                  'Raw model action history and smoothed physical target are different state variables.'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            (Path(__file__),ARCHIVE/'controller.py',ARCHIVE/'policy.py',ARCHIVE/'recovery.py',ARCHIVE/'rl_core.py')},
        simulation_transitions_verified=False,deployment_ready=False,hardware_io=False,
        scope='Source/metadata and pure state-machine audit; no dynamics, ONNX inference, training or hardware')
    OUT.mkdir(exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='passed',models=[dict(role=m['role'],max_home_offset_deg=m['max_home_offset_deg'],offset_joints=m['offset_joints']) for m in models],runtime=result['runtime'],simulation=result['simulation'],hardware_io=False),indent=2))


if __name__=='__main__':main()
