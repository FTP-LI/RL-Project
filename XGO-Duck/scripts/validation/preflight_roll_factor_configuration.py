"""Inspect real CPU configuration and compiled models for three roll factors.

This uses the existing factor adapter on copies of the official nominal route.
It creates no environment, takes no physical steps, and does not authorize a
dynamic experiment. Voltage also changes compiled actuator force limits; that
coupling is checked and recorded instead of being hidden as a single field.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/validation'))
sys.path.insert(0,str(ROOT/'scripts/training'))
OUT = ROOT/'records/reports/roll-matched-factor-preflight-20261116'
PLAN = ROOT/'configs/experiments/roll_native_nominal_observed_s1241_20261115.json'
DRAFT = ROOT/'configs/experiments/roll_parameter_matched_review_draft.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        result=[]
        for key in sorted(set(a)|set(b)):
            name=f'{path}.{key}' if path else str(key)
            if key not in a or key not in b:
                result.append(name)
            else:
                result.extend(differences(a[key],b[key],name))
        return result
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):return [path+'.length']
        return [item for i,(x,y) in enumerate(zip(a,b)) for item in differences(x,y,f'{path}[{i}]')]
    return [] if a==b else [path]


def run():
    import numpy as np
    import torch
    from mjlab.scene import Scene
    from common_scene_contract import model_arrays, physical_description
    from inspect_common_skill_plant import stable
    import run_continuous_roll_stand_walk as route
    import run_roll_domain_parameter_screen as screen

    assert not torch.cuda.is_initialized()
    p=json.loads(PLAN.read_text(encoding='utf-8')); draft=json.loads(DRAFT.read_text(encoding='utf-8'))
    assert p['seed']==p['roll']['seed']==1241 and p['worlds']==128 and p['maximum_controls']==1550
    assert p['cross_model_bridge_controls']==draft['base']['cross_model_bridge_controls']==0
    assert p['feedback']['maximum_yaw_input_rad_s']==draft['base']['feedback_cap_rad_s']==.15
    for role,key in [('roll','roll_model'),('walking','walk_model')]:
        assert p[role]['model']==draft['base'][key]
        assert sha(ROOT/p[role]['model'])==draft['base'][key+'_sha256']
    cases=[('nominal',1.0,7.7),('friction-low',.8,7.7),('voltage-high',1.0,8.0)]
    configs,models,rows={},{},{}
    for name,friction,voltage in cases:
        configured=screen.parameter_configuration(route.configuration,
                    {'factors':{'foot_friction':friction,'vin_range':voltage}})
        cfg=configured(p)
        assert cfg.seed==1241 and cfg.scene.num_envs==128
        assert tuple(cfg.events['foot_friction'].params['ranges'])==(friction,friction)
        assert tuple(cfg.scene.entities['robot'].articulation.actuators[0].vin_range)==(voltage,voltage)
        configs[name]=stable(physical_description(cfg))
        # Compiles the CPU model only. No manager-based environment is built.
        model=Scene(cfg.scene,device='cpu').compile()
        models[name]={'fields':model_arrays(model),
                      'forcerange':np.asarray(model.actuator_forcerange).copy(),
                      'forcelimited':np.asarray(model.actuator_forcelimited).copy()}
        rows[name]={'cfg_seed':cfg.seed,'worlds':cfg.scene.num_envs,
                    'friction_event_range':list(cfg.events['foot_friction'].params['ranges']),
                    'voltage_config_range':list(cfg.scene.entities['robot'].articulation.actuators[0].vin_range),
                    'compiled_actuators':int(model.nu),
                    'compiled_actuator_force_range':models[name]['forcerange'].tolist(),
                    'compiled_actuator_forcelimited':models[name]['forcelimited'].tolist()}
        del model,cfg
    ref=models['nominal']; diffs={}
    for name in ('friction-low','voltage-high'):
        fields=models[name]['fields'];differences_in_model=[]
        assert set(fields)==set(ref['fields'])
        for key in fields:
            if not np.array_equal(fields[key],ref['fields'][key]):differences_in_model.append(key)
        diffs[name]={'physical_config_paths':differences(configs['nominal'],configs[name]),
                     'changed_checked_compiled_fields':differences_in_model,
                     'force_range_changed':not np.array_equal(ref['forcerange'],models[name]['forcerange']),
                     'force_limited_changed':not np.array_equal(ref['forcelimited'],models[name]['forcelimited'])}
    assert diffs['friction-low']['physical_config_paths']==['startup.foot_friction.params.ranges[0]','startup.foot_friction.params.ranges[1]']
    assert diffs['voltage-high']['physical_config_paths']==['robot.articulation.actuators[0].vin_range[0]','robot.articulation.actuators[0].vin_range[1]']
    assert not diffs['friction-low']['changed_checked_compiled_fields'] and not diffs['voltage-high']['changed_checked_compiled_fields']
    assert not diffs['friction-low']['force_range_changed'] and diffs['voltage-high']['force_range_changed']
    assert not diffs['friction-low']['force_limited_changed'] and not diffs['voltage-high']['force_limited_changed']
    np.testing.assert_allclose(models['voltage-high']['forcerange'],ref['forcerange']*(8.0/7.7),rtol=1e-6,atol=1e-7)
    assert not torch.cuda.is_initialized()
    names=[PLAN,DRAFT,Path(__file__),ROOT/'scripts/validation/run_continuous_roll_stand_walk.py',
           ROOT/'scripts/validation/run_roll_domain_parameter_screen.py',
           ROOT/'scripts/validation/common_scene_contract.py',ROOT/'scripts/validation/inspect_common_skill_plant.py',
           ROOT/p['roll']['model'],ROOT/p['walking']['model']]
    return {'status':'passed-cpu-configuration-and-compile-only','checked_at':datetime.now(timezone.utc).isoformat(),
            'cases':rows,'differences':diffs,'checked_compiled_fields':len(ref['fields']),
            'source_sha256':{x.relative_to(ROOT).as_posix():sha(x) for x in names},
            'seed_scope':'1241 is used only to reproduce the prior nominal configuration; choose a fresh seed for future evaluation.',
            'runtime_caveat':'Friction is applied by an event; per-world values, voltage tensors, RNG and actuator buffers require actual GPU zero-step readback.',
            'no_cuda_initialized':True,'physical_steps':0,'learning_updates':0,'hardware_io':False,
            'dynamic_execution_authorized':False}


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    output=OUT/'cpu-configuration.json';assert not output.exists()
    try:
        result=run()
    except BaseException as exc:
        result={'status':'failed-cpu-preflight','checked_at':datetime.now(timezone.utc).isoformat(),
                'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
                'physical_steps':0,'learning_updates':0,'hardware_io':False,'dynamic_execution_authorized':False}
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False));raise
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','checked_compiled_fields','differences','no_cuda_initialized','physical_steps')}))
