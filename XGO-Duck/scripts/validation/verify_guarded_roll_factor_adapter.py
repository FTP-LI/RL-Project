"""Check scoped factor hooks on real CPU models and archived zero arrays.

Fake environment objects exercise constructor refusal and cleanup. This does
not establish actual GPU entry, observer, zero-state matching, or dynamics.
"""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/validation'))
sys.path.insert(0,str(ROOT/'scripts/training'))
OUT=ROOT/'records/reports/roll-matched-factor-preflight-20261116/adapter-check.json'
PLAN=ROOT/'configs/experiments/roll_native_nominal_observed_s1241_20261115.json'
BASE=ROOT/'records/reports/common-plant-screen/startup/roll/nominal-primary-fields.npz'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    import numpy as np
    import torch
    from mjlab.scene import Scene
    import common_scene_contract as scene
    import run_continuous_roll_stand_walk as route
    import run_roll_domain_parameter_screen as screen
    from guarded_roll_factor_adapter import CASES,expected_factors,factor_hooks
    assert not OUT.exists() and not torch.cuda.is_initialized()
    prior=json.loads((OUT.parent/'cpu-configuration.json').read_text(encoding='utf-8'))
    assert prior['status']=='passed-cpu-configuration-and-compile-only'
    p=json.loads(PLAN.read_text(encoding='utf-8'))
    with np.load(BASE,allow_pickle=False) as f:primary={k:f[k].copy() for k in f.files}
    configs={};models={}
    for name,(friction,voltage) in CASES.items():
        cfg=screen.parameter_configuration(route.configuration,{'factors':{'foot_friction':friction,'vin_range':voltage}})(p)
        configs[name]=cfg;models[name]=Scene(cfg.scene,device='cpu').compile()
    nominal=models['nominal'];nominal_fields=scene.model_arrays(nominal)
    class OriginalEnvironment:pass
    env_module=SimpleNamespace(ManagerBasedRlEnv=OriginalEnvironment)
    closed=[]
    class FakeEnv:
        def __init__(self,model):self.sim=SimpleNamespace(mj_model=model)
        def close(self):closed.append(True)
    def fake_install(*args,**kwargs):
        env_module.ManagerBasedRlEnv=FakeEnv
        return FakeEnv
    fake_route=SimpleNamespace(configuration=route.configuration,install_guard=fake_install)
    original_cfg=fake_route.configuration;original_installer=fake_route.install_guard;original_compare=scene.compare_arrays
    cases=[]
    for name,(friction,voltage) in CASES.items():
        spec=deepcopy(p);spec.update(factor_arm=name,parameter_factors={'foot_friction':friction,'vin_range_v':voltage})
        with factor_hooks(fake_route,scene,screen,env_module,spec,nominal,models[name]) as proof:
            cfg=fake_route.configuration(spec)
            assert tuple(cfg.events['foot_friction'].params['ranges'])==(friction,friction)
            assert tuple(cfg.scene.entities['robot'].articulation.actuators[0].vin_range)==(voltage,voltage)
            # The 33-field compiled model check remains exact, including all friction values.
            scene.compare_arrays(scene.model_arrays(models[name]),nominal_fields)
            adjusted={k:v.copy() for k,v in primary.items()}
            adjusted['geom_friction'][list(proof['foot_geom_ids']),0]=friction
            scene.compare_arrays(adjusted,primary)
            corrupt={k:v.copy() for k,v in adjusted.items()};corrupt['geom_friction'][0,0]+=.1
            try:scene.compare_arrays(corrupt,primary)
            except (ValueError,AssertionError):pass
            else:raise AssertionError('Non-foot friction was ignored')
            strict={k:v.copy() for k,v in nominal_fields.items()};strict['geom_friction'][0,0]+=.1
            try:scene.compare_arrays(strict,nominal_fields)
            except (ValueError,AssertionError):pass
            else:raise AssertionError('Compiled scene friction was ignored')
            cls=fake_route.install_guard(None,None,None,None);instance=cls(models[name])
            assert instance.sim.mj_model is models[name]
            if name=='voltage-high':
                before=len(closed)
                try:cls(nominal)
                except (ValueError,AssertionError):pass
                else:raise AssertionError('Wrong runtime force limit was accepted')
                assert len(closed)==before+1
        assert fake_route.configuration is original_cfg and fake_route.install_guard is original_installer
        assert scene.compare_arrays is original_compare and env_module.ManagerBasedRlEnv is OriginalEnvironment
        cases.append({'arm':name,'foot_geom_ids':proof['foot_geom_ids'],'configuration_and_strict_compare':'passed',
                      'constructor_check_and_restoration':'passed'})
    invalid=[{'factor_arm':'unknown','parameter_factors':{'foot_friction':1.,'vin_range_v':7.7}},
             {'factor_arm':'friction-low','parameter_factors':{'foot_friction':.8,'vin_range_v':8.0}},
             {'factor_arm':'nominal','parameter_factors':{'foot_friction':True,'vin_range_v':7.7}},
             {'factor_arm':'nominal','parameter_factors':{'foot_friction':1.,'vin_range_v':7.7,'extra':1}}]
    for spec in invalid:
        try:expected_factors(spec)
        except ValueError:continue
        raise AssertionError('Invalid factor plan was accepted')
    assert not torch.cuda.is_initialized()
    sources=[Path(__file__),ROOT/'scripts/validation/guarded_roll_factor_adapter.py',
             ROOT/'scripts/validation/roll_parameter_array_contract.py',PLAN,BASE,OUT.parent/'cpu-configuration.json']
    result={'status':'passed-cpu-adapter-fixture-only','checked_at':datetime.now(timezone.utc).isoformat(),
            'cases':cases,'invalid_plans_rejected':len(invalid),'wrong_runtime_force_limit_rejected_and_closed':True,
            'hooks_restored':True,'source_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in sources},
            'physical_steps':0,'learning_updates':0,'cuda_initialized':False,'hardware_io':False,
            'dynamic_execution_authorized':False,'scope':'CPU compiled models and fake environment constructor; real guarded entry and GPU zero remain unverified.'}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
