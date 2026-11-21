"""Exercise strict factor comparison on archived GPU zero-step arrays.

Historical low friction is actual saved data; synthetic faults test rejection.
The compiled model resolves foot geometry IDs but no environment is started.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/validation'))
sys.path.insert(0,str(ROOT/'scripts/training'))
OUT=ROOT/'records/reports/roll-matched-factor-preflight-20261116/array-contract.json'
NOMINAL=ROOT/'records/reports/common-plant-screen/startup/roll/nominal-primary-fields.npz'
LOW=ROOT/'records/reports/roll-domain-parameter-screen/friction-low/zero/nominal-primary-fields.npz'
PLAN=ROOT/'configs/experiments/roll_native_nominal_observed_s1241_20261115.json'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    import mujoco
    import numpy as np
    import torch
    from mjlab.scene import Scene
    from roll_parameter_array_contract import compare_primary, compare_force_limits
    import run_continuous_roll_stand_walk as route
    import run_roll_domain_parameter_screen as screen
    assert not OUT.exists() and not torch.cuda.is_initialized()
    p=json.loads(PLAN.read_text(encoding='utf-8'))
    cfg=screen.parameter_configuration(route.configuration,{'factors':{'foot_friction':1.0,'vin_range':7.7}})(p)
    model=Scene(cfg.scene,device='cpu').compile()
    names=('robot/left_foot_collision','robot/right_foot_collision')
    ids=[int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)) for name in names]
    assert all(i>=0 for i in ids) and len(set(ids))==2
    with np.load(NOMINAL,allow_pickle=False) as f:base={k:f[k].copy() for k in f.files}
    with np.load(LOW,allow_pickle=False) as f:low={k:f[k].copy() for k in f.files}
    compare_primary(base,base,ids,1.0)
    compare_primary(low,base,ids,.7)
    synthetic={k:v.copy() for k,v in base.items()}
    synthetic['geom_friction'][ids,0]=np.asarray(.8,dtype=synthetic['geom_friction'].dtype)
    compare_primary(synthetic,base,ids,.8)
    checks=[]
    def rejected(label,fn):
        try:fn()
        except (ValueError,AssertionError):checks.append(label);return
        raise AssertionError('Invalid case was accepted: '+label)
    def changed(field,index,value):
        data={k:v.copy() for k,v in synthetic.items()};data[field][index]=value;return data
    rejected('nonfoot-friction-changed',lambda:compare_primary(changed('geom_friction',(0,0),.8),base,ids,.8))
    rejected('foot-rolling-friction-changed',lambda:compare_primary(changed('geom_friction',(ids[0],1),.123),base,ids,.8))
    rejected('nonfinite-friction',lambda:compare_primary(changed('geom_friction',(ids[0],0),np.nan),base,ids,.8))
    rejected('missing-field',lambda:compare_primary({'geom_friction':synthetic['geom_friction']},base,ids,.8))
    rejected('wrong-factor-value',lambda:compare_primary(synthetic,base,ids,.7))
    rejected('duplicate-foot-id',lambda:compare_primary(synthetic,base,[ids[0],ids[0]],.8))
    rejected('out-of-range-foot-id',lambda:compare_primary(synthetic,base,[ids[0],base['geom_friction'].shape[0]],.8))
    rejected('wrong-shape',lambda:compare_primary({**synthetic,'geom_friction':synthetic['geom_friction'][:-1]},base,ids,.8))
    fr=np.asarray(model.actuator_forcerange).copy();limited=np.asarray(model.actuator_forcelimited).copy()
    high_cfg=screen.parameter_configuration(route.configuration,{'factors':{'foot_friction':1.0,'vin_range':8.0}})(p)
    high_model=Scene(high_cfg.scene,device='cpu').compile()
    high=np.asarray(high_model.actuator_forcerange).copy();high_flags=np.asarray(high_model.actuator_forcelimited).copy()
    compare_force_limits(high,fr,high_flags,limited,8.0,7.7)
    rejected('wrong-force-range',lambda:compare_force_limits(fr,fr,high_flags,limited,8.0,7.7))
    rejected('disabled-force-limit',lambda:compare_force_limits(high,fr,np.zeros_like(high_flags),limited,8.0,7.7))
    assert not torch.cuda.is_initialized()
    sources=[Path(__file__),ROOT/'scripts/validation/roll_parameter_array_contract.py',NOMINAL,LOW,PLAN]
    result={'status':'passed-saved-arrays-and-cpu-model-only','checked_at':datetime.now(timezone.utc).isoformat(),
            'foot_geom_names':list(names),'foot_geom_ids':ids,'actual_historical_friction':.7,
            'synthetic_future_friction':.8,'real_saved_primary_fields':list(base),
            'faults_rejected':checks,'fault_count':len(checks),'compiled_force_count':int(model.nu),
            'source_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in sources},
            'physical_steps':0,'learning_updates':0,'cuda_initialized':False,'hardware_io':False,
            'dynamic_execution_authorized':False,'scope':'CPU and archived arrays; new factor runtime and GPU zero remain unverified.'}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
