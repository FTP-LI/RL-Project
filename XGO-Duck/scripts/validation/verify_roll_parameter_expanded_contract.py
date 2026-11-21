"""Check every saved environment in the historical low-friction zero state.

The nominal per-world reference is broadcast without changing source data.
Injected faults must be rejected in any world, not only in world zero.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from roll_parameter_array_contract import compare_primary

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'records/reports/roll-matched-factor-preflight-20261116'
BASE=ROOT/'records/reports/common-plant-screen/startup/roll/nominal-primary-fields.npz'
EXPANDED=ROOT/'records/reports/roll-domain-parameter-screen/friction-low/zero/expanded-fields-before-physics.npz'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    out=REPORT/'expanded-contract.json';assert not out.exists()
    prior=json.loads((REPORT/'array-contract.json').read_text(encoding='utf-8'))
    ids=prior['foot_geom_ids'];assert prior['status']=='passed-saved-arrays-and-cpu-model-only'
    with np.load(BASE,allow_pickle=False) as f:base={k:f[k].copy() for k in f.files}
    with np.load(EXPANDED,allow_pickle=False) as f:actual={k:f[k].copy() for k in base}
    worlds=actual['geom_friction'].shape[0];assert worlds==128
    reference={k:np.broadcast_to(v,(worlds,*v.shape)).copy() for k,v in base.items()}
    compare_primary(actual,reference,ids,.7)
    rejected=[]
    def check(label,key,index,value):
        bad={k:v.copy() for k,v in actual.items()};bad[key][index]=value
        try:compare_primary(bad,reference,ids,.7)
        except (AssertionError,ValueError):rejected.append(label);return
        raise AssertionError('Invalid case accepted: '+label)
    check('one-world-foot-value','geom_friction',(127,ids[0],0),.8)
    check('one-world-other-geom','geom_friction',(64,0,0),.8)
    check('one-world-nonfinite','geom_friction',(42,ids[1],0),np.nan)
    index=(37,0);check('one-world-damping','dof_damping',index,float(actual['dof_damping'][index])+1)
    result={'status':'passed-all-saved-worlds-and-injected-faults','checked_at':datetime.now(timezone.utc).isoformat(),
            'worlds_checked':worlds,'foot_geom_ids':ids,'historical_friction':.7,
            'faults_rejected':rejected,'source_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in
                (Path(__file__),ROOT/'scripts/validation/roll_parameter_array_contract.py',BASE,EXPANDED,REPORT/'array-contract.json')},
            'physical_steps':0,'learning_updates':0,'cuda_initialized':False,'hardware_io':False,
            'dynamic_execution_authorized':False,'scope':'Saved historical GPU arrays only; new factor runtime still requires actual zero-step readback.'}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
