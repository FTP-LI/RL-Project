"""Check recording-before-validation and nonzero prior torque on CPU fixtures."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import roll_entry_snapshot
from roll_factor_start_fingerprint import REQUIRED
from roll_factor_zero_recorded_hook import recorded_actors

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'records/reports/roll-matched-factor-preflight-20261116'
PLAN=ROOT/'configs/experiments/roll_factor_fp_zero_nominal_s1242_20261116.json'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    out=REPORT/'recorded-hook-fixture.json';assert not out.exists() and not torch.cuda.is_initialized()
    slots=[(name,SimpleNamespace(value=np.array([i],dtype=np.float32)),'value') for i,name in enumerate(sorted(REQUIRED))]
    slots.append(('sensor.feet_ground_contact.valid',SimpleNamespace(value=torch.ones(2)),'value'))
    actuator=SimpleNamespace(vin_tensor=torch.full((2,1),7.7),vin_drop_gain=torch.full((2,1),.1),
                             _prev_motor_torque=torch.full((2,14),.25))
    model=SimpleNamespace(nu=14,actuator_forcerange=np.tile([-.978,.978],(14,1)),actuator_forcelimited=np.ones(14,dtype=bool))
    env=SimpleNamespace(device='cpu',common_step_counter=0,_sim_step_counter=0,num_envs=2,
                        scene={'robot':SimpleNamespace(actuators=[actuator])},sim=SimpleNamespace(mj_model=model))
    returned=({'roll':'network'},{'roll':'home'},'sim-home','limits')
    def actors(*args):return returned
    base=SimpleNamespace(actors=actors,PolicyControl=lambda *args:SimpleNamespace(previous=torch.ones(2)))
    plan={'run_id':'fixture','parameter_factors':{'vin_range_v':7.7}}
    rejected=[]
    with tempfile.TemporaryDirectory(prefix='recorded-hook-fixture-',dir=REPORT) as tmp:
        folder=Path(tmp)
        with patch.object(roll_entry_snapshot,'roll_slots',return_value=(slots,{})):
            good=folder/'good';good.mkdir()
            with recorded_actors(base,PLAN,lambda:True) as state:
                assert base.actors(env,None,plan,good)==returned and state['completed']
            assert base.actors is actors
            meta=json.loads((good/'actuator-observed-start.json').read_text())
            assert not meta['errors'] and meta['previous_torque_zero_required'] is False
            with np.load(good/'actuator-observed-start.npz') as a:assert np.all(a['previous_motor_torque']==.25)
            assert sha(good/'actuator-observed-start.npz')==meta['arrays_sha256']
            for name,field,value,reason in [('wrong-voltage','vin_tensor',torch.full((2,1),8.),'voltage-value'),
                    ('bad-torque-shape','_prev_motor_torque',torch.zeros((2,13)),'shape:previous_motor_torque'),
                    ('nonfinite-torque','_prev_motor_torque',torch.full((2,14),float('nan')),'nonfinite-or-dtype:previous_motor_torque'),
                    ('missing-gain','vin_drop_gain',None,'missing:vin_drop_gain')]:
                old=getattr(actuator,field);setattr(actuator,field,value);case=folder/name;case.mkdir()
                with recorded_actors(base,PLAN,lambda:True):
                    try:base.actors(env,None,plan,case)
                    except ValueError:pass
                    else:raise AssertionError('Invalid actuator data accepted: '+name)
                saved=json.loads((case/'actuator-observed-start.json').read_text())
                assert reason in saved['errors'] and sha(case/'actuator-observed-start.npz')==saved['arrays_sha256']
                assert not (case/'start-fingerprints.json').exists()
                rejected.append(name);setattr(actuator,field,old)
    assert not torch.cuda.is_initialized()
    sources=[Path(__file__),ROOT/'scripts/validation/roll_factor_zero_recorded_hook.py',
             ROOT/'scripts/validation/roll_factor_start_fingerprint.py',PLAN,
             REPORT/'nominal-zero-stop-analysis.json']
    result={'status':'passed-cpu-recorded-hook-fixture-only','checked_at':datetime.now(timezone.utc).isoformat(),
            'nonzero_prior_torque_preserved':True,'invalid_arrays_saved_before_refusal':True,
            'rejected':rejected,'actor_hook_restored':base.actors is actors,'temporary_fixture_removed':True,
            'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            'physical_steps':0,'learning_updates':0,'cuda_initialized':False,'hardware_io':False,
            'dynamic_execution_authorized':False,'scope':'CPU fake environment; real captured values and cause still require a separately admitted zero diagnostic.'}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
