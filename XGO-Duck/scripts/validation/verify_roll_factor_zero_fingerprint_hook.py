"""Verify the zero actor hook on an isolated CPU fixture, without a simulator."""
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
from roll_factor_zero_fingerprint_hook import fingerprint_actors

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'records/reports/roll-matched-factor-preflight-20261116'
PLAN=ROOT/'configs/experiments/roll_native_nominal_observed_s1241_20261115.json'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    out=REPORT/'zero-fingerprint-hook-fixture.json';assert not out.exists() and not torch.cuda.is_initialized()
    owners=[SimpleNamespace(value=np.array([i],dtype=np.float32)) for i,_ in enumerate(sorted(REQUIRED))]
    slots=[(name,owner,'value') for name,owner in zip(sorted(REQUIRED),owners,strict=True)]
    slots.append(('sensor.feet_ground_contact.valid',SimpleNamespace(value=torch.ones(2)),'value'))
    actuator=SimpleNamespace(vin_tensor=torch.full((2,1),7.7),vin_drop_gain=torch.full((2,1),.1),
                             _prev_motor_torque=torch.zeros((2,14)))
    model=SimpleNamespace(actuator_forcerange=np.tile([-.978,.978],(14,1)),actuator_forcelimited=np.ones(14,dtype=bool))
    env=SimpleNamespace(device='cpu',common_step_counter=0,_sim_step_counter=0,num_envs=2,
                        scene={'robot':SimpleNamespace(actuators=[actuator])},sim=SimpleNamespace(mj_model=model))
    returned=({'roll':'network'},{'roll':'home'},'sim-home','limits')
    def actors(*args):return returned
    def controller(*args):return SimpleNamespace(previous=torch.ones(2))
    base=SimpleNamespace(actors=actors,PolicyControl=controller)
    plan={'run_id':'fixture','parameter_factors':{'vin_range_v':7.7}}
    rejected=[]
    with tempfile.TemporaryDirectory(prefix='zero-hook-fixture-',dir=REPORT) as tmp:
        folder=Path(tmp)
        with patch.object(roll_entry_snapshot,'roll_slots',return_value=(slots,{})):
            with fingerprint_actors(base,PLAN,lambda:True) as state:
                assert base.actors(env,None,plan,folder)==returned and state['completed']
                proof=json.loads((folder/'actuator-start-state.json').read_text())
                assert proof['arrays_sha256']==sha(folder/'actuator-start-state.npz')
                assert proof['fingerprint_sha256']==sha(folder/'start-fingerprints.json')
                try:base.actors(env,None,plan,folder)
                except RuntimeError:rejected.append('repeat')
                else:raise AssertionError('Repeated actor hook accepted')
            assert base.actors is actors

            def reject(label,env_value=env,allow=lambda:True):
                child=folder/label;child.mkdir()
                with fingerprint_actors(base,PLAN,allow):
                    try:base.actors(env_value,None,plan,child)
                    except (ValueError,RuntimeError):rejected.append(label)
                    else:raise AssertionError('Invalid hook accepted: '+label)
                assert base.actors is actors
            reject('pause',allow=lambda:False)
            wrong=SimpleNamespace(**vars(env));wrong.common_step_counter=1
            reject('already-stepped',wrong)
            actuator.vin_tensor=torch.full((2,1),8.0);reject('wrong-voltage');actuator.vin_tensor=torch.full((2,1),7.7)
            actuator._prev_motor_torque=torch.ones((2,14));reject('nonzero-prior-torque');actuator._prev_motor_torque=torch.zeros((2,14))
            def rng_controller(*args):torch.rand(1);return None
            rng=torch.get_rng_state().clone();base.PolicyControl=rng_controller
            reject('controller-consumed-rng');torch.set_rng_state(rng);base.PolicyControl=controller
    assert not torch.cuda.is_initialized()
    sources=[Path(__file__),ROOT/'scripts/validation/roll_factor_zero_fingerprint_hook.py',
             ROOT/'scripts/validation/roll_factor_start_fingerprint.py',PLAN]
    result={'status':'passed-cpu-zero-hook-fixture-only','checked_at':datetime.now(timezone.utc).isoformat(),
            'actor_result_preserved':True,'fingerprint_and_actuator_hashes_verified':True,'rejected':rejected,
            'original_actor_hook_restored':True,'temporary_fixture_removed':True,
            'source_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in sources},
            'physical_steps':0,'learning_updates':0,'cuda_initialized':False,'hardware_io':False,
            'dynamic_execution_authorized':False,'scope':'CPU fake environment; actual zero entry and GPU capture remain unverified.'}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
