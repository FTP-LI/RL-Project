"""Exercise zero-step fingerprint boundaries using isolated CPU fixtures."""
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
from roll_factor_start_fingerprint import REQUIRED,capture

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'records/reports/roll-matched-factor-preflight-20261116'
PLAN=ROOT/'configs/experiments/roll_native_nominal_observed_s1241_20261115.json'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    output=REPORT/'start-fingerprint-fixture.json';assert not output.exists() and not torch.cuda.is_initialized()
    env=SimpleNamespace(device='cpu',common_step_counter=0,_sim_step_counter=0,num_envs=2)
    owners=[SimpleNamespace(value=np.array([i],dtype=np.float32)) for i,_ in enumerate(sorted(REQUIRED))]
    slots=[(name,owner,'value') for name,owner in zip(sorted(REQUIRED),owners,strict=True)]
    slots.append(('sensor.feet_ground_contact.valid',SimpleNamespace(value=torch.tensor([1,1])),'value'))
    generator=torch.Generator(device='cpu').manual_seed(32); generators={'delay':generator}
    rejected=[]
    with tempfile.TemporaryDirectory(prefix='fingerprint-fixture-',dir=REPORT) as tmp:
        folder=Path(tmp)
        with patch.object(roll_entry_snapshot,'roll_slots',return_value=(slots,generators)):
            a=capture(env,None,folder/'a.json','fixture-a',PLAN)
            b=capture(env,None,folder/'b.json','fixture-b',PLAN)
        assert a['fields_digest']==b['fields_digest'] and a['rng_digest']==b['rng_digest']
        saved=json.loads((folder/'a.json').read_text(encoding='utf-8'))
        assert saved['protocol_sha256']==sha(PLAN) and saved['counters']==[0,0] and saved['worlds']==2

        def refuse(label,chosen_slots,chosen_env=env,chosen_fn=None):
            path=folder/(label+'.json')
            with patch.object(roll_entry_snapshot,'roll_slots',chosen_fn or (lambda *_:(chosen_slots,generators))):
                try:capture(chosen_env,None,path,'fixture',PLAN)
                except (ValueError,RuntimeError):rejected.append(label)
                else:raise AssertionError('Invalid capture accepted: '+label)
            assert not path.exists()

        refuse('missing-field',slots[1:])
        refuse('duplicate-field',slots+[slots[0]])
        refuse('already-stepped',slots,SimpleNamespace(device='cpu',common_step_counter=1,_sim_step_counter=5,num_envs=2))
        cpu_state=torch.get_rng_state().clone()
        def bad_inventory(*_):
            torch.rand(1)
            return slots,generators
        refuse('rng-changed-during-inventory',slots,chosen_fn=bad_inventory)
        torch.set_rng_state(cpu_state)
        class MutatingOwner:
            @property
            def value(self):
                torch.rand(1)
                return np.array([1],dtype=np.float32)
        altered=list(slots);altered[0]=(altered[0][0],MutatingOwner(),'value')
        refuse('rng-changed-during-read',altered)
        torch.set_rng_state(cpu_state)
    assert not torch.cuda.is_initialized()
    sources=[Path(__file__),ROOT/'scripts/validation/roll_factor_start_fingerprint.py',
             ROOT/'scripts/validation/roll_entry_snapshot.py',ROOT/'scripts/validation/getup_release_fork_state.py',PLAN]
    result={'status':'passed-cpu-fingerprint-fixture-only','checked_at':datetime.now(timezone.utc).isoformat(),
            'fields_checked':a['field_count'],'identical_state_and_rng_fingerprints':True,'rejected':rejected,
            'temporary_fixture_removed':True,'source_sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in sources},
            'physical_steps':0,'learning_updates':0,'cuda_initialized':False,'hardware_io':False,
            'dynamic_execution_authorized':False,'scope':'Synthetic CPU slots; actual CUDA inventory and cross-factor comparisons remain unverified.'}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
