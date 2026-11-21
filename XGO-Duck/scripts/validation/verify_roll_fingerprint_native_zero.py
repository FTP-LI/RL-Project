"""Check the new zero wrapper's scope and hook restoration with CPU fixtures."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import roll_factor_zero_fingerprint_hook as hook
import roll_fingerprint_native_zero as entry

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'records/reports/roll-matched-factor-preflight-20261116'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    out=REPORT/'native-zero-wrapper-fixture.json';assert not out.exists() and not torch.cuda.is_initialized()
    plan={'stage':'fixture'};name='configs/experiments/fixture.json'
    project={'goal_execution':{'status':'active','pause_requested_by_user':False},
             'staged_training_request':{'pause_requested':False,'learning_running':False,'simulation_running':False,
                 'current_protocol_frozen':True,'current_protocol':name,'current_stage':'fixture'}}
    assert entry.project_contract(project,plan,name)
    rejected=[]
    for label,branch,key,value in [('paused','staged_training_request','pause_requested',True),
            ('goal-paused','goal_execution','status','paused'),('learning','staged_training_request','learning_running',True),
            ('simulation','staged_training_request','simulation_running',True),
            ('unfrozen','staged_training_request','current_protocol_frozen',False),
            ('wrong-stage','staged_training_request','current_stage','other'),
            ('wrong-plan','staged_training_request','current_protocol','other')]:
        changed=deepcopy(project);changed[branch][key]=value
        assert not entry.project_contract(changed,plan,name);rejected.append(label)
    with tempfile.TemporaryDirectory(prefix='native-zero-fixture-',dir=REPORT) as tmp:
        folder=Path(tmp);up=folder/'plan.json';wrapper=folder/'wrapper.json';up.write_text('{}\n')
        up_name=up.relative_to(ROOT).as_posix();wrapper_name=wrapper.relative_to(ROOT).as_posix()
        wrapper.write_text(json.dumps({'upstream_plan':up_name,'upstream_plan_sha256':sha(up)}))
        original=(entry.repair.UPSTREAM,entry.repair.WRAPPER,entry.repair.REQUIRED_SOURCES,
                  entry.coordinator.provenance.audit,entry.coordinator.project_allows)
        with entry.bound(up_name,wrapper_name) as data:
            assert data['upstream_plan']==up_name and entry.repair.UPSTREAM==up_name
            assert entry.repair.WRAPPER==wrapper_name and entry.repair.REQUIRED_SOURCES==entry.REQUIRED_SOURCES
        assert original==(entry.repair.UPSTREAM,entry.repair.WRAPPER,entry.repair.REQUIRED_SOURCES,
                          entry.coordinator.provenance.audit,entry.coordinator.project_allows)
        wrapper.write_text(json.dumps({'upstream_plan':up_name,'upstream_plan_sha256':'0'*64}))
        try:
            with entry.bound(up_name,wrapper_name):pass
        except ValueError:rejected.append('wrapper-digest')
        else:raise AssertionError('Wrong wrapper digest accepted')

    @contextmanager
    def fixture_hook(base,protocol,may_continue):
        yield {'calls':1,'completed':True,'proof':{'fingerprint_sha256':'a'*64,'arrays_sha256':'b'*64}}
    def zero(*args):return {'status':'zero-completed-awaiting-independent-verification'}
    with patch.object(entry.old,'execute_zero',zero),patch.object(hook,'fingerprint_actors',fixture_hook):
        with entry.add_fingerprint_hook():
            result=entry.old.execute_zero('plan',{}, {},(SimpleNamespace(),),lambda:True)
            assert result['start_fingerprint']=='a'*64 and result['actuator_start_state']=='b'*64
        assert entry.old.execute_zero is zero
    assert not torch.cuda.is_initialized()
    sources=[Path(__file__),ROOT/'scripts/validation/roll_fingerprint_native_zero.py',
             ROOT/'scripts/validation/roll_factor_zero_fingerprint_hook.py',
             ROOT/'scripts/validation/roll_factor_start_fingerprint.py']
    result={'status':'passed-cpu-native-zero-wrapper-fixture-only','checked_at':datetime.now(timezone.utc).isoformat(),
            'refusals':rejected,'wrapper_and_hook_restored':True,'fingerprint_composition_verified':True,
            'temporary_fixture_removed':True,'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            'physical_steps':0,'learning_updates':0,'cuda_initialized':False,'hardware_io':False,
            'dynamic_execution_authorized':False,'scope':'CPU fixtures only; real plan, observer and GPU zero require separate checks.'}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
