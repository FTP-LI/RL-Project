"""Check the diagnostic wrapper and its evidence gate using CPU fixtures."""
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import roll_factor_zero_recorded_hook as hook
import roll_fingerprint_diagnostic_zero as diagnostic

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'records/reports/roll-matched-factor-preflight-20261116'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    out=REPORT/'diagnostic-wrapper-fixture.json';assert not out.exists() and not torch.cuda.is_initialized()
    stop=diagnostic.diagnostic_evidence();assert stop['actual_failing_condition_identified'] is False
    real_read=diagnostic.old.read
    def wrong(path):
        value=real_read(path)
        if Path(path).name=='recorded-hook-fixture.json':value['invalid_arrays_saved_before_refusal']=False
        return value
    with patch.object(diagnostic.old,'read',side_effect=wrong):
        try:diagnostic.diagnostic_evidence()
        except ValueError:pass
        else:raise AssertionError('Invalid diagnostic evidence accepted')
    with tempfile.TemporaryDirectory(prefix='diagnostic-wrapper-fixture-',dir=REPORT) as tmp:
        folder=Path(tmp);up=folder/'plan.json';wrapper=folder/'wrapper.json';up.write_text('{}\n')
        up_name=up.relative_to(ROOT).as_posix();wrapper_name=wrapper.relative_to(ROOT).as_posix()
        wrapper.write_text(json.dumps({'upstream_plan':up_name,'upstream_plan_sha256':sha(up)}))
        original=(diagnostic.entry.REQUIRED_SOURCES,diagnostic.repair.UPSTREAM,diagnostic.repair.WRAPPER,
                  diagnostic.coordinator.provenance.audit)
        with diagnostic.diagnostic_bound(up_name,wrapper_name):
            assert diagnostic.entry.REQUIRED_SOURCES==diagnostic.REQUIRED
            assert diagnostic.repair.UPSTREAM==up_name
        assert original==(diagnostic.entry.REQUIRED_SOURCES,diagnostic.repair.UPSTREAM,diagnostic.repair.WRAPPER,
                          diagnostic.coordinator.provenance.audit)
    @contextmanager
    def fixture_hook(base,protocol,may_continue):
        yield {'calls':1,'completed':True,'proof':{'fingerprint_sha256':'a'*64,'arrays_sha256':'b'*64}}
    def zero(*args):return {'status':'zero-completed-awaiting-independent-verification'}
    with patch.object(diagnostic.old,'execute_zero',zero),patch.object(hook,'recorded_actors',fixture_hook):
        with diagnostic.add_recorded_hook():
            result=diagnostic.old.execute_zero('plan',{}, {},(SimpleNamespace(),),lambda:True)
            assert result['start_fingerprint']=='a'*64 and result['actuator_start_state']=='b'*64
        assert diagnostic.old.execute_zero is zero
    assert not torch.cuda.is_initialized()
    sources=[Path(__file__),ROOT/'scripts/validation/roll_fingerprint_diagnostic_zero.py',
             ROOT/'scripts/validation/roll_factor_zero_recorded_hook.py',
             REPORT/'nominal-zero-stop-analysis.json',REPORT/'recorded-hook-fixture.json']
    result={'status':'passed-cpu-diagnostic-wrapper-fixture-only','checked_at':datetime.now(timezone.utc).isoformat(),
            'real_saved_evidence_verified':True,'invalid_evidence_refused':True,'wrapper_and_hook_restored':True,
            'temporary_fixture_removed':True,'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            'physical_steps':0,'learning_updates':0,'cuda_initialized':False,'hardware_io':False,
            'dynamic_execution_authorized':False,'scope':'CPU fixtures only; real zero diagnosis remains separately admitted.'}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
