"""Verify refusal logging preserves the original decisions on CPU fixtures."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import roll_zero_guard_rejection_log as module

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'records/reports/roll-matched-factor-preflight-20261116'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    out=REPORT/'guard-rejection-log-fixture.json';assert not out.exists()
    child=SimpleNamespace(pid=23);plan={'stage':'fixture'};allowed={'allowed':True,'reason':None}
    denied={'allowed':False,'reason':'observer-heartbeat-stale'}
    with tempfile.TemporaryDirectory(prefix='guard-log-fixture-',dir=REPORT) as tmp:
        root=Path(tmp);health=root/'health.json';health.write_text('{"health":"stale"}\n');events=root/'events.jsonl'
        with patch.object(module.coordinator,'check_child',return_value=allowed) as child_fn, \
             patch.object(module.coordinator,'project_allows',return_value=True) as project_fn:
            with module.log_refusals(events):
                assert module.coordinator.check_child(child,health)==allowed
                assert module.coordinator.project_allows(plan,'abc',running=False) is True
                assert not events.exists()
                child_fn.return_value=denied;project_fn.return_value=False
                with patch.object(module,'_project_snapshot',return_value={'scope':'after refusal'}):
                    assert module.coordinator.check_child(child,health)==denied
                    assert module.coordinator.project_allows(plan,'abc',running=False) is False
            assert module.coordinator.check_child is child_fn and module.coordinator.project_allows is project_fn
        rows=[json.loads(line) for line in events.read_text().splitlines()]
        assert len(rows)==2 and rows[0]['component']=='observer' and rows[0]['original_decision']==denied
        assert rows[0]['health_after_refusal']=={'health':'stale'}
        assert rows[1]['component']=='project' and rows[1]['original_allowed'] is False
        assert rows[1]['project_after_refusal']['scope']=='after refusal'
    sources=[Path(__file__),ROOT/'scripts/validation/roll_zero_guard_rejection_log.py',
             ROOT/'scripts/validation/native_zero_coordinator.py']
    result={'status':'passed-cpu-refusal-log-fixture-only','checked_at':datetime.now(timezone.utc).isoformat(),
            'allow_unchanged':True,'refusals_unchanged':True,'reason_and_post_failure_snapshot_recorded':True,
            'hooks_restored':True,'temporary_fixture_removed':True,
            'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            'physical_steps':0,'learning_updates':0,'hardware_io':False,
            'dynamic_execution_authorized':False,'scope':'CPU fixture; the previous transient refusal cannot be reconstructed.'}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
