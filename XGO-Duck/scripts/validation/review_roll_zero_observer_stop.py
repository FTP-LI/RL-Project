"""Read the stopped observer record and reconstruct current project checks.

The historical health file was overwritten at shutdown. This cannot determine
whether observer health or project authorization failed at the earlier moment.
Only a temporary in-memory project copy is used; no run is launched or resumed.
"""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import roll_fingerprint_diagnostic_zero as diagnostic

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'records/reports/roll-matched-factor-preflight-20261116'
RUN=ROOT/'records/reports/duck_roll-factor_nominal_s1242_20261116T133748'
INDEX=REPORT/'diagnostic-zero-plan.json'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    out=REPORT/'observer-stop-readback.json';assert not out.exists()
    index=json.loads(INDEX.read_text());plan_path=ROOT/index['plan'];wrapper_path=ROOT/index['wrapper']
    assert sha(plan_path)==index['plan_sha256'] and sha(wrapper_path)==index['wrapper_sha256']
    plan=json.loads(plan_path.read_text());wrapper=json.loads(wrapper_path.read_text())
    result=json.loads((RUN/'native-zero-run/result.json').read_text());receipt=json.loads((RUN/'native-zero-execution.json').read_text())
    assert result['status']=='stopped-review-required' and result['error']=='observer or project changed before UI confirmation'
    assert result['physics_substeps']==0 and result['learning_updates']==0 and not (RUN/'zero').exists()
    assert receipt['exit_code']==2 and receipt['log_sha256']==sha(RUN/'native-zero-execution.log')
    assert not (RUN/'native-zero-run/ui-proof.json').exists()
    health=json.loads((RUN/'native-zero-run/view/native-health.json').read_text())
    assert health['health']=='stopped' and health['run_id']==index['run_id']
    fake=json.loads((ROOT/'project.json').read_text());task=fake['staged_training_request'];goal=fake['goal_execution']
    assert task['current_protocol']==index['plan'] and goal['status']=='active'
    task.update(current_protocol_frozen=True,simulation_running=False,learning_running=False)
    original=diagnostic.old.read
    def fake_read(path):
        return deepcopy(fake) if Path(path).resolve()==(ROOT/'project.json').resolve() else original(path)
    with patch.object(diagnostic.old,'read',side_effect=fake_read), \
         diagnostic.diagnostic_bound(index['plan'],index['wrapper']):
        with diagnostic.repair.repair_context(wrapper):
            checks=[diagnostic.coordinator.project_allows(plan,index['wrapper_sha256'],running=False) for _ in range(10)]
    record={'status':'saved-observer-stop-reviewed','checked_at':datetime.now(timezone.utc).isoformat(),
            'run_id':index['run_id'],'log_sha256':receipt['log_sha256'],'physics_substeps':0,'learning_updates':0,
            'ui_proof_written':False,'health_after_exit':health['health'],
            'reconstructed_project_checks_allowed':sum(checks),'reconstructed_project_checks_total':len(checks),
            'reconstruction_scope':'Current files and an in-memory authorized project; not the historical instant.',
            'historical_rejection_cause_known':False,
            'new_physical_steps':0,'new_learning_updates':0,'hardware_io':False,'automatic_retry':False}
    out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(record))


if __name__=='__main__':main()
