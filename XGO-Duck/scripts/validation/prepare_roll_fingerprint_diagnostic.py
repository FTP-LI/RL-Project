"""Freeze one new zero-only diagnosis; keep the stopped attempt unchanged."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import roll_fingerprint_diagnostic_zero as diagnostic

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'records/reports/roll-matched-factor-preflight-20261116'
FAILED=ROOT/'configs/experiments/roll_factor_fp_zero_nominal_s1242_20261116.json'
FAILED_WRAPPER=ROOT/'configs/experiments/roll_factor_fp_native_nominal_s1242_20261116.json'
PLAN=ROOT/'configs/experiments/roll_factor_fp_diagnostic_zero_nominal_s1242_20261116.json'
WRAPPER=ROOT/'configs/experiments/roll_factor_fp_diagnostic_native_nominal_s1242_20261116.json'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def encode(obj):return (json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


def main():
    index=REPORT/'diagnostic-zero-plan.json';assert not PLAN.exists() and not WRAPPER.exists() and not index.exists()
    project=read(ROOT/'project.json');t=project['staged_training_request'];g=project['goal_execution']
    assert g['status']=='active' and g['pause_requested_by_user'] is False and t['pause_requested'] is False
    assert not t['learning_running'] and not t['simulation_running'] and not t['current_protocol_frozen']
    assert t['current_stage']=='roll-parameter-zero-diagnostic-preparation'
    stop=diagnostic.diagnostic_evidence();fixture=read(REPORT/'diagnostic-wrapper-fixture.json')
    assert fixture['status']=='passed-cpu-diagnostic-wrapper-fixture-only'
    for name,digest in fixture['source_sha256'].items():assert sha(ROOT/name)==digest,name
    base=read(FAILED);old_wrapper=read(FAILED_WRAPPER)
    assert base['seed']==1242 and base['factor_arm']=='nominal' and old_wrapper['upstream_plan_sha256']==sha(FAILED)
    assert stop['protocol_sha256']==sha(FAILED) and stop['zero_physics_substeps']==0
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id='duck_roll-factor_nominal_s1242_'+stamp;output=ROOT/'records/reports'/run_id
    assert not output.exists()
    p=deepcopy(base);p.update(run_id=run_id,guarded_output=output.relative_to(ROOT).as_posix(),
         page=f'records/{run_id}-live.html',admission_record=f'records/reports/{run_id}/admission.json')
    for source in stop['source_checks']:
        if not source['pinned_by_failed_plan']:
            assert sha(Path(source['path']))==source['sha256']
            p['library_source_sha256'][source['path']]=source['sha256']
    extras=diagnostic.REQUIRED|{'scripts/validation/prepare_roll_fingerprint_diagnostic.py',
        'scripts/validation/verify_roll_fingerprint_diagnostic_zero.py',
        'records/reports/roll-matched-factor-preflight-20261116/diagnostic-wrapper-fixture.json',
        'configs/experiments/roll_factor_fp_zero_nominal_s1242_20261116.json',
        'records/reports/duck_roll-factor_nominal_s1242_20261116T133133/native-zero-execution.log',
        'records/reports/duck_roll-factor_nominal_s1242_20261116T133133/native-zero-run/result.json',
        'records/reports/duck_roll-factor_nominal_s1242_20261116T133133/zero/runtime-contract.json'}
    p['frozen_sha256'].update({name:sha(ROOT/name) for name in sorted(extras)})
    plan_bytes=encode(p);plan_hash=hashlib.sha256(plan_bytes).hexdigest()
    plan_name=PLAN.relative_to(ROOT).as_posix();wrapper_name=WRAPPER.relative_to(ROOT).as_posix()
    wrapper=deepcopy(old_wrapper);wrapper.update(created_at=datetime.now(timezone.utc).isoformat(),
         upstream_plan=plan_name,upstream_plan_sha256=plan_hash,
         source_sha256={name:sha(ROOT/name) for name in sorted(diagnostic.REQUIRED)},
         scope='One new observed zero-only diagnostic records actuator state before checks; no physical steps or learning.')
    PLAN.write_bytes(plan_bytes);WRAPPER.write_bytes(encode(wrapper));output.mkdir()
    (output/'windows-entry.json').write_bytes(encode({'status':'native-diagnostic-prepared',
         'protocol_sha256':plan_hash,'browser_visual_verified':False,'real_observer_verified':False}))
    fake=deepcopy(project);fake['staged_training_request'].update(current_stage=p['stage'],current_protocol=plan_name,
         current_protocol_frozen=True,learning_running=False,simulation_running=False)
    fake['goal_execution']['native_file_observer_evaluation'].update(native_zero_plan=wrapper_name,native_zero_plan_sha256=sha(WRAPPER))
    original_read=diagnostic.old.read
    def fake_read(path):return deepcopy(fake) if Path(path).resolve()==(ROOT/'project.json').resolve() else original_read(path)
    with patch.object(diagnostic.old,'read',side_effect=fake_read),diagnostic.diagnostic_bound(plan_name,wrapper_name):
        with diagnostic.repair.repair_context(wrapper):
            assert diagnostic.coordinator.verify_wrapper(WRAPPER,wrapper,fake)==sha(WRAPPER)
            assert diagnostic.coordinator.provenance.audit()['status']=='source-audit-passed-native-zero-only'
    result={'status':'frozen-separate-diagnostic-awaiting-observer-and-registration',
            'created_at':datetime.now(timezone.utc).isoformat(),'run_id':run_id,'seed':1242,
            'seed_rationale':'Same seed isolates the changed recording/check code; this is a new zero-only diagnosis with a new run ID, not a restart of the stopped process.',
            'plan':plan_name,'plan_sha256':sha(PLAN),'wrapper':wrapper_name,'wrapper_sha256':sha(WRAPPER),
            'output':output.relative_to(ROOT).as_posix(),'previous_stop':stop['run_id'],
            'newly_pinned_runtime_sources':stop['unpinned_runtime_sources'],
            'physical_steps':0,'learning_updates':0,'hardware_io':False,'automatic_retry':False,
            'dynamic_execution_authorized':False}
    index.write_bytes(encode(result));print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
