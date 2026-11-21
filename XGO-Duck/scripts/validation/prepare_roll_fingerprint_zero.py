"""Freeze three fresh zero-only factor plans; this file never runs a model.

The existing, independently checked nominal plan and its sources remain
unchanged. Every new arm shares a fresh development seed and original action
limits. Real observation confirmation and actual CUDA zero are separate gates.
"""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import roll_fingerprint_native_zero as entry

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'records/reports/roll-matched-factor-preflight-20261116'
BASE=ROOT/'configs/experiments/roll_factor_zero_bool_s1241_20261115.json'
BASE_WRAPPER=ROOT/'configs/experiments/roll_factor_native_bool_zero_s1241_20261115.json'
SEED=1242


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def encode(obj):return (json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


def seeds(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key=='seed' and type(item) is int:yield item
            yield from seeds(item)
    elif isinstance(value,list):
        for item in value:yield from seeds(item)


def main():
    index=REPORT/'zero-plans.json';assert not index.exists()
    project=read(ROOT/'project.json');task=project['staged_training_request'];goal=project['goal_execution']
    assert goal['status']=='active' and goal['pause_requested_by_user'] is False
    assert task['pause_requested'] is False and task['learning_running'] is False and task['simulation_running'] is False
    assert task['current_protocol_frozen'] is False and task['current_stage']=='roll-parameter-matched-cpu-preflight'
    for path in (ROOT/'configs/experiments').glob('*.json'):
        try:obj=read(path)
        except (UnicodeError,json.JSONDecodeError):continue
        if SEED in set(seeds(obj)):raise ValueError('Seed already appears in existing experiment: '+str(path))
    if any(f'_s{SEED}_' in p.name for p in (ROOT/'records/evaluations').iterdir()):
        raise ValueError('Seed already appears in evaluation output')
    base=read(BASE);base_wrapper=read(BASE_WRAPPER)
    assert base_wrapper['upstream_plan_sha256']==sha(BASE) and base['seed']==1241
    for root,key in ((ROOT,'frozen_sha256'),(entry.WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
        allowed=ROOT.resolve() if root==ROOT else entry.WORK
        for name,expected in base[key].items():
            path=(root/name).resolve()
            if not path.is_relative_to(allowed) or not path.is_file() or sha(path)!=expected:
                raise ValueError('Previously frozen source differs: '+str(path))
    evidence=['start-fingerprint-fixture.json','zero-fingerprint-hook-fixture.json','native-zero-wrapper-fixture.json']
    for name in evidence:
        value=read(REPORT/name)
        assert value['status'].startswith('passed-cpu-') and value['physical_steps']==0
        for source,expected in value['source_sha256'].items():assert sha(ROOT/source)==expected,source
    extras=entry.REQUIRED_SOURCES|{'scripts/validation/prepare_roll_fingerprint_zero.py',
        'scripts/validation/verify_roll_fingerprint_native_zero.py',
        'records/reports/roll-matched-factor-preflight-20261116/native-zero-wrapper-fixture.json'}
    for name in extras:assert (ROOT/name).is_file(),name
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    draft=read(ROOT/entry.old.DRAFT);arms={a['name']:a for a in draft['arms']}
    staged=[]
    for role in ('nominal','friction-low','voltage-high'):
        run_id=f'duck_roll-factor_{role}_s{SEED}_{stamp}'
        plan_name=f'configs/experiments/roll_factor_fp_zero_{role}_s{SEED}_20261116.json'
        wrapper_name=f'configs/experiments/roll_factor_fp_native_{role}_s{SEED}_20261116.json'
        output=ROOT/'records/reports'/run_id
        assert not (ROOT/plan_name).exists() and not (ROOT/wrapper_name).exists() and not output.exists()
        p=deepcopy(base);p.update(stage='S20-roll-factor-zero-'+role,seed=SEED,run_id=run_id,
             factor_arm=role,parameter_factors={k:arms[role][k] for k in ('foot_friction','vin_range_v')},
             guarded_output=output.relative_to(ROOT).as_posix(),page=f'records/{run_id}-live.html',
             admission_record=f'records/reports/{run_id}/admission.json')
        p['roll']['seed']=SEED
        p['frozen_sha256'].update({name:sha(ROOT/name) for name in sorted(extras)})
        plan_bytes=encode(p);digest=hashlib.sha256(plan_bytes).hexdigest()
        w=deepcopy(base_wrapper);w.update(created_at=datetime.now(timezone.utc).isoformat(),
             upstream_plan=plan_name,upstream_plan_sha256=digest,
             source_sha256={name:sha(ROOT/name) for name in sorted(entry.REQUIRED_SOURCES)},
             scope='Native-observed zero only with per-field fingerprints; no physics, learning or hardware.')
        staged.append((role,p,plan_name,plan_bytes,w,wrapper_name,output))
    for role,p,plan_name,plan_bytes,w,wrapper_name,output in staged:
        (ROOT/plan_name).write_bytes(plan_bytes);(ROOT/wrapper_name).write_bytes(encode(w));output.mkdir()
        (output/'windows-entry.json').write_bytes(encode({'status':'native-zero-prepared',
            'protocol_sha256':sha(ROOT/plan_name),'browser_visual_verified':False,'real_observer_verified':False}))
        fake=deepcopy(project);fake['staged_training_request'].update(current_stage=p['stage'],current_protocol=plan_name,
            current_protocol_frozen=True,learning_running=False,simulation_running=False)
        fake['goal_execution']['native_file_observer_evaluation'].update(native_zero_plan=wrapper_name,
            native_zero_plan_sha256=sha(ROOT/wrapper_name))
        original_read=entry.old.read
        def fake_read(path):
            return deepcopy(fake) if Path(path).resolve()==(ROOT/'project.json').resolve() else original_read(path)
        with patch.object(entry.old,'read',side_effect=fake_read),entry.bound(plan_name,wrapper_name):
            with entry.repair.repair_context(w):
                assert entry.coordinator.verify_wrapper(ROOT/wrapper_name,w,fake)==sha(ROOT/wrapper_name)
                assert entry.coordinator.provenance.audit()['status']=='source-audit-passed-native-zero-only'
    result={'status':'frozen-zero-plans-awaiting-observer-and-registration','created_at':datetime.now(timezone.utc).isoformat(),
            'seed':SEED,'seed_scope':'new development zero only; no physics or holdout used',
            'arms':[{'name':role,'run_id':p['run_id'],'plan':plan_name,'plan_sha256':sha(ROOT/plan_name),
                     'wrapper':wrapper_name,'wrapper_sha256':sha(ROOT/wrapper_name),'output':output.relative_to(ROOT).as_posix()}
                    for role,p,plan_name,_,w,wrapper_name,output in staged],
            'base_protocol_sha256':sha(BASE),'workspace_sources_checked':len(base['frozen_sha256']),
            'training_sources_checked':len(base['training_source_sha256']),'library_sources_checked':len(base['library_source_sha256']),
            'physical_steps':0,'learning_updates':0,'hardware_io':False,'dynamic_execution_authorized':False,
            'automatic_retry':False}
    index.write_bytes(encode(result));print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
