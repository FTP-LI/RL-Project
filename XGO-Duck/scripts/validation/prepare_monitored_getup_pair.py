"""Prepare or finalize the bounded monitor pair; never launch a worker.

The freeze step reuses the established saved-zero verifier with new paths via
an isolated module global. Historical scripts, protocols and receipts retain
their bytes; the new phase is recorded explicitly below.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/training"))
import observed_getup_entry as entry

BASE = "records/reports/getup-monitored-authorization-20261116"
DRAFT = "configs/experiments/observed_getup_monitor_draft_20261116.json"
ZERO = "configs/experiments/observed_getup_monitor_zero_20261116.json"
FINAL = "configs/experiments/observed_getup_monitor_pair_20261116.json"


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def stage(project, spec, path, execution):
    g,t=project['goal_execution'],project['staged_training_request']
    assert g['status']=='active' and t['pause_requested'] is False
    assert t['learning_running'] is False and t['simulation_running'] is False
    project['current_phase']=g['current_phase']=t['current_stage']=spec['stage']
    g['current_protocol']=t['current_protocol']=path
    project['current_protocol_execution']=g['current_protocol_execution']=t['current_protocol_execution']=execution
    t['current_protocol_frozen']=True
    project['next_gate']=t['next_gate']='Run only the new bounded pair after actual viewing and CUDA start verification; preserve closed runs and independently review performance and artifacts.'


def prepare():
    assert not (ROOT/DRAFT).exists() and not (ROOT/ZERO).exists() and not (ROOT/BASE).exists()
    checked=entry.read(ROOT/'records/reports/getup-runtime-review-20261116/monitor-check.json')
    assert checked['status']=='passed-cpu-monitor-contract'
    plan=entry.read(ROOT/'configs/experiments/observed_getup_path_pair_20261116.json')
    assert all(entry.sha(ROOT/n)==d for n,d in plan['source_sha256'].items())
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S')
    plan.update(status='draft-awaiting-monitored-cuda-zero',stage='observed-getup-monitored-authorization',seed=3266,
                purpose='同预算训练检查调度对照；记录原始指标、完整动作和分段耗时',coordinator=BASE+'/training',zero_step_record=BASE+'/cuda-zero')
    plan.pop('zero_result_sha256',None)
    plan['factor']={'control':'canonical-synchronous','throughput':'bounded-authorization-monitor',
                    'changed':'After matched initialization, move full worker authorization to a 0.1-second monitor; reject age over 0.5 s; outer supervision remains synchronous.'}
    plan['authorization_monitor']={'interval_seconds':0.1,'max_age_seconds':0.5,'scope':'worker-after-matched-initialization'}
    for role,label in [('control','auth-sync'),('throughput','auth-monitor')]:
        plan['arms'][role]['run_id']=f'duck_getup_{label}_e3072_s3266_{stamp}'
    source=plan['source_sha256']
    for name in ('scripts/training/monitored_ppo_authorization.py','scripts/training/observed_getup_monitor_entry.py',
                 'scripts/validation/check_monitored_ppo_authorization.py','scripts/validation/review_getup_start_and_costs.py',
                 'records/reports/getup-runtime-review-20261116/monitor-check.json'):
        source[name]=entry.sha(ROOT/name)
    plan['supervisor_workers']={a['run_id']:[sys.executable,'-B',str(ROOT/'scripts/training/observed_getup_monitor_entry.py'),
                                '--mode','worker','--protocol',str(ROOT/FINAL),'--arm',role] for role,a in plan['arms'].items()}
    write(ROOT/DRAFT,plan)
    zero=entry.read(ROOT/'configs/experiments/observed_getup_path_zero_20261116.json')
    zero.update(stage='observed-getup-monitored-zero',training_draft=DRAFT,training_draft_sha256=entry.sha(ROOT/DRAFT),
                coordinator=BASE+'/zero-observation',source_sha256=deepcopy(source),
                training_settings_sha256={role:entry.settings_digest(plan,role) for role in plan['arms']},
                zero_run_ids={role:a['run_id']+'_zero' for role,a in plan['arms'].items()})
    zero['supervisor_workers']={run_id:[sys.executable,'-B',str(ROOT/'scripts/training/observed_getup_zero.py'),
                               '--mode','worker','--protocol',str(ROOT/ZERO),'--arm',role] for role,run_id in zero['zero_run_ids'].items()}
    write(ROOT/ZERO,zero)
    (ROOT/BASE).mkdir()
    p=entry.read(ROOT/'project.json')
    stage(p,zero,ZERO,'zero-step-admitted-awaiting-view')
    p['goal_execution']['monitored_authorization_pair']={'status':'zero-step-preparation','scope':plan['factor']['changed'],
        'acceptance':'Both groups complete the same fixed budget with matched saved starts, complete clips, verified checkpoints and no monitor faults; independently compute total time and retain numerical differences.',
        'draft':DRAFT,'zero_protocol':ZERO,'report':BASE+'/README.md','run_ids':{k:v['run_id'] for k,v in plan['arms'].items()},
        'model_promotion_allowed':False,'hardware_io':False}
    write(ROOT/'project.json',p)
    print(json.dumps({'status':'prepared-new-zero-only','sources':len(source),'run_ids':p['goal_execution']['monitored_authorization_pair']['run_ids']}))


def freeze():
    # Keep the original source and its historical output intact. Execute its
    # reviewed readback routine from a new in-memory module with explicit paths.
    from types import ModuleType
    original=ROOT/'scripts/validation/freeze_canonical_getup_pair.py'
    text=original.read_text(encoding='utf-8')
    replacements={'observed_getup_path_draft_20261116':'observed_getup_monitor_draft_20261116',
                  'observed_getup_path_pair_20261116':'observed_getup_monitor_pair_20261116',
                  'observed_getup_path_zero_20261116':'observed_getup_monitor_zero_20261116',
                  'getup-canonical-path-20261116':'getup-monitored-authorization-20261116',
                  'canonical_authorization_pair':'monitored_authorization_pair',
                  'Run original then canonical checks':'Run synchronous then monitored checks'}
    for old,new in replacements.items():
        assert old in text
        text=text.replace(old,new)
    module=ModuleType('new_monitored_zero_readback'); module.__file__=str(Path(__file__).resolve())
    exec(compile(text,str(original),'exec'),module.__dict__)
    module.main()
    p=entry.read(ROOT/'project.json'); stage(p,entry.read(ROOT/FINAL),FINAL,'matched-zero-verified-awaiting-training-view')
    write(ROOT/'project.json',p)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','freeze'])
    {'prepare':prepare,'freeze':freeze}[parser.parse_args().action]()
