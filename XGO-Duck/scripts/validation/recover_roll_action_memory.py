"""Repair a zero-step missing MODEL import while retaining frozen originals.

The first copied collector failed before configuration/environment creation.
Supply only its existing official-model constant in this explicit adapter.
Use new protocols/records with unchanged config, algorithm and remaining
32+32 episode budget. Never delete old progress to force a retry.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import numpy as np
import yaml
import collect_roll_action_memory as collector
import evaluate_roll_reference as original
import roll_action_memory_pair as pair
import verify_roll_action_memory as verifier
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write

PLAN=ROOT/'configs/experiments/roll_action_memory_startup_recovery.json'
OUT=ROOT/'records/reports/roll-action-memory/startup-recovery'


def supply_constant():
    assert not hasattr(collector,'MODEL'),'Frozen original unexpectedly changed'
    assert collector.config is original.config
    collector.MODEL=original.MODEL


def prepare_recovery():
    import torch
    assert not PLAN.exists() and not OUT.exists()
    failure=read(pair.OUT/'pair-progress.json');first=read(pair.OUT/'control/progress.json')
    assert failure['completed_arms']==[] and failure['error']=="NameError(\"name 'MODEL' is not defined\")"
    folder=Path(first['path']);assert first['status']=='failed'
    assert not (folder/'env.yaml').exists() and not list(folder.glob('*trace.npz')) and not list((folder/'videos').glob('*.mp4'))
    base=read(pair.PLAN);pre=read(pair.OUT/'preflight.json');plans={}
    for arm in ('control','candidate'):
        p=deepcopy(read(ROOT/base['protocols'][arm]))
        for file,digest in p['source_sha256'].items():assert sha(ROOT/file)==digest,file
        for file,digest in p['training_source_sha256'].items():assert sha(WORK/file)==digest,file
        p['source_sha256']['scripts/validation/recover_roll_action_memory.py']=sha(Path(__file__))
        p['startup_repair']='Explicitly supply frozen collector.MODEL from unchanged evaluate_roll_reference.MODEL'
        assert sha(original.MODEL)==p['model_sha256']
        actual=yaml.load(yaml.dump(asdict(collector.config(p))),Loader=yaml.BaseLoader);assert actual==pre['config']
        plans[arm]=p
    # CPU lookup and compilation of the function confirm its only missing
    # global constant; actual GPU behavior still requires the visible task.
    assert 'MODEL' in collector.evaluate.__code__.co_names
    supply_constant();assert collector.MODEL==original.MODEL and not torch.cuda.is_initialized()
    proto_dir=ROOT/'configs/experiments/roll_action_memory_startup_recovery';proto_dir.mkdir(exist_ok=True)
    OUT.mkdir();paths={}
    for arm,p in plans.items():
        file=proto_dir/f'{arm}.json';write(file,p);paths[arm]=str(file.relative_to(ROOT))
    master=deepcopy(base);master.update(protocols=paths,arm_protocol_sha256={k:sha(ROOT/v) for k,v in paths.items()},
        original_failed_protocol=str(pair.PLAN.relative_to(ROOT)),original_failed_protocol_sha256=sha(pair.PLAN),
        original_failed_progress_sha256=sha(pair.OUT/'pair-progress.json'),
        startup_repair='MODEL import only; no physical config or action computation change',
        consumed_control_steps_before_repair=0,remaining_control_steps=900)
    write(PLAN,master);new_pre=deepcopy(pre);new_pre.update(protocol_sha256=sha(PLAN),arm_protocol_sha256=master['arm_protocol_sha256'])
    write(OUT/'preflight.json',new_pre)
    record=dict(status='prepared-cpu-checked-not-yet-run',original_error=failure['error'],
        original_failed_run=first['run_id'],no_environment_yaml=True,no_trace=True,no_video=True,
        original_failed_report_sha256=sha(folder/'evaluation.json'),original_failed_progress_sha256=sha(pair.OUT/'pair-progress.json'),
        repair='Import missing original MODEL constant in explicit adapter; frozen source retained',
        physical_config_matches_preflight=True,algorithm_unchanged=True,maximum_remaining_transitions=28800,
        learning_updates=0,script_sha256=sha(Path(__file__)),recovery_protocol_sha256=sha(PLAN))
    write(OUT/'startup-repair.json',record);print(json.dumps(record,indent=2))


def run():
    supply_constant();pair.PLAN=PLAN;pair.OUT=OUT;verifier.OUT=OUT
    pair.run()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
    prepare_recovery() if args.prepare else run()
