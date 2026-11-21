"""Verify the completed live probes and reconcile final checkpoint indexes.

The final models were already audited by the foreground controller. Recheck
their bytes and saved labels on CPU, retain the old index, and publish the
actual latest complete model. No environment, optimizer step or inference is
created. This closes the bookkeeping defect without repeating training.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sys
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'records/reports/s1-integration'
sys.path.insert(0,str(ROOT/'scripts/training'))
from training_watch import read,write
from guarded_training import sha


def main():
    torch.set_num_threads(1)
    run=read(OUT/'progress.json')
    assert run['status']=='passed' and run['actual_updates']==113
    offline=read(ROOT/'records/reports/training-workflow/verification.json')
    assert offline['status']=='passed'
    for path,value in offline['source_hashes'].items():assert sha(ROOT/path)==value
    rows=[];cfgs={};protocols={}
    for name,entry in run['probes'].items():
        assert entry['status']=='passed'
        record=Path(entry['record']);checked=read(record/'engineering-verification.json')
        assert checked==entry['verification']
        for file,value in read(record/'source-hashes.json').items():assert sha(record/'sources'/file)==value
        checkpoint=Path(checked['checkpoint'])
        assert sha(checkpoint)==checked['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert int(saved['iter'])==checked['saved_iteration']
        assert int(saved['infos']['env_state']['common_step_counter'])==checked['common_step']
        old=record/'last-checkpoint.json';backup=record/'last-checkpoint-before-index-repair.json'
        if not backup.exists():shutil.copy2(old,backup)
        actual=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            iteration=checked['saved_iteration'],common_step_counter=checked['common_step'],
            complete_update=True,index_source='CPU reconciliation of the actually saved and audited final model')
        write(old,actual);assert read(old)==actual
        rows.append(dict(probe=name,record=str(record.relative_to(ROOT)),
            actual_updates=checked['actual_updates'],final_iteration=checked['saved_iteration'],
            final_checkpoint_sha256=checked['checkpoint_sha256'],latest_index_verified=True))
        cfg={part:yaml.load((checkpoint.parent/'params'/f'{part}.yaml').read_text(),Loader=yaml.BaseLoader)
             for part in ('env','agent')}
        cfg['agent'].pop('run_name');cfgs[name]=cfg
        protocols[name]=read(record/'protocol.json')
    assert cfgs['observer-control']==cfgs['observer-video']
    a,b=protocols['observer-control'].copy(),protocols['observer-video'].copy()
    for plan in (a,b):
        for key in ('run_label','visuals_enabled','reason'):plan.pop(key)
    assert a==b
    assert protocols['observer-resume']['parent_sha256']==run['probes']['observer-stop']['verification']['checkpoint_sha256']
    native=read(OUT/'native-video-check.json');assert native['status']=='passed'
    assert sha(ROOT/native['screenshot'])==native['screenshot_sha256']
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        actual_live_updates=113,zero_update_failed_start_retained=True,
        visible_stop_and_resume_verified=True,same_parent_overhead_pair_verified=True,
        per_model=rows,measurement=run['measurement'],native_video_playback_verified=True,
        browser_dashboard_verified=False,browser_reason='Localhost origin access was declined; direct local MP4 playback used',
        index_repair_scope='Real live final models unchanged; generic atomic-save index fix checked through installed runner save on CPU',
        source_hashes={'verification_script':sha(Path(__file__)),
            'current_guarded_training':sha(ROOT/'scripts/training/guarded_training.py'),
            'current_training_watch':sha(ROOT/'scripts/training/training_watch.py'),
            'live_controller_record':sha(OUT/'progress.json'),
            'offline_checks':sha(ROOT/'records/reports/training-workflow/verification.json')},
        gait_quality_verified=False,hardware_started=False)
    write(OUT/'verification.json',result)
    gate=dict(status='S1-evidence-and-engineering-complete',bounded_learning_allowed=True,
        engineering_integration_allowed=False,allowed_scope='Frozen single-factor S2 walking repair pair from the chosen existing parent; no hardware',
        source_evidence={'inputs':sha(ROOT/'records/reports/s1-inputs/progress.json'),
            'seven_commands':sha(ROOT/'records/reports/s1-evaluation/analysis.json'),
            'parent_choice':sha(ROOT/'records/reports/s1-evaluation/next-repair-decision.json'),
            'integration':sha(OUT/'verification.json')},automatic_next_run=False)
    write(OUT/'completion-gate.json',gate)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','per_model')},indent=2))


if __name__=='__main__':main()
