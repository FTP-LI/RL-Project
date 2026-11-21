"""Explicit zero-step repair of the collision diagnosis, no budget extension.

The parent startup checked prone, but the one-stratum collector correctly
selected supine. Its exact reset guard stopped at0physics steps. Preserve that
attempt. Set the startup config's reset pose to the declared first stratum.
Reuse the frozen parent implementation without changing its bytes; new paths,
wrapper and parent hashes are recorded in the new protocol. All original
32-per-arm/11s/seed1010/budget/criteria/mask/IMU settings remain unchanged.
Validation: CPU config diff must be solely the reset pose, then both actual
zero-step snapshots must match each other and their later rollout resets.
"""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import sys
import traceback
import getup_contact_isolation as base

WRAPPER=Path(__file__).resolve()
PARENT_IMPLEMENTATION=Path(base.__file__).resolve()
PARENT_PLAN=base.PLAN
PARENT_OUT=base.OUT
old_config=base.config
old_read=base.read

def config(p):
    cfg=old_config(p)
    cfg.events['set_ground_state'].params['pose']=p['poses'][0]
    return cfg

def read(p):
    value=old_read(p)
    if Path(p)==base.ROOT/'configs/experiments/common_plant/getup.json':
        value=deepcopy(value)
        value['source_sha256'][str(PARENT_IMPLEMENTATION.relative_to(base.ROOT))]=base.sha(PARENT_IMPLEMENTATION)
        state=old_read(PARENT_OUT/'progress.json');assert state['status']=='stopped-review-required'
        attempt=old_read(PARENT_OUT/'full/progress.json');folder=Path(attempt['path'])
        runtime=old_read(folder/'runtime-contract.json');assert runtime['control_steps']==0 and not attempt['conditions']
        value['startup_recovery']=dict(parent_protocol=str(PARENT_PLAN.relative_to(base.ROOT)),parent_protocol_sha256=base.sha(PARENT_PLAN),
            parent_implementation_sha256=base.sha(PARENT_IMPLEMENTATION),original_runtime=str((folder/'runtime-contract.json').relative_to(base.ROOT)),
            original_physics_steps=0,change='Only initialize set_ground_state.pose to the declared supine stratum before startup reset',
            unchanged='All policy, masks, CAD exclusions, timing, IMU, source criteria, seeds and total35200control budget',automatic_extension=False)
    return value

base.config=config
base.read=read
base.PLAN=base.ROOT/'configs/experiments/getup_contact_release_recovery.json'
base.OUT=base.ROOT/'records/reports/getup-contact-release-recovery'
# The parent coordinator launches its own __file__. Point that entry to this
# explicit wrapper so each fresh child receives exactly this declared repair.
base.__file__=str(WRAPPER)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--startup',choices=base.ARMS);parser.add_argument('--evaluate',choices=base.ARMS);parser.add_argument('--analyze',action='store_true');a=parser.parse_args()
    logs=base.ROOT/'records/reports/common-plant';path=logs/f'getup-release-recovery-{datetime.now().strftime("%Y%m%dT%H%M%S")}-{os.getpid()}.log'
    event=dict(argv=sys.argv,started_at=datetime.now(timezone.utc).isoformat(),foreground_tty=os.isatty(1),log=str(path.relative_to(base.ROOT)))
    with path.open('x',buffering=1) as file:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=base.collector.Tee(stdout,file),base.collector.Tee(stderr,file)
        try:
            print('VISIBLE_RELEASE_RECOVERY_COMMAND',json.dumps(event),flush=True)
            if a.startup:base.stage(a.startup,True)
            elif a.evaluate:base.stage(a.evaluate,False)
            elif a.analyze:base.analyze()
            else:base.run()
            event['exit_code']=0
        except BaseException as e:event.update(exit_code=1,error=repr(e));traceback.print_exc();raise
        finally:
            event['ended_at']=datetime.now(timezone.utc).isoformat()
            with (logs/'getup-release-recovery-commands.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            sys.stdout,sys.stderr=stdout,stderr
