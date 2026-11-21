"""Sequential bounded learning and screening in one visible VS Code PTY.

Children inherit this terminal. Any failed child stops the sequence; there is
no retry, budget extension or background worker. Existing attempts are kept.
"""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import subprocess
import sys
from s1_reference_inputs import ROOT, WORK, read, sha
from record_io import write

OUT=ROOT/'records/reports/tracking-precision'


def main():
    assert os.isatty(1),'Use the single visible VS Code terminal'
    path=ROOT/'configs/experiments/tracking_precision_pair.json';pair=read(path)
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(path)
    for name,digest in pair['source_sha256'].items():assert sha(ROOT/name)==digest,name
    target=OUT/'progress.json';assert not target.exists(),'Existing attempt retained; review before retry'
    state=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(path),models={},commands=[])
    def save():write(target,state)
    def call(script,*args):
        command=[sys.executable,'-B',str(ROOT/script),*map(str,args)]
        item=dict(command=command,started_at=datetime.now(timezone.utc).isoformat());state['commands'].append(item);save()
        print('PRECISION_COMMAND:',json.dumps(command),flush=True)
        code=subprocess.call(command,cwd=WORK)
        item.update(exit_code=code,ended_at=datetime.now(timezone.utc).isoformat());save()
        assert code==0,'Child failed; no automatic restart or next arm'
    save()
    try:
        for arm in ('control','candidate'):
            state['active_arm']=arm;save()
            call('scripts/training/tracking_precision.py',arm)
            call('scripts/validation/verify_tracking_precision.py',arm)
            checked=read(OUT/arm/'training-verification.json')
            state['models'][arm]=checked;save()
            # Conditions are frozen before training. Only fill the verified LAST
            # model hash now; do not select a favorable intermediate checkpoint.
            screen=deepcopy(pair['evaluation_template'])
            screen.update(model=str(Path(checked['model']).relative_to(ROOT)),model_sha256=checked['model_sha256'],
                scope='Three-command fixed development screen of declared velocity-width pair')
            write(OUT/arm/'screen-protocol.json',screen)
            call('scripts/validation/evaluate_tracking_precision.py',arm)
            state['models'][arm]['evaluation']=read(OUT/arm/'evaluation-progress.json')['evaluation'];save()
        call('scripts/validation/analyze_tracking_precision.py')
        state.update(status='completed',decision=read(OUT/'analysis.json')['decision'],learning_updates=100,
                     ended_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:save()
    print('TRACKING_PRECISION_PAIR_COMPLETE:',state['decision'],'No automatic extension.',flush=True)


if __name__=='__main__':main()
