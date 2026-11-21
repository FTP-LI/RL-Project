"""One foreground PTY, bounded learning then fixed evaluation, no retries."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import subprocess
import sys
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write

OUT=ROOT/'records/reports/planar-filter'


def main():
    assert os.isatty(1),'Use the existing visible shared VS Code terminal'
    path=ROOT/'configs/experiments/planar_filter_pair.json';pair=read(path)
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(path)
    for name,digest in pair['source_sha256'].items():assert sha(ROOT/name)==digest,name
    target=OUT/'progress.json';assert not target.exists(),'Existing attempt retained; review before retry'
    state=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(path),models={},commands=[])
    def save():write(target,state)
    def call(script,*args):
        command=[sys.executable,'-B',str(ROOT/script),*map(str,args)]
        item=dict(command=command,started_at=datetime.now(timezone.utc).isoformat());state['commands'].append(item);save()
        print('PLANAR_FILTER_COMMAND:',json.dumps(command),flush=True)
        code=subprocess.call(command,cwd=WORK)
        item.update(exit_code=code,ended_at=datetime.now(timezone.utc).isoformat());save()
        assert code==0,'Child failed; no automatic restart or next arm'
    save()
    try:
        for arm in ('control','candidate'):
            state['active_arm']=arm;save()
            call('scripts/training/planar_filter_training.py',arm)
            call('scripts/validation/verify_planar_filter_training.py',arm)
            checked=read(OUT/arm/'training-verification.json');state['models'][arm]=checked;save()
            screen=deepcopy(pair['evaluation_template'])
            screen.update(model=str(Path(checked['model']).relative_to(ROOT)),model_sha256=checked['model_sha256'],
                scope='Frozen three-command screen of planar velocity statistic pair; legacy criteria preserved')
            write(OUT/arm/'screen-protocol.json',screen)
            call('scripts/validation/evaluate_planar_filter.py',arm)
            state['models'][arm]['evaluation']=read(OUT/arm/'evaluation-progress.json')['evaluation'];save()
        call('scripts/validation/analyze_planar_filter_pair.py')
        state.update(status='completed',decision=read(OUT/'analysis.json')['decision'],learning_updates=100,
            ended_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:save()
    print('PLANAR_FILTER_PAIR_COMPLETE:',state['decision'],'No automatic extension.',flush=True)


if __name__=='__main__':main()
