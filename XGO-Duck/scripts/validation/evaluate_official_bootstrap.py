"""Three-command screening of the bounded official-actor initialization.

Reuse the archived evaluator unchanged. Only select stop/forward/backward in
a separately frozen protocol. This screen cannot certify seven-command gait.
All simulation is run as a foreground child of the visible VS Code task.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
import sys

from s1_reference_inputs import ROOT,read,sha

PLAN=ROOT/'configs/experiments/official_bootstrap_screen.json'
OUT=ROOT/'records/reports/official-actor-bootstrap'


def main():
    assert os.isatty(1),'Use the single visible VS Code terminal'
    plan=read(PLAN);assert plan['status']=='frozen-before-simulation'
    checked=read(OUT/'training-verification.json');assert checked['status']=='passed'
    assert checked['model_sha256']==plan['model_sha256']
    assert sha(ROOT/plan['model'])==plan['model_sha256']
    for path,digest in plan['source_sha256'].items():assert sha(ROOT/path)==digest,path
    target=OUT/'evaluation-progress.json'
    assert not target.exists(),'Existing attempt retained; review before another run'
    record=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(PLAN),scope='Three-command development screen; no learning')
    def save():target.write_text(json.dumps(record,indent=2)+'\n')
    save()
    before=set((ROOT/'records/evaluations').iterdir())
    import evaluate_fixed_policy as evaluator
    evaluator.PROTOCOL=PLAN
    sys.argv=[str(Path(evaluator.__file__)),'--onnx',str(ROOT/plan['model'])]
    try:
        try:
            evaluator.main()
        except SystemExit as exit_status:
            if exit_status.code not in (None,0):raise
        created=[x for x in set((ROOT/'records/evaluations').iterdir())-before
                 if x.is_dir() and (x/'evaluation.json').exists()]
        assert len(created)==1
        folder=created[0];record['evaluation']=str(folder)
        shutil.copy2(Path(__file__),folder/Path(__file__).name)
        import verify_fixed_evaluation
        sys.argv=[str(Path(verify_fixed_evaluation.__file__)),str(folder)]
        verify_fixed_evaluation.main()
        record.update(status='collected-and-verified',episodes=96,learning_updates=0,
                      ended_at=datetime.now(timezone.utc).isoformat(),seven_command_acceptance=False)
    except BaseException as error:
        record.update(status='failed',error=repr(error));raise
    finally:save()
    print('BOOTSTRAP_SCREEN_COMPLETE: review results before any further learning.',flush=True)


if __name__=='__main__':main()
