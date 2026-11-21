"""Exercise transient atomic-publish faults on CPU; audit saved recovery state."""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import patch
import torch
from s1_reference_inputs import ROOT,read,sha
from s1_observer_integration import tensors
sys.path.insert(0,str(ROOT/'scripts/training'))
import record_io

OUT=ROOT/'records/reports/official-actor-bootstrap'
RECORD=ROOT/'records/experiments/duck_walk_official-adaptation_candidate_s42_20261102T183003'


def main():
    torch.set_num_threads(1)
    # Inject the observed failure at the OS replacement boundary, then verify
    # exact old/new JSON visibility. No GPU or physical simulation is involved.
    with tempfile.TemporaryDirectory(prefix='record-io-',dir=OUT) as directory:
        target=Path(directory)/'state.json';target.write_text('{"old":true}')
        replace=Path.replace;attempts=[]
        def transient(path,destination):
            attempts.append(1)
            if len(attempts)<4:
                assert json.loads(target.read_text())=={'old':True}
                raise PermissionError(13,'simulated temporary sharing violation')
            return replace(path,destination)
        with patch.object(Path,'replace',transient):record_io.write(target,dict(new=True))
        assert len(attempts)==4 and json.loads(target.read_text())=={'new':True}
        with patch.object(Path,'replace',side_effect=PermissionError(13,'persistent fault')):
            try:record_io.write(target,dict(rejected=True))
            except PermissionError:pass
            else:raise AssertionError('Persistent errors must stop')
        assert json.loads(target.read_text())=={'new':True}
        with patch.object(Path,'replace',side_effect=OSError(28,'disk full')):
            try:record_io.write(target,dict(rejected=True))
            except OSError as error:assert error.errno==28
            else:raise AssertionError('Other I/O errors must propagate')
    state=read(RECORD/'state.json');last=read(RECORD/'last-checkpoint.json')
    saved=torch.load(last['checkpoint'],map_location='cpu',weights_only=True)
    assert state['status']=='failed' and state['completed_updates_this_run']==99
    assert saved['iter']==125 and saved['infos']['env_state']['common_step_counter']==3024
    assert all(bool(torch.isfinite(t).all()) for t in tensors(saved))
    assert int(saved['actor_state_dict']['obs_normalizer.count'])==0
    assert int(saved['critic_state_dict']['obs_normalizer.count'])==126*24*4096
    assert {int(v['step']) for v in saved['optimizer_state_dict']['state'].values()}=={2320,2520}
    assert saved['infos']['bootstrap']['actor_normalizer_frozen']
    result=dict(status='passed',reviewed_continuation_allowed=True,
        reason='Transient mounted-drive record publication failed after optimizer update; gait tripwire not triggered',
        record=str(RECORD),checkpoint=last['checkpoint'],checkpoint_sha256=sha(Path(last['checkpoint'])),
        attempted_completed_updates=99,durable_additional_updates=76,completed_but_unsaved_updates=23,
        original_additional_budget=150,remaining_compute_budget=51,final_durable_total_if_complete=177,
        actual_total_including_bootstrap_if_complete=200,exact_trajectory_resume=False,
        transient_fault_retry_checked=True,persistent_failure_propagation_checked=True,
        previous_complete_json_preserved=True,record_writer_sha256=sha(ROOT/'scripts/training/record_io.py'),
        scope='Recovery review and failure-injection test; no new learning or extra budget',
        script_sha256=sha(Path(__file__)))
    (OUT/'io-recovery-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
