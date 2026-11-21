"""Replay saved trajectories and exercise save/stop contracts, without learning.

No environment construction, policy inference, optimizer step or CUDA. Runner
checkpoint serialization uses the actual installed method with small CPU test
tensors. It does not establish live rendering, throughput or gait improvement.
"""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import hashlib
import inspect
import json
import sys
import tempfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from training_watch import (read, write, CONFIG, GaitWatch, signed_foot_gap_mm,
                            run_name, UpdateBoundary, ControlledStop, UnsafeState)
from guarded_training import guarded_class, require_protocol


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    cfg = read(CONFIG)
    checks=[]
    traces=[]
    for run, expected in (
            ('custom_onnx_reference_20261101T190142_s1001',True),
            ('custom_onnx_reference_20261101T191205_s1001',True),
            ('custom_onnx_reference_20261101T195044_s1001',False)):
        folder=ROOT/'records/evaluations'/run
        report=read(folder/'evaluation.json')
        names=report['model_metadata']['joint_names'].split(',')
        ids=[names.index(s+'_hip_roll') for s in ('left','right')]
        for condition in ('stop','forward','backward','turn_left','turn_right'):
            source=folder/(condition+'-trace.npz')
            with np.load(source,allow_pickle=False) as data:
                guard=GaitWatch(cfg,data['active'].shape[1])
                gap=signed_foot_gap_mm(data['foot_positions'],data['root_quaternion'])
                first_stop=None
                for i in range(len(gap)):
                    guard.record_falls(data['fell_over'][i].astype(bool)&data['active'][i].astype(bool))
                    if (i+1)%cfg['sample_stride_steps']:continue
                    age=np.where(data['active'][i],(i+1)*report['step_dt'],0)
                    guard.sample(hips_deg=np.rad2deg(data['joint_positions'][i][:,ids]),
                                 gap_mm=gap[i],age_s=age,dt=cfg['sample_stride_steps']*report['step_dt'])
                    if guard.stop_reason:
                        first_stop=(i+1)*report['step_dt'];break
                assert bool(first_stop)==expected,(run,condition,guard.stop_reason)
                traces.append(dict(run_id=run,condition=condition,expected_stop=expected,
                    stopped=bool(first_stop),stop_sim_seconds=first_stop,reason=guard.stop_reason,
                    windows=guard.windows,source=str(source.relative_to(ROOT)),sha256=sha(source)))
    checks.append('15 real saved evaluation trajectories: failure arms stop, official pose reference passes')
    # One bad window must not cause a stop; a good window clears its streak.
    benign=dict(hips_deg=np.zeros((32,2)),gap_mm=np.full(32,50.),age_s=np.full(32,2.),dt=2.)
    guard=GaitWatch(cfg,32)
    larger={**cfg,'minimum_valid_samples':32}
    guard=GaitWatch(larger,32)
    bad={**benign,'hips_deg':np.tile([-23.,23.],(32,1))}
    guard.sample(**bad);assert guard.stop_reason is None
    guard.sample(**benign);assert guard.streak['inward']==0
    guard.sample(**bad);guard.sample(**bad);assert guard.stop_reason=='sustained-inward'
    guard=GaitWatch(larger,32)
    guard.sample(**{**bad,'age_s':np.zeros(32)})
    assert guard.streak['inward']==0
    guard.sample(**benign);assert guard.stop_reason is None
    guard=GaitWatch(larger,32)
    for _ in range(2):guard.record_falls(np.ones(32,dtype=bool));guard.sample(**benign)
    assert guard.stop_reason=='sustained-falls'
    guard=GaitWatch(larger,32);guard.sample(**{**benign,'gap_mm':np.full(32,np.nan)})
    assert guard.stop_reason=='nonfinite-gait-state'
    checks.append('brief anomaly, reset settling, persistent falls and NaN behavior')
    repair=dict(inherited_faults=['inward','crossed'],review_after_updates=50)
    guard=GaitWatch(larger,32,repair=repair)
    for update in range(1,50):
        guard.sample(**bad);guard.complete_update(update)
        assert guard.stop_reason is None, 'Inherited defect stopped before declared diagnostic opportunity'
    guard.complete_update(50)
    assert guard.stop_reason=='repair-window-complete-review' and guard.review
    guard=GaitWatch(larger,32,repair=dict(inherited_faults=['crossed'],review_after_updates=50))
    guard.sample(**bad);guard.sample(**bad)
    assert guard.stop_reason=='sustained-inward', 'New posture fault was incorrectly deferred'
    guard=GaitWatch(larger,32,repair=repair)
    for _ in range(2):guard.record_falls(np.ones(32,dtype=bool));guard.sample(**bad)
    assert guard.stop_reason=='sustained-falls'
    guard=GaitWatch(larger,32,repair=repair)
    guard.sample(**{**benign,'gap_mm':np.full(32,np.nan)})
    assert guard.stop_reason=='nonfinite-gait-state'
    checks.append('repair window defers only declared inherited posture; new faults/falls/NaN still stop, review is mandatory by 50 updates')
    with tempfile.TemporaryDirectory(prefix='xgo-workflow-') as td:
        td=Path(td)
        calls=[]
        runner=SimpleNamespace(current_learning_iteration=17,logger=SimpleNamespace(log_dir=str(td)),
            env=SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=432)),
            save=lambda p:(calls.append(p),Path(p).write_text('test-save')))
        clock=[0.]
        b=UpdateBoundary(cfg,td,clock=lambda:clock[0])
        b.complete(runner,17);assert len(calls)==1
        runner.current_learning_iteration=18;clock[0]=121.
        b.complete(runner,18);assert len(calls)==2
        runner.current_learning_iteration=19
        try:b.complete(runner,19,'sustained-inward')
        except ControlledStop:pass
        else:raise AssertionError('Stop did not leave learn loop')
        assert len(calls)==3 and read(td/'stop.json')['complete_update']
        try:b.complete(runner,20,'bad-label')
        except ValueError:pass
        else:raise AssertionError('Accepted incomplete update')
        assert len(calls)==3
        checks.append('complete-update boundary saves first, then stops; wall-time save and label guard')
        from mjlab_microduck.tasks import MicroduckOnPolicyRunner
        from rsl_rl.runners import OnPolicyRunner
        cls=guarded_class(MicroduckOnPolicyRunner,{},cfg,{},td)
        obj=object.__new__(cls)
        tensor=torch.tensor([1.,2.])
        obj.alg=SimpleNamespace(save=lambda:dict(actor_state_dict={'x':tensor},
            optimizer_state_dict={'state':{0:{'step':torch.tensor(20)}}},
            normalizer={'count':torch.tensor(100)}))
        obj.env=SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=48))
        obj.cfg={'upload_model':False};obj.current_learning_iteration=1
        target=td/'complete.pt';obj.save(str(target))
        loaded=torch.load(target,weights_only=True)
        assert loaded['infos']['env_state']['common_step_counter']==48 and loaded['iter']==1
        assert loaded['normalizer']['count'].item()==100
        index=read(td/'last-checkpoint.json')
        assert index['iteration']==1 and index['checkpoint_sha256']==sha(target)
        # Exercise the same save method used by upstream at normal budget end,
        # without a logger boundary call. The index must advance with it.
        obj.current_learning_iteration=2;obj.env.unwrapped.common_step_counter=72
        obj.save(str(target))
        index=read(td/'last-checkpoint.json')
        assert index['iteration']==2 and index['common_step_counter']==72
        assert index['checkpoint']==str(target.resolve()) and index['checkpoint_sha256']==sha(target)
        original=sha(target);tensor[0]=float('nan')
        try:obj.save(str(target))
        except UnsafeState:pass
        else:raise AssertionError('Saved NaN')
        assert sha(target)==original and not list(td.glob('*.onnx'))
        assert read(td/'last-checkpoint.json')==index
        checks.append('real atomic save publishes the latest complete index even without a boundary hook; NaN preserves model and index')
        checks.append('installed runner CPU save preserves optimizer/normalizer/counter, rejects NaN, no ONNX')
        learned=inspect.getsource(OnPolicyRunner.learn)
        assert learned.index('self.alg.update()')<learned.index('self.current_learning_iteration = it')<learned.index('self.logger.log(')
        checks.append('installed runner calls logger hook after completed optimizer update and label assignment')
    name=run_name('walk','hip-margin','candidate',42,'20261102T010000')
    assert name=='duck_walk_hip-margin_candidate_s42_20261102T010000'
    try:run_name('walk','../bad','control',42)
    except ValueError:pass
    else:raise AssertionError('Invalid name allowed')
    try:require_protocol(read(ROOT/'configs/experiments/next_training.json'))
    except ValueError as e:assert 'not frozen' in str(e)
    else:raise AssertionError('Draft unexpectedly permits learning')
    checks.append('identifiable safe naming and draft protocol refuses training')
    assert not torch.cuda.is_initialized()
    out=ROOT/'records/reports/training-workflow';out.mkdir(exist_ok=True)
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),checks=checks,
        trace_replays=traces,new_training=False,new_simulation=False,policy_inference=False,
        live_renderer_tested=False,live_performance_tested=False,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in
            [Path(__file__),CONFIG,ROOT/'scripts/training/training_watch.py',ROOT/'scripts/training/guarded_training.py',
             ROOT/'scripts/training/training_dashboard.html',ROOT/'scripts/training/display_preferences.py']})
    write(out/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','trace_replays')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
