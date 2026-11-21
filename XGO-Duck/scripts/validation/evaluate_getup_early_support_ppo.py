"""Compare the parent and the paired PPO exports on a new shared release."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

import getup_shared_release as shared
import getup_supine_evaluation as evaluation
import training_live_page as live
import verify_getup_supine_evaluation as verifier
from check_getup_early_support_clock import TZ
from getup_anchor_same_release import model_metadata, visible_log
from getup_shared_release_terrain import cpu_scene_readback
from record_io import write
from run_getup_early_support_ppo import relative, zero
from run_local_getup_ppo import MemoryWatch, check_project
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_early_support_ppo_evaluation.json'
BASE = ROOT/'configs/experiments/local_getup_ppo_evaluation.json'
OUT = ROOT/'records/reports/getup-early-support-ppo-functional'
PAGE = ROOT/'records/getup-early-support-evaluation-live.html'
STAGE = 'S3-early-support-ppo-functional-development'
ROLES = ('fixed-weighted','control','candidate')


def freeze(training_protocol,evidence):
    assert not PLAN.exists() and not OUT.exists()
    training_protocol,evidence = (ROOT/path for path in (training_protocol,evidence))
    assert training_protocol.is_relative_to(ROOT/'configs/experiments')
    assert evidence.is_relative_to(ROOT/'records/reports')
    proof = read(evidence)
    assert proof['status']=='passed-online-training-integrity' and proof['protocol_sha256']==sha(training_protocol)
    spec = deepcopy(read(BASE))
    p = spec['environment']
    seed = 1084
    assert not any(f'_s{seed}_' in f.name for f in (ROOT/'records/evaluations').iterdir())
    codes = np.concatenate(([1],np.random.default_rng(seed).permutation(np.repeat(np.arange(4),[32,31,128,128]))))
    assert len(codes)==p['num_envs']==320
    groups = {pose:np.flatnonzero(codes==i).tolist() for i,pose in enumerate(shared.POSES)}
    stamp = datetime.now(TZ).strftime('%Y%m%dT%H%M%S')
    p.update(seed=seed,stage=STAGE,split=dict(pose_codes=codes.tolist(),groups=groups),
             run_id=f'duck_getup_early-support-ppo_paired_s{seed}_{stamp}')
    fixed = p['model']
    assert sha(ROOT/fixed)==p['model_sha256']==read(training_protocol)['environment']['model_sha256']
    policies = {ROLES[0]:dict(model=fixed,model_sha256=p['model_sha256'],seed=seed)}
    files = [evidence,training_protocol,BASE,zero.SCENE,Path(__file__).resolve()]
    for role in ROLES[1:]:
        arm = proof['arms'][role]
        model = ROOT/arm['model']
        assert arm['status']=='passed' and sha(model)==arm['model_sha256']
        assert model_metadata(p,arm)==spec['model_interface']
        policies[role] = dict(model=arm['model'],model_sha256=arm['model_sha256'],seed=seed)
        files.append(model)
    prior_gain = ROOT/'configs/experiments/getup_supine_weighted_learning.json'
    minimum_gain = read(prior_gain)['actual_supine_minimum_gain']
    files.append(prior_gain)
    spec['references'].update({relative(path):sha(path) for path in files})
    spec.update(stage=STAGE,branches=list(ROLES),policies=policies,training_evidence=relative(evidence),
        cameras={role:{pose:groups[pose][0] for pose in shared.POSES} for role in ROLES},
        choice='Original timely and final all-pose gates and saved supine improvement criteria.',
        actual_supine_minimum_gain=minimum_gain,parent_protocol_sha256=sha(training_protocol),observation=relative(PAGE))
    shared.assert_sources(spec)
    cpu = cpu_scene_readback(p,spec)
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='passed-cpu-zero-step',scene=cpu,protocol_sha256=sha(PLAN),seed=seed,
          physical_substeps=0,heldout_used=False))
    print('EARLY_SUPPORT_PPO_EVAL_FROZEN',sha(PLAN),p['run_id'],flush=True)


def checked():
    check_project()
    spec = read(PLAN)
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(PLAN)
    shared.assert_sources(spec)
    return spec,ROOT/'records/evaluations'/spec['environment']['run_id']


def run(spec,record):
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage']==STAGE and stage['current_protocol_frozen'] and ROOT/stage['current_protocol']==PLAN
    p = spec['environment']
    began = perf_counter()
    updates = []
    watch = MemoryWatch(.85)

    def publish(path,state):
        write(path,state)
        if path!=record/'evaluation.json':
            return
        description = record/'README.md'
        if not description.exists():
            description.write_text(
                '# 起身早期支撑独立评估\n\n三组共用一次物理释放，按原四姿态门槛评分。\n\n'
                '[冻结方案](../../../configs/experiments/getup_early_support_ppo_evaluation.json) · '
                '[状态](evaluation.json) · [日志](execution.log)\n',encoding='utf-8')
            log = stack.enter_context((record/'execution.log').open('x',buffering=1))
            stack.enter_context(visible_log(log))
        actual = state['actual']['controls']
        if actual and (not updates or updates[-1]['control']!=actual):
            updates.append(dict(control=actual,transitions_per_second=actual*p['num_envs']/(perf_counter()-began)))
        first = state['conditions'][0]['video']['path'] if state['conditions'] else None
        status = state['status']
        frame = next(iter(sorted(record.rglob('frame-step-0.jpg'))),None)
        live.publish(record,dict(status=status,run_id=p['run_id'],purpose=f"起身对照 · {state['phase']}",
            updated_at=datetime.now(TZ).isoformat(),controls=actual,maximum_controls=spec['maximum_unique_controls'],
            transitions=actual*p['num_envs'],learning_updates=0,updates=updates,latest_clip=first,frame=frame.relative_to(record).as_posix() if frame else None))

    with ExitStack() as stack, watch.active(), patch.object(evaluation,'write',publish), patch.object(live,'PAGE',PAGE):
        try:
            evaluation.run(spec,record,zero.SCENE,watch.check)
            watch.check()
        finally:
            if record.is_dir():
                write(record/'gpu-telemetry.json',dict(samples=watch.rows,failure=repr(watch.error) if watch.error else None))


def verify(spec,record):
    state = read(record/'evaluation.json')
    assert state['status']=='collected-review-required'
    result = verifier.verify(spec,record,state,minimum_gain=spec['actual_supine_minimum_gain'])
    write(OUT/'analysis.json',dict(result,protocol_sha256=sha(PLAN)))
    view = read(record/'state.json')
    view.update(status='completed',purpose='起身对照 · 独立评分已完成',updated_at=datetime.now(TZ).isoformat())
    with patch.object(live,'PAGE',PAGE):
        live.publish(record,view)
    print('EARLY_SUPPORT_PPO_EVAL_VERIFIED',{role:arm['combined_successes'] for role,arm in result['arms'].items()},
          result['selection'],flush=True)


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--freeze':
        return freeze(Path(sys.argv[2]),Path(sys.argv[3]))
    spec,record = checked()
    if sys.argv[1:]==['--verify']:
        return verify(spec,record)
    assert not sys.argv[1:] and not record.exists()
    run(spec,record)


if __name__=='__main__':
    main()
