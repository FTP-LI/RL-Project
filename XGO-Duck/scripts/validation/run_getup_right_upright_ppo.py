"""Run the bounded right-side reward pair from the matched CUDA start."""
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import torch

import check_getup_right_upright_native_contract as native
import getup_ppo_live as live
import run_getup_early_support_ppo as paired
import run_getup_right_upright_ppo_zero as startup
import run_getup_right_upright_ppo_zero_probe as zero
import run_getup_supine_training as rollout
from check_getup_early_support_clock import TZ, native_contact
from getup_teacher_anchor_train import prepare, save_checkpoint
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_right_upright_ppo.json'
OUT = ROOT/'records/reports/getup-right-upright-ppo'
PAGE = ROOT/'records/getup-right-upright-ppo-live.html'
STAGE = 'S3-right-side-upright-ppo-development'
SCRIPT = Path(__file__).resolve()
ROLES = startup.ROLES


def freeze():
    check_project()
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage'] == zero.STAGE and ROOT/stage['current_protocol'] == zero.PLAN
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    spec = deepcopy(read(zero.PLAN))
    receipt = zero.OUT/'analysis.json'
    proof = read(receipt)
    assert proof['status'] == 'passed-actual-matched-cuda-zero' and not proof['mismatched']
    assert proof['protocol_sha256'] == sha(zero.PLAN) and not proof['heldout_used']
    p = spec['environment']
    n = p['num_envs']
    for role in ROLES:
        result = proof['runs'][role]['result']
        assert result['physical_substeps'] == result['actor_updates'] == result['critic_updates'] == 0
        assert result['environments'] == n == 3072
        state = zero.OUT/role/'matched-state.json'
        assert sha(state) == proof['common_state_signatures'][role]
        spec['sources'][startup.relative(state)] = sha(state)
    budget = spec['training_budget']
    horizon, updates = budget['steps_per_update'], budget['maximum_updates_per_arm']
    controls = horizon*updates
    assert controls*n == budget['maximum_transitions_per_arm'] == 1253376
    stamp = datetime.now(TZ).strftime('%Y%m%dT%H%M%S')
    arms = {role:f'duck_getup_right-upright-ppo_{role}_s{p["seed"]}_{stamp}' for role in ROLES}
    assert not any((ROOT/'records/experiments'/run).exists() for run in arms.values())
    paths = (zero.PLAN, receipt, SCRIPT, SCRIPT.with_suffix('.sh'), SCRIPT.with_name('watch_getup_right_upright_ppo.py'),
             SCRIPT.with_name('verify_getup_right_upright_ppo.py'), SCRIPT.with_name('evaluate_getup_right_upright_ppo.py'))
    spec['sources'].update({startup.relative(path):sha(path) for path in paths})
    spec.update(status='frozen-matched-online-ppo',stage=STAGE,arms=arms,updates=updates,controls=controls,
                limit=dict(controls=controls,substeps=controls*5,transitions=controls*n,
                           rollout_controls=horizon,wall_seconds_per_arm=900),
                reward_audit_ages=[100,120,140,160,180,220,400], observation=startup.relative(PAGE),
                complete_episodes=n*(controls//400), scope='Matched PPO with signed right-side progress only in candidate.')
    spec['evaluation_gates']['requested_pose_counts'] = [32,32,128,128]
    verify_sources(spec)
    OUT.mkdir()
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),zero_sha256=sha(receipt),
          physical_substeps=0,learning_updates=0,observation_reference=spec['observation_reference']))
    print('RIGHT_UPRIGHT_PPO_FROZEN',sha(PLAN),arms,flush=True)


def verify_sources(spec):
    startup.value.checked()
    for path,digest in spec['sources'].items():
        assert sha(ROOT/path) == digest, path


def require_stage():
    check_project()
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage'] == STAGE and stage['current_protocol_frozen'] and ROOT/stage['current_protocol'] == PLAN


def checked():
    require_stage()
    spec = read(PLAN)
    assert spec['status'] == 'frozen-matched-online-ppo' and not spec['heldout_used']
    assert read(OUT/'preflight.json')['protocol_sha256'] == sha(PLAN)
    verify_sources(spec)
    return spec


def audit_reward(env,bridge,spec):
    manager = env.reward_manager
    assert manager._scale_by_dt and manager._term_names[-1] == native.TERM
    native_term, compute, step = bridge.native, manager.compute, env.step
    audits, terminals = [], []
    raw, controls = None, 0

    def read_native(actual):
        nonlocal raw
        assert actual is env and raw is None
        raw = native_term(actual)
        return raw

    def read_compute(dt):
        nonlocal raw, controls
        controls += 1
        total = compute(dt)
        age = (controls-1)%400+1
        if age in spec['reward_audit_ages']:
            assert dt == .02 and raw.isfinite().all()
            expected = raw*bridge.term.weight*dt/dt if bridge.enabled else torch.zeros_like(raw)
            torch.testing.assert_close(manager._step_reward[:,-1],expected,rtol=0,atol=0)
            _,feet,head = native.inputs(env)
            exact_feet,exact_head = native_contact(env)
            torch.testing.assert_close(feet,exact_feet,rtol=0,atol=0)
            torch.testing.assert_close(head,exact_head,rtol=0,atol=0)
            audits.append(dict(control=controls,age=age,raw_nonzero=int(raw.count_nonzero()),
                               contribution_nonzero=int(expected.count_nonzero()),
                               triggered=int((bridge.state.trigger_age>=0).sum()),native_contact_equal=True))
        raw = None
        return total

    def read_step(action):
        result = step(action)
        if controls%400 == 0:
            state = bridge.state
            assert not state.release_known.any() and not state.supported.any() and (state.trigger_age==-1).all()
            terminals.append(dict(control=controls,reset_worlds=env.num_envs,cleared=True))
        return result

    bridge.native, manager.compute, env.step = read_native, read_compute, read_step
    return dict(reward=audits,terminals=terminals)


def arm_plan(spec,role):
    return dict(deepcopy(spec['environment']),run_id=spec['arms'][role],arm=role,training_updates=spec['updates'])


def worker(spec,role):
    from mjlab.utils.torch import configure_torch_backends

    require_stage()
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    p = arm_plan(spec,role)
    n,updates = p['num_envs'],p['training_updates']
    record = ROOT/'records/experiments'/p['run_id']
    reference = zero.OUT/role
    watch = MemoryWatch(spec['memory_limit'])
    rows = []
    audit, recorder, began = None, None, None
    video = rollout.ActionVideo

    def initialize(plan,output,scene):
        nonlocal audit
        _,display,metadata = prepare(plan)
        env,obs,learner,prediction,bridge = startup.initialize(spec,role=='candidate',output,watch.check)
        try:
            match = startup.matched_signature(env,obs,learner,prediction,bridge)
            # The comparison uses a common JSON representation of Adam's integer keys.
            assert json.loads(json.dumps(match,allow_nan=False)) == read(reference/'matched-state.json')
            write(output/'matched-state.json',match)
            write(output/'display.json',display)
            save_checkpoint(output/'model-init.pt',learner,env,0)
            audit = audit_reward(env,bridge,spec)
            watch.sample()
            return env,obs,learner,metadata
        except BaseException:
            env.close()
            raise

    def create_video(*args,**kwargs):
        nonlocal recorder
        recorder = video(*args,**kwargs)
        return recorder

    def match_start(output,expected):
        assert sha(output/'matched-state.json') == sha(expected/'matched-state.json')
        if role == 'candidate':
            control = ROOT/'records/experiments'/spec['arms']['control']
            assert sha(output/'matched-state.json') == sha(control/'matched-state.json')

    def publish(path,state):
        nonlocal began
        if path == record/'command.json':
            state = dict(state,visible_tty=False,process_pty=os.isatty(1),argv=sys.argv,pid=os.getpid())
        write(path,state)
        if path != record/'state.json':
            return
        if not (record/'README.md').exists():
            (record/'README.md').write_text(
                f'# 右侧起身 · {role}\n\n[冻结方案](../../../{startup.relative(PLAN)}) · '
                f'[观察页](../../{PAGE.name}) · [逐次更新](observation.json) · '
                f'[执行日志](../../reports/{OUT.name}/{role}-execution.log)\n',encoding='utf-8')
        if began is None and state['status'] == 'training':
            began = perf_counter()
        metric = state.get('metrics')
        if metric and metric['update'] > len(rows):
            assert metric['update'] == len(rows)+1
            rows.append(dict(metric,net_transitions_per_second=metric['controls']*n/(perf_counter()-began),
                             gpu_peak_mib=watch.summary()['peak_mib'],video_sample_seconds=recorder.elapsed_seconds))
        log = startup.relative(OUT/f'{role}-execution.log').removeprefix('records/')
        live.publish(record,dict(state,pair_running=True,live_log=log),rows,n,updates,watch.summary())

    try:
        with watch.active(), patch.object(live,'PAGE',PAGE), patch.multiple(
            rollout,initialize=initialize,write=publish,ActionVideo=create_video,check_matched_start=match_start,
        ):
            state = rollout.train(p,record,native.SCENE,reference,sha(PLAN),spec['limit'],watch.check)
            watch.check()
        assert state['status'] == 'collected-review-required' and len(rows) == updates
        assert sum(row['completed_episodes'] for row in rows) == spec['complete_episodes']
        write(record/'online-result.json',dict(status='collected-review-required',actual_transitions=state['controls']*n,
              actual_physical_substeps=state['substeps'],actual_updates=updates,elapsed_seconds=state['elapsed_seconds'],
              ppo_seconds=sum(row['ppo_seconds'] for row in rows),memory=watch.summary(),heldout_used=False,
              samples_sha256=sha(record/'learning-metrics.jsonl'),observation_sha256=sha(record/'observation.json')))
    finally:
        if record.is_dir():
            write(record/'gpu-telemetry.json',dict(samples=watch.rows,failure=repr(watch.error) if watch.error else None))
            if audit is not None:
                write(record/'right-upright-audit.json',audit)


if __name__ == '__main__':
    if sys.argv[1:] == ['--freeze']:
        freeze()
    else:
        with patch.multiple(paired,PLAN=PLAN,OUT=OUT,PAGE=PAGE,STAGE=STAGE,__file__=str(SCRIPT),
                            checked=checked,require_stage=require_stage,worker=worker):
            paired.main()
