"""Admit native reward readback on a fresh full CUDA episode, without learning."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np
import torch

import recover_getup_early_support_ppo_start as stopped
import run_getup_early_support_ppo as original
import training_live_page as live
from action_video import ActionVideo
from check_getup_early_support_clock import TZ
from getup_anchor_runtime import RecordingSnapshot
from getup_early_support_reward import component
from getup_release_fork_state import signature
from local_getup_scale import now
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_early_support_native_reward.json'
OUT = ROOT/'records/reports/getup-early-support-native-reward'
PAGE = ROOT/'records/getup-early-support-native-reward-live.html'
STAGE = 'S3-early-support-native-reward-preflight'


def native_rate(*args,**kwargs):
    # Match the installed RewardManager for this weight-one term. Comparing
    # the recorded manager rate directly to the raw term loses a float32 bit.
    raw = component(*args,**kwargs)
    return torch.nan_to_num(raw*1.*.02,nan=0.,posinf=0.,neginf=0.)/.02


def freeze():
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    receipt = stopped.OUT/'stop-analysis.json'
    proof = read(receipt)
    assert proof['status']=='verified-arithmetic-stop' and proof['native_arithmetic']['manager_roundtrip_exact']
    assert proof['candidate_completed_actor_updates']==0 and not proof['heldout_used']
    spec = deepcopy(read(stopped.PLAN))
    p = spec['environment']
    seed = 1085
    assert not any(f'_s{seed}_' in folder.name for folder in (ROOT/'records/experiments').iterdir())
    p.update(seed=seed,anchor_seed=108504,retention_seed=108505)
    p['learner']['minibatch_seed'] = 108503
    spec.update(stage=STAGE,status='frozen-native-reward-readback-only',controls=400,learning_updates=0,
        run_id=f"duck_getup_early-support-native-reward_development_s{seed}_{datetime.now(TZ).strftime('%Y%m%dT%H%M%S')}",
        scope='Fresh 3072-world matched zero step, then candidate fixed-actor one full episode. No optimizer update.')
    for path in (receipt,stopped.PLAN,Path(__file__).resolve()):
        spec['sources'][original.relative(path)] = sha(path)
    original.verify_sources(spec)
    OUT.mkdir()
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='passed-cpu-source',protocol_sha256=sha(PLAN),physical_substeps=0,
        learning_updates=0,seed=seed,storage_roundtrip_verified=proof['native_arithmetic']['manager_roundtrip_exact']))
    print('EARLY_SUPPORT_NATIVE_REWARD_FROZEN',sha(PLAN),spec['run_id'],flush=True)


def checked():
    spec = read(PLAN)
    assert spec['status']=='frozen-native-reward-readback-only' and spec['learning_updates']==0
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(PLAN)
    original.verify_sources(spec)
    return spec


def require_stage():
    check_project()
    p = read(ROOT/'project.json')['staged_training_request']
    assert p['current_stage']==STAGE and p['current_protocol_frozen'] and ROOT/p['current_protocol']==PLAN


def collect(spec, candidate):
    from mjlab.utils.torch import configure_torch_backends

    require_stage()
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    record = ROOT/'records/experiments'/spec['run_id'] if candidate else OUT/'control-zero'
    record.mkdir()
    watch = MemoryWatch(spec['memory_limit'])
    n = spec['environment']['num_envs']
    duration = spec['controls']
    assert n==3072 and duration==400
    controls = 0
    rates = []
    clip = None

    def publish(status):
        live.publish(record,dict(status=status,run_id=spec['run_id'],purpose='起身原生奖励读回检查',
            updated_at=now(),controls=controls,maximum_controls=duration,transitions=controls*n,learning_updates=0,
            updates=rates,latest_clip=clip,frame='frame-step-0.jpg' if (record/'frame-step-0.jpg').is_file() else None))

    if candidate:
        (record/'README.md').write_text('# 原生奖励读回检查\n\n固定策略采样一个完整起身回合，核对实际奖励与重置。\n\n'
            '[冻结方案](../../../configs/experiments/getup_early_support_native_reward.json) · [状态](state.json) · '
            '[协调日志](../../reports/getup-early-support-native-reward/candidate.log)\n',encoding='utf-8')
        write(record/'protocol.json',spec)
        write(record/'command.json',dict(argv=sys.argv,pid=os.getpid(),started_at=now()))

    with watch.active(), patch.object(live,'PAGE',PAGE):
        env = snapshot = recorder = None
        success = False
        try:
            if candidate:
                publish('running')
            env,obs,learner,prediction = original.zero.initialize(spec,candidate,record,watch.check)
            watch.sample()
            initial = json.loads(json.dumps(original.initial_signature(env,obs,learner,prediction),allow_nan=False))
            write(record/'matched-state.json',initial)
            assert env._sim_step_counter==0
            if not candidate:
                write(record/'result.json',dict(status='passed-cuda-zero',physical_substeps=0,memory=watch.summary()))
                return
            other = OUT/'control-zero/matched-state.json'
            assert initial==read(other)
            write(record/'zero-step.json',dict(status='passed-matched-cuda-zero',physical_substeps=0,
                  counterpart_sha256=sha(other),candidate_sha256=sha(record/'matched-state.json'),memory=watch.summary()))
            fixed = signature(learner.actor.state_dict())
            critic = signature(learner.critic.state_dict())
            snapshot = RecordingSnapshot(env,record)
            recorder = ActionVideo(snapshot,'getup',record)
            recorder.sample()
            first = env.sim.data.qpos[0].cpu().numpy().copy()
            trajectory = [first]
            terminal = []
            hook = env.recorder_manager.record_pre_reset

            def terminal_state(ids):
                assert (ids==0).any()
                terminal.append(env.sim.data.qpos[0].cpu().numpy().copy())
                hook(ids)

            env.recorder_manager.record_pre_reset = terminal_state
            began = perf_counter()
            with patch.object(original,'component',native_rate),torch.no_grad():
                original.install_reward_readback(env,spec,record,True)
                for k in range(1,duration+1):
                    action = learner.actor(obs,stochastic_output=True)
                    obs,_,ended,timeout,_ = env.step(action)
                    controls = k
                    assert not ended.any() and bool(timeout.all() if k==400 else ~timeout.any())
                    if k<400:
                        trajectory.append(env.sim.data.qpos[0].cpu().numpy().copy())
                    if recorder.writer is not None:
                        recorder.sample()
                    if k%25==0:
                        watch.check()
                        require_stage()
                        rates.append(dict(control=k,transitions_per_second=k*n/(perf_counter()-began)))
                        publish('running')
                        print('EARLY_SUPPORT_NATIVE_REWARD',k,'/',duration,flush=True)
            assert env._sim_step_counter==duration*5 and len(terminal)==1
            assert fixed==signature(learner.actor.state_dict()) and critic==signature(learner.critic.state_dict())
            assert learner.actor_steps==learner.critic_steps==0
            assert len(recorder.clips)==1 and recorder.clips[0]['complete_clip']
            clip = recorder.clips[0]['file']
            np.save(record/'camera-qpos.npy',np.stack(trajectory+terminal),allow_pickle=False)
            write(record/'result.json',dict(status='collected-review-required',physical_substeps=duration*5,
                  transitions=duration*n,learning_updates=0,actor_unchanged=True,critic_unchanged=True,
                  memory=watch.summary(),wall_seconds=perf_counter()-began))
            success = True
        finally:
            if recorder is not None:
                recorder.close()
            if snapshot is not None:
                snapshot.close()
            if env is not None:
                env.close()
            if candidate:
                write(record/'gpu-samples.json',watch.rows)
                publish('collected-review-required' if success else 'stopped-review-required')


def verify(spec):
    assert not torch.cuda.is_initialized()
    record = ROOT/'records/experiments'/spec['run_id']
    run = read(record/'result.json')
    zero = read(record/'zero-step.json')
    assert zero['status']=='passed-matched-cuda-zero' and zero['physical_substeps']==0
    assert zero['counterpart_sha256']==zero['candidate_sha256']==sha(record/'matched-state.json')
    assert run['learning_updates']==0 and run['actor_unchanged'] and run['critic_unchanged']
    assert run['physical_substeps']==2000 and run['transitions']==1228800
    audit = read(record/'early-support-audits.json')
    assert [row['episode_age'] for row in audit['reward']]==[114,115,125,150,151,400]
    assert audit['terminals']==[dict(control=400,reset_worlds=3072,next_latch_zero=True)]
    assert any(row.get('active_positive',0)>0 for row in audit['reward'])
    assert all(row.get('native_contact_equal') or row.get('outside_window_rate_zero') for row in audit['reward'])
    clip, = read(record/'clips.json')
    source = np.load(record/'camera-qpos.npy',allow_pickle=False)
    steps = list(range(0,401,2))
    assert clip['common_steps']==steps and clip['complete_clip'] and clip['extra_physics_steps']==0
    assert len(source)==401
    expected = [dict(common_step=step,qpos_sha256=hashlib.sha256(source[step].tobytes()).hexdigest()) for step in steps]
    assert read(record/'recording-states.json')==expected
    video = record/clip['file']
    frames = imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
    assert next(frames)['fps']==25 and sum(1 for _ in frames)==201
    proof = dict(status='passed-actual-native-reward-preflight',protocol_sha256=sha(PLAN),record=original.relative(record),
        native_audits=audit,video=dict(path=original.relative(video),sha256=sha(video),frames=201,seconds=8.),
        transitions=run['transitions'],physical_substeps=run['physical_substeps'],memory=run['memory'],
        heldout_used=False,learning_updates=0)
    write(OUT/'analysis.json',proof)
    state = read(record/'state.json')
    state.update(status='completed',updated_at=now())
    with patch.object(live,'PAGE',PAGE):
        live.publish(record,state)
    print('EARLY_SUPPORT_NATIVE_REWARD_PASSED',run['transitions'],flush=True)


def main():
    args = sys.argv[1:]
    if args==['--freeze']:
        return freeze()
    spec = checked()
    if args==['--verify']:
        return verify(spec)
    if args in (['control'],['candidate']):
        return collect(spec,args==['candidate'])
    assert not args and os.isatty(1)
    for role in ('control','candidate'):
        argv = [sys.executable,'-B',__file__,role]
        write(OUT/f'{role}-command.json',dict(argv=argv,started_at=now()))
        with (OUT/f'{role}.log').open('x') as log:
            code = subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
        write(OUT/f'{role}-exit.json',dict(exit_code=code,ended_at=now()))
        if code:
            raise SystemExit(code)


if __name__=='__main__':
    main()
