"""Run the admitted matched pair, with one enabled reward as the experimental change."""
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

import getup_early_support_ppo_zero as zero
import getup_ppo_live as live
import run_getup_supine_training as rollout
from check_getup_early_support_clock import TERM, native_contact, TZ
from getup_early_support_reward import component, inputs
from getup_release_fork_state import signature
from getup_teacher_anchor_train import prepare, save_checkpoint
from local_getup_scale import now
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_early_support_ppo.json'
OUT = ROOT/'records/reports/getup-early-support-ppo'
PAGE = ROOT/'records/getup-early-support-ppo-live.html'
STAGE = 'S3-early-support-paired-ppo-development'
ROLES = ('control', 'candidate')


def relative(path):
    return path.relative_to(ROOT).as_posix()


def verify_sources(spec):
    zero.checked_value()
    for source, expected in spec['sources'].items():
        assert sha(ROOT/source) == expected, source


def freeze():
    assert not PLAN.exists() and not OUT.exists()
    assert not torch.cuda.is_initialized()
    spec = deepcopy(read(zero.PLAN))
    draft = read(zero.DRAFT)
    design = draft['paired_development']
    proof_file = zero.OUT/'analysis.json'
    proof = read(proof_file)
    assert proof['status'] == 'passed-actual-matched-cuda-zero' and not proof['mismatched']
    assert proof['protocol_sha256'] == sha(zero.PLAN) and not proof['heldout_used']
    for role in ROLES:
        result = proof['runs'][role]['result']
        assert result['physical_substeps'] == result['actor_updates'] == result['critic_updates'] == 0
        assert result['environments'] == design['environments_per_arm'] == 3072
        assert result['peak_gpu']['peak_mib']/result['peak_gpu']['gpu']['total_mib'] < design['memory_fraction_limit']
        state = zero.OUT/role/'matched-state.json'
        assert sha(state) == proof['common_state_signatures'][role]
        spec['sources'][relative(state)] = sha(state)
    count, horizon = design['updates_per_arm'], design['rollout_controls']
    n, seed = spec['environment']['num_envs'], spec['environment']['seed']
    controls = count*horizon
    assert controls*n == design['control_transitions_per_arm'] == 1253376
    stamp = datetime.now(TZ).strftime('%Y%m%dT%H%M%S')
    arms = {role: f'duck_getup_early-support-ppo_{role}_s{seed}_{stamp}' for role in ROLES}
    assert not any((ROOT/'records/experiments'/name).exists() for name in arms.values())
    limit = dict(controls=controls, substeps=controls*5, transitions=controls*n,
                 rollout_controls=horizon, wall_seconds_per_arm=900)
    files = [zero.PLAN, proof_file, Path(__file__).resolve(),
             Path(__file__).with_name('verify_getup_early_support_ppo.py'),
             Path(__file__).with_name('verify_getup_early_support_rollout.py'),
             Path(__file__).with_name('watch_getup_early_support_ppo.py')]
    for path in files:
        spec['sources'][relative(path)] = sha(path)
    value_record = ROOT/read(zero.VALUE_OUT/'analysis.json')['record']
    windows_view = value_record/'windows-view-verification.json'
    observation_proof = read(windows_view)
    assert observation_proof['status'] == 'passed' and observation_proof['running_snapshots'] >= 3
    assert observation_proof['snapshots_with_complete_clip'] > 0
    spec['sources'][relative(windows_view)] = sha(windows_view)
    spec.update(status='frozen-matched-online-ppo', stage=STAGE, arms=arms, updates=count,
                controls=controls, limit=limit, reward_audit_ages=[115,125,150], observation=relative(PAGE),
                complete_episodes=n*(controls//400), scope='Fresh matched PPO from admitted seed1083 starts; only early reward differs')
    verify_sources(spec)
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT/'preflight.json', dict(status='passed', protocol_sha256=sha(PLAN), zero_sha256=sha(proof_file),
          prior_windows_observation=relative(windows_view), physical_substeps=0, learning_updates=0))
    print('EARLY_SUPPORT_PPO_FROZEN',sha(PLAN),arms,flush=True)


def checked():
    spec = read(PLAN)
    assert spec['status'] == 'frozen-matched-online-ppo' and not spec['heldout_used']
    assert read(OUT/'preflight.json')['protocol_sha256'] == sha(PLAN)
    verify_sources(spec)
    return spec


def require_stage():
    check_project()
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage'] == STAGE and stage['current_protocol_frozen'] and ROOT/stage['current_protocol'] == PLAN


def arm_plan(spec, role):
    p = deepcopy(spec['environment'])
    p.update(run_id=spec['arms'][role], arm=role, training_updates=spec['updates'],
             early_support_enabled=role=='candidate')
    return p


def initial_signature(env, features, learner, prediction):
    raw = env.action_manager.get_term('joint_pos')
    return signature(dict(observations=features, critic_output=prediction,
        actor=learner.actor.state_dict(), critic=learner.critic.state_dict(),
        actor_adam=learner.actor_optimizer.state_dict(), critic_adam=learner.critic_optimizer.state_dict(),
        replay_rng=learner.generator.get_state(), anchor_rng=learner.anchor_generator.get_state(),
        qpos=env.sim.data.qpos, qvel=env.sim.data.qvel, control=env.sim.data.ctrl, ages=env.episode_length_buf,
        action_history=raw._raw_actions, applied=raw._processed_actions, initial_imu=env._imu_misalign_quat,
        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(env.device),
        expanded={name: getattr(env.sim.model,name) for name in sorted(env.sim.expanded_fields)}))


def install_reward_readback(env, spec, record, enabled):
    manager = env.reward_manager
    assert manager._term_names[-1] == TERM and manager._scale_by_dt
    native_compute, native_step = manager.compute, env.step
    audits, terminals = [], []
    path = record/'early-support-audits.json'

    def readback(dt):
        result = native_compute(dt)
        assert dt == .02
        control = int(env.common_step_counter)
        age = (control-1)%400+1
        rate = manager._step_reward[:,-1]
        if age in spec['reward_audit_ages']:
            z, feet, head = inputs(env)
            raw_feet, raw_head = native_contact(env)
            torch.testing.assert_close(feet,raw_feet,atol=0,rtol=0)
            torch.testing.assert_close(head,raw_head,atol=0,rtol=0)
            latch = env._early_support_release
            assert latch.latched.all() and (env.episode_length_buf==age).all()
            expected = component(z,raw_feet,raw_head,env.episode_length_buf,latch.supine,**spec['reward'])
            torch.testing.assert_close(rate,expected if enabled else torch.zeros_like(rate),atol=0,rtol=0)
            audits.append(dict(control=control, episode_age=age, enabled=enabled, native_contact_equal=True,
                latched_supine=int(latch.supine.sum()), raw_positive=int((expected>0).sum()), active_positive=int((rate>0).sum()),
                maximum_absolute_rate=float(rate.abs().max())))
        elif age in (114,151,400):
            assert not rate.any()
            audits.append(dict(control=control, episode_age=age, enabled=enabled, outside_window_rate_zero=True))
        return result

    def step(action):
        result = native_step(action)
        if env.common_step_counter%400 == 0:
            assert result[3].all() and not result[2].any()
            assert not env._early_support_release.latched.any() and not result[0]['critic'][:,75].any()
            terminals.append(dict(control=int(env.common_step_counter), reset_worlds=int(result[3].sum()),
                                  next_latch_zero=True))
            write(path,dict(reward=audits,terminals=terminals))
        return result

    manager.compute = readback
    env.step = step


def worker(spec, role):
    from mjlab.utils.torch import configure_torch_backends

    require_stage()
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    p = arm_plan(spec, role)
    record = ROOT/'records/experiments'/p['run_id']
    zero_record = zero.OUT/role
    n, count = p['num_envs'], p['training_updates']
    watch = MemoryWatch(spec['memory_limit'])
    rows, start, recorder = [], None, None
    base_video = rollout.ActionVideo

    def initialize(plan, output, _scene):
        # Configure export metadata before initialization, then verify the entire
        # live startup signature against the frozen no-physics run.
        _, display, metadata = prepare(plan)
        env, obs, learner, prediction = zero.initialize(spec,plan['early_support_enabled'],output,watch.check)
        try:
            actual = initial_signature(env,obs,learner,prediction)
            assert actual == read(zero_record/'matched-state.json')
            write(output/'matched-state.json',actual)
            write(output/'display.json',display)
            save_checkpoint(output/'model-init.pt',learner,env,0)
            install_reward_readback(env,spec,output,plan['early_support_enabled'])
            watch.sample()
            return env,obs,learner,metadata
        except BaseException:
            env.close()
            raise

    def match_start(output, reference):
        assert sha(output/'matched-state.json') == sha(reference/'matched-state.json')
        if role == 'candidate':
            control = ROOT/'records/experiments'/spec['arms']['control']
            assert sha(output/'matched-state.json') == sha(control/'matched-state.json')

    def video(*args, **kwargs):
        nonlocal recorder
        recorder = base_video(*args, **kwargs)
        return recorder

    def publish_state(path, data):
        nonlocal start
        if path == record/'command.json':
            data = dict(data,visible_tty=os.isatty(1),argv=sys.argv,pid=os.getpid(),started_at=now())
        write(path,data)
        if path != record/'state.json':
            return
        if not (record/'README.md').exists():
            (record/'README.md').write_text(
                f'# 起身早期支撑 · {role}\n\n'
                '[冻结方案](../../../configs/experiments/getup_early_support_ppo.json) · '
                '[运行观察](../../getup-early-support-ppo-live.html) · [逐次更新](observation.json) · '
                f'[执行日志](../../reports/getup-early-support-ppo/{role}-execution.log)\n',encoding='utf-8')
        if start is None and data['status']=='training':
            start = perf_counter()
        metric = data.get('metrics')
        if metric and metric['update'] > len(rows):
            assert metric['update'] == len(rows)+1
            elapsed = perf_counter()-start
            rows.append(dict(metric, net_transitions_per_second=metric['controls']*n/elapsed,
                             gpu_peak_mib=watch.summary()['peak_mib'],video_sample_seconds=recorder.elapsed_seconds))
        log = relative(OUT/f'{role}-execution.log').removeprefix('records/')
        live.publish(record,dict(data,pair_running=True,live_log=log),rows,n,count,watch.summary())

    try:
        with watch.active(), patch.object(live,'PAGE',PAGE), patch.object(rollout,'initialize',initialize), \
             patch.object(rollout,'check_matched_start',match_start), patch.object(rollout,'write',publish_state), \
             patch.object(rollout,'ActionVideo',video):
            state = rollout.train(p,record,zero.SCENE,zero_record,sha(PLAN),spec['limit'],watch.check)
            watch.check()
        assert state['status']=='collected-review-required' and len(rows)==count
        assert sum(row['completed_episodes'] for row in rows)==spec['complete_episodes']
        write(record/'online-result.json',dict(status='collected-review-required',actual_transitions=state['controls']*n,
              actual_physical_substeps=state['substeps'],actual_updates=count,elapsed_seconds=state['elapsed_seconds'],
              ppo_seconds=sum(row['ppo_seconds'] for row in rows),memory=watch.summary(),heldout_used=False,
              samples_sha256=sha(record/'learning-metrics.jsonl'),observation_sha256=sha(record/'observation.json')))
    finally:
        if record.is_dir():
            write(record/'gpu-telemetry.json',dict(samples=watch.rows,failure=repr(watch.error) if watch.error else None))


def main():
    if sys.argv[1:] == ['--freeze']:
        return freeze()
    spec = checked()
    require_stage()
    if len(sys.argv)>1:
        assert len(sys.argv)==2 and sys.argv[1] in ROLES
        return worker(spec,sys.argv[1])
    assert os.isatty(1)
    entry = read(OUT/'windows-entry.json')
    assert entry['status']=='passed' and entry['protocol_sha256']==sha(PLAN) and PAGE.is_file()
    path = OUT/'coordinator.json'
    assert not path.exists(), 'A frozen run is single-use'
    state = dict(status='running',started_at=now(),pid=os.getpid(),protocol_sha256=sha(PLAN),arms={})
    for role in ROLES:
        require_stage()
        record = ROOT/'records/experiments'/spec['arms'][role]
        argv = [sys.executable,'-B',str(Path(__file__).resolve()),role]
        log = OUT/f'{role}-execution.log'
        with log.open('x',buffering=1) as file:
            with subprocess.Popen(argv,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT) as child:
                state.update(role=role,worker_pid=child.pid)
                write(path,state)
                try:
                    code = child.wait(timeout=spec['limit']['wall_seconds_per_arm']+120)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    code = 124
        state['arms'][role] = dict(record=relative(record),argv=argv,execution_log=relative(log),exit_code=code)
        write(path,state)
        print('EARLY_SUPPORT_PPO_ARM',role,code,flush=True)
        if code:
            break
    complete = len(state['arms'])==2 and all(arm['exit_code']==0 for arm in state['arms'].values())
    state.update(status='collected-review-required' if complete else 'stopped-review-required',ended_at=now())
    write(path,state)
    if (record/'observation.json').exists():
        observation, physical = read(record/'observation.json'),read(record/'state.json')
        with patch.object(live,'PAGE',PAGE):
            live.publish(record,dict(physical,status=state['status'],live_log=relative(log).removeprefix('records/')),
                         observation['rows'],observation['environments'],observation['planned_updates'],
                         observation['memory'],write_snapshot=False)
    if not complete:
        raise SystemExit(2)


if __name__=='__main__':
    main()
