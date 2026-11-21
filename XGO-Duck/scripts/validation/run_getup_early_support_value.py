"""Fit one common value start from a new fixed-actor training-plant rollout."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
from time import monotonic, perf_counter
from unittest.mock import patch

import numpy as np
import torch

from s1_reference_inputs import ROOT, WORK, read, sha
from action_video import ActionVideo
from check_getup_early_support_clock import OUT as CLOCK_OUT, PLAN as CLOCK_PLAN, TZ, TERM, config, cpu, native_contact
from common_scene_contract import compare_arrays, model_arrays
from getup_anchor_runtime import RecordingSnapshot, startup
from getup_anchor_same_release import visible_log
from getup_contact_training import POSES, discounted_returns, split_worlds
from getup_early_support_reward import SupineReleaseLatch, inputs
from getup_release_fork_state import signature
from getup_value_bootstrap import fit, networks, td
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
import training_live_page as live


PLAN = ROOT/'configs/experiments/getup_early_support_value.json'
OUT = ROOT/'records/reports/getup-early-support-value'
PAGE = ROOT/'records/getup-early-support-value-live.html'
STAGE = 'S3-early-support-value-development'
SCENE = ROOT/'records/reports/getup-interface-value/cpu-scene-fields.npz'


def check_sources(spec):
    for base, items in ((ROOT,spec['references']),(ROOT,spec['source_sha256']),
                        (WORK,spec['environment']['training_source_sha256']),(Path('/'),spec['library_source_sha256'])):
        for name, expected in items.items():
            assert sha(base/name)==expected,name


def checked():
    spec = read(PLAN)
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(PLAN)
    check_sources(spec)
    return spec


def freeze():
    from mjlab.scene import Scene

    assert not PLAN.exists() and not OUT.exists()
    receipt = CLOCK_OUT/'analysis.json'
    proof = read(receipt)
    assert proof['status']=='passed-actual-cuda-reward-clock' and proof['protocol_sha256']==sha(CLOCK_PLAN)
    assert not proof['heldout_used'] and proof['learning_updates']==0 and proof['positive_reward_samples']>0
    assert sha(ROOT/proof['value_target']['path'])==proof['value_target']['sha256']
    spec = deepcopy(read(CLOCK_PLAN))
    check_sources(spec)
    prior = ROOT/'configs/experiments/getup_persistent_value.json'
    baseline = read(prior)
    p = spec['environment']
    seed = 1082
    assert not any(f'_s{seed}_' in f.name for f in (ROOT/'records/experiments').iterdir())
    for name in ('critic_updates','minibatch_size','critic_learning_rate','gradient_norm_max','training_metric_samples',
                 'evaluation_updates','value_gate'):
        p[name] = baseline[name]
    p.update(seed=seed,num_envs=1024,split=split_worlds(seed=108202),minibatch_seed=108203,stage=STAGE,
             run_id=f"duck_getup_early-support-value_development_s{seed}_{datetime.now(TZ).strftime('%Y%m%dT%H%M%S')}")
    assert p['critic_updates']==50 and len(p['split']['train_ids'])==960 and len(p['split']['validation_ids'])==64
    spec.update(stage=STAGE,controls=400,physics_substeps=2000,transitions=409600,
        critic_inputs=76,actor_updates=0,maximum_seconds=240,memory_limit=.85,reward_audit_controls=[115,125,150],
        scope='One fresh shared critic on the mean of two finite returns. Validation by reserved development world and pose; no actor updates.')
    del spec['partial_reset_control']
    for path in (CLOCK_PLAN,receipt,prior,ROOT/proof['value_target']['path']):
        spec['references'][path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (Path(__file__).resolve(),Path(__file__).resolve().with_name('verify_getup_early_support_value.py'),
                 ROOT/'scripts/training/getup_value_bootstrap.py',ROOT/'scripts/validation/run_local_getup_ppo.py'):
        spec['source_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    check_sources(spec)
    cfg,_ = config(spec)
    with np.load(SCENE) as expected:
        compare_arrays(model_arrays(Scene(cfg.env.scene,device='cpu').compile()),{k:expected[k] for k in expected.files})
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='passed-cpu-zero-step',protocol_sha256=sha(PLAN),physical_substeps=0,
          common_scene_exact=True,clock_proof_sha256=sha(receipt),critic_width=76,actor_width=61))
    print('EARLY_SUPPORT_VALUE_FROZEN',sha(PLAN),p['run_id'],flush=True)


def scores(prediction,returns,p):
    valid = np.asarray(p['split']['validation_ids'])
    codes = np.asarray(p['split']['pose_codes'])[valid]

    def metrics(pred,target):
        error = pred.astype(float)-target
        return dict(normalized_mse=float((error**2).mean()),explained_variance=float(1-error.var()/target.var()))

    result = {}
    gate = p['value_gate']
    for label,target in returns.items():
        report = metrics(prediction,target)
        poses = {name:metrics(prediction[:,codes==i],target[:,codes==i]) for i,name in enumerate(POSES)}
        passed = (report['explained_variance']>=gate['validation_explained_variance_min'] and
                  report['normalized_mse']<=gate['validation_normalized_mse_max'] and
                  all(row['explained_variance']>=gate['each_pose_validation_explained_variance_min'] for row in poses.values()))
        result[label] = dict(validation=report,poses=poses,gate_passed=bool(passed))
    return result


def run():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.os import dump_yaml

    assert os.isatty(1)
    spec = checked()
    p = spec['environment']
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage']==STAGE and stage['current_protocol_frozen'] and ROOT/stage['current_protocol']==PLAN
    record = ROOT/'records/experiments'/p['run_id']
    record.mkdir()
    shutil.copy2(PLAN,record/'protocol.json')
    (record/'README.md').write_text('# 仰卧早期支撑价值起点\n\n[冻结方案](../../../configs/experiments/getup_early_support_value.json) · '
        '[执行日志](execution.log) · [状态](state.json)\n',encoding='utf-8')
    env = video = snapshot = None
    controls, last, updates = 0,0,0
    instant, rates = perf_counter(),[]
    watch = MemoryWatch(spec['memory_limit'])

    def publish(status):
        nonlocal last, instant
        now = perf_counter()
        if controls>last:
            rates.append(dict(control=controls,transitions_per_second=(controls-last)*p['num_envs']/(now-instant)))
            last,instant = controls,now
        clips = video.clips if video is not None else []
        clip = next((row['file'] for row in clips if row['complete_clip']),None)
        live.publish(record,dict(status=status,run_id=p['run_id'],purpose='仰卧早期支撑 · 价值起点',
            updated_at=datetime.now(TZ).isoformat(),controls=controls,maximum_controls=400,
            transitions=controls*p['num_envs'],learning_updates=updates,updates=rates,latest_clip=clip,
            frame='frame-step-0.jpg' if (record/'frame-step-0.jpg').is_file() else None))

    def fitting(**row):
        nonlocal updates
        watch.check()
        updates = row['completed_updates']
        publish('running')

    status = 'stopped-review-required'
    with (record/'execution.log').open('x',buffering=1) as log, visible_log(log), patch.object(live,'PAGE',PAGE), watch.active():
        publish('running')
        try:
            configure_torch_backends(allow_tf32=False)
            torch.set_num_threads(4)
            cfg,display = config(spec)
            write(record/'display.json',display)
            dump_yaml(record/'env.yaml',asdict(cfg.env))
            env = ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None)
            latch = SupineReleaseLatch(env.num_envs,env.device,spec['release_age'],spec['alignment'])
            env._early_support_release = latch
            obs,_ = env.reset(seed=p['seed'])
            actor,critic = networks(76,p,env.device)
            startup(env,obs,actor,p,record,SCENE)
            actor_initial = signature(actor.state_dict())
            watch.sample()
            assert env._sim_step_counter==0 and not latch.latched.any()
            write(record/'zero-step.json',dict(status='passed',actor_width=61,critic_width=76,gpu=watch.summary(),physical_substeps=0))
            snapshot = RecordingSnapshot(env,record)
            video = ActionVideo(snapshot,'getup',record)
            video.sample()
            first_qpos = cpu(env.sim.data.qpos[0])
            manager = env.reward_manager
            assert manager._scale_by_dt and manager._term_names[-1]==TERM
            compute = manager.compute
            raw, rows = None, []
            begin = monotonic()

            def capture(dt):
                nonlocal raw
                total = compute(dt)
                z,feet,head = inputs(env)
                if env.common_step_counter in spec['reward_audit_controls']:
                    expected_feet,expected_head = native_contact(env)
                    torch.testing.assert_close(feet,expected_feet,atol=0,rtol=0)
                    torch.testing.assert_close(head,expected_head,atol=0,rtol=0)
                assert raw is None and dt==.02
                raw = dict(control_reward=cpu(manager._step_reward[:,:-1].sum(dim=1)*dt),
                           candidate_reward=cpu(total),support=cpu(manager._step_reward[:,-1]),
                           gravity_z=cpu(z),feet=cpu(feet),head=cpu(head),qpos_env0=cpu(env.sim.data.qpos[0]))
                return total

            with torch.no_grad(), patch.object(manager,'compute',capture):
                for k in range(1,401):
                    age = env.episode_length_buf
                    assert (age==k-1).all()
                    latch.observe(obs['actor'],age)
                    features = torch.cat((obs['critic'],latch.supine.float()[:,None]),dim=1)
                    gravity = cpu(obs['actor'][:,3:6])
                    action = actor(obs,stochastic_output=True)
                    obs,_,terminated,timeout,_ = env.step(action)
                    controls += 1
                    assert not terminated.any() and bool(timeout.all() if k==400 else ~timeout.any())
                    assert raw is not None and env._sim_step_counter==5*k
                    rows.append(dict(raw,critic_obs=cpu(features),actor_gravity=gravity,done=cpu(timeout)))
                    raw = None
                    if video.writer is not None:
                        video.sample()
                    if k%25==0:
                        watch.check()
                        check_project()
                        assert monotonic()-begin<spec['maximum_seconds']
                        publish('running')
                        print('EARLY_SUPPORT_VALUE_SAMPLE',k,'/ 400',flush=True)
            assert not latch.latched.any() and not env.episode_length_buf.any()
            data = {name:np.stack([row[name] for row in rows]) for name in rows[0]}
            np.savez_compressed(record/'value-data.npz',initial_qpos_env0=first_qpos,**data)
            shared_reward = (data['control_reward'].astype(float)+data['candidate_reward'].astype(float))*.5
            fitted = fit(critic,dict(critic_obs=data['critic_obs'],rewards=shared_reward),p,record,fitting)
            assert updates==50 and actor_initial==signature(actor.state_dict())
            valid = np.asarray(p['split']['validation_ids'])
            features = torch.from_numpy(data['critic_obs'][:,valid].reshape(-1,76)).to(env.device)
            with torch.no_grad():
                pred = torch.cat([critic(td('critic',part)).squeeze(-1) for part in features.split(4096)]).cpu().numpy().reshape(400,64)
            targets = {label:discounted_returns(torch.from_numpy(data[label+'_reward'][:,valid]),p['gamma']).numpy()/fitted['reward_scale']
                       for label in ('control','candidate')}
            dual = scores(pred,targets,p)
            write(record/'fit-result.json',dict(status='fit-complete-review-required',scores=dual,
                  both_value_gates_passed=all(row['gate_passed'] for row in dual.values()),mean_value_gate_passed=fitted['value_gate_passed'],
                  data_sha256=sha(record/'value-data.npz'),critic_sha256=sha(record/'critic-final.pt'),
                  actor_updates=0,critic_updates=updates,actor_unchanged=True,gpu=watch.summary(),
                  physical_substeps=env._sim_step_counter,transitions=controls*env.num_envs))
            status = 'collected-review-required'
        finally:
            if video is not None:
                video.close()
            if snapshot is not None:
                snapshot.close()
            if env is not None:
                env.close()
            write(record/'gpu-samples.json',watch.rows)
            publish(status)


if __name__ == '__main__':
    check_project()
    assert sys.argv[1:] in (['--freeze'],['--run'])
    (freeze if sys.argv[1]=='--freeze' else run)()
