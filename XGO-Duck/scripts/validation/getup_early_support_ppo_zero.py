"""Freeze and compare both actual 3072-environment CUDA starts without stepping."""
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from run_getup_early_support_value import OUT as VALUE_OUT, PLAN as VALUE_PLAN, SCENE, checked as checked_value
from check_getup_early_support_clock import TERM, config, cpu, startup
from getup_adaptive_ppo import AdaptiveGetupPPO
from getup_early_support_ppo_interface import SupportCriticObservations, paired_reward
from getup_release_fork_state import signature
from getup_value_bootstrap import networks, td
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_early_support_ppo_zero.json'
DRAFT = ROOT/'configs/experiments/getup_early_support_ppo_draft.json'
OUT = ROOT/'records/reports/getup-early-support-ppo-zero'
STAGE = 'S3-early-support-ppo-zero-development'
LOCAL = ROOT/'configs/experiments/local_getup_ppo.json'


def freeze():
    from mjlab.scene import Scene
    from common_scene_contract import compare_arrays, model_arrays

    assert not PLAN.exists() and not OUT.exists()
    spec = checked_value()
    draft, local = read(DRAFT), read(LOCAL)
    proof = read(VALUE_OUT/'analysis.json')
    cpu_proof = read(VALUE_OUT/'ppo-interface-cpu.json')
    interface = ROOT/'scripts/training/getup_early_support_ppo_interface.py'
    assert cpu_proof['status']=='passed-cpu-adapter' and cpu_proof['adapter_sha256']==sha(interface)
    assert proof['both_value_gates_passed'] and proof['critic_sha256']==draft['critic_sha256']
    assert cpu_proof['critic_sha256']==proof['critic_sha256']
    p = deepcopy(spec['environment'])
    parent = read(ROOT/local['parent_protocol'])['training']['candidate']
    n, seed = draft['paired_development']['environments_per_arm'], 1083
    assert n==local['arms']['throughput']['num_envs']==3072
    assert not any(f'_s{seed}_' in f.name for f in (ROOT/'records/experiments').iterdir())
    assert sha(ROOT/parent['model'])==p['model_sha256']
    for key in ('anchor_data','anchor_cells','anchor_weight'):
        assert p[key]==parent[key]
    for key in ('retention_data','retention_cells','retention_weight','retention_seed','retention_samples_per_cell'):
        p[key] = parent[key]
    p.update(seed=seed,num_envs=n,focus_weight=1.,reward_scale=proof['reward_scale'])
    p['split'] = dict(pose_codes=[0,1,2,3]*(n//4))
    p['learner']['minibatches'] = local['arms']['throughput']['minibatches']
    p['learner']['minibatch_seed'] = 108303
    p['anchor_seed'] = 108304
    p['retention_seed'] = 108305
    assert n*24//p['learner']['minibatches']==3072
    paths = [VALUE_PLAN,DRAFT,LOCAL,VALUE_OUT/'analysis.json',VALUE_OUT/'ppo-interface-cpu.json',
        VALUE_OUT/'readback-recovery.json',ROOT/draft['critic_checkpoint'],SCENE,interface,Path(__file__).resolve(),
        ROOT/p['retention_data'],ROOT/p['retention_cells'],ROOT/p['anchor_data'],ROOT/p['anchor_cells']]
    paths.extend(ROOT/path for path in local['source_sha256'] if Path(path).suffix=='.py')
    result = dict(status='frozen-zero-step-only',stage=STAGE,environment=p,
        memory_limit=draft['paired_development']['memory_fraction_limit'],arm_timeout_seconds=150,
        protocol_sha256=sha(VALUE_PLAN),reward=spec['reward'],release_age=spec['release_age'],alignment=spec['alignment'],
        critic_checkpoint=draft['critic_checkpoint'],critic_sha256=proof['critic_sha256'],
        controls=0,learning_updates=0,hardware_io=False,heldout_used=False,
        adaptive_settings=local['adaptive_settings'],step_scales=local['step_scales'],limits_rad=local['limits_rad'],
        sources={path.relative_to(ROOT).as_posix():sha(path) for path in paths})
    cfg,_ = config(result)
    with np.load(SCENE) as expected:
        compare_arrays(model_arrays(Scene(cfg.env.scene,device='cpu').compile()),{k:expected[k] for k in expected.files})
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN,result)
    write(OUT/'preflight.json',dict(status='passed-cpu-scene',protocol_sha256=sha(PLAN),physical_substeps=0,
        learning_updates=0,actor_width=61,critic_width=76,environments=n,seed=seed))
    print('EARLY_SUPPORT_PPO_ZERO_FROZEN',sha(PLAN),flush=True)


def checked():
    spec = read(PLAN)
    assert spec['status']=='frozen-zero-step-only' and spec['controls']==spec['learning_updates']==0
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(PLAN)
    checked_value()
    assert all(sha(ROOT/path)==digest for path,digest in spec['sources'].items())
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage']==STAGE and stage['current_protocol_frozen'] and ROOT/stage['current_protocol']==PLAN
    check_project()
    return spec


def initialize(spec, enabled, record, check_active):
    from mjlab.envs import ManagerBasedRlEnv

    p = spec['environment']
    cfg, _ = config(spec)
    cfg.env.rewards[TERM].func = paired_reward
    cfg.env.rewards[TERM].params = dict(spec['reward'],enabled=enabled)
    env = ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None)
    try:
        adapter = SupportCriticObservations(env,spec['release_age'],spec['alignment'])
        observations,_ = env.reset(seed=p['seed'])
        actor,critic = networks(76,p,env.device)
        startup(env,observations,actor,p,record,SCENE)
        features = adapter.before_action(observations)
        env.step = adapter.step_wrapper(env.step)
        assert sha(ROOT/spec['critic_checkpoint'])==spec['critic_sha256']
        warm = torch.load(ROOT/spec['critic_checkpoint'],map_location='cpu',weights_only=True)
        assert warm['reward_scale']==p['reward_scale']
        critic.load_state_dict(warm['critic_state_dict'],strict=True)
        with np.load(ROOT/p['anchor_data']) as source:
            anchors = source['observations'].copy(),source['teacher_targets'].copy()
        with np.load(ROOT/p['retention_data']) as source:
            replay = source['observations'].copy(),source['targets'].copy(),read(ROOT/p['retention_cells'])
        learner = AdaptiveGetupPPO(actor,critic,warm['critic_optimizer_state_dict'],p['learner'],
            anchor_inputs=anchors[0],anchor_targets=anchors[1],anchor_cells=read(ROOT/p['anchor_cells']),
            anchor_weight=p['anchor_weight'],anchor_seed=p['anchor_seed'],focus_weight=p['focus_weight'],
            retention_replay=replay,retention_weight=p['retention_weight'],retention_seed=p['retention_seed'],
            retention_samples_per_cell=p['retention_samples_per_cell'],adaptive_settings=spec['adaptive_settings'],
            reward_scale=p['reward_scale'],check_active=check_active,replay_limits=spec['limits_rad'],step_scales=spec['step_scales'])
        assert not learner.actor_optimizer.state and {int(row['step']) for row in learner.critic_optimizer.state.values()}=={50}
        assert learner._within_limits(learner.replay_drift())
        assert env._sim_step_counter==env.common_step_counter==0
        assert not adapter.latch.latched.any() and not features['critic'][:,75].any()
        term = env.reward_manager._term_cfgs[-1]
        assert env.reward_manager._term_names[-1]==TERM and term.params['enabled'] is enabled and term.weight==1.
        with torch.no_grad():
            critic_output = critic(features)
        assert critic_output.isfinite().all()
        return env,features,learner,critic_output
    except BaseException:
        env.close()
        raise


def zero(spec, arm):
    from mjlab.utils.torch import configure_torch_backends

    enabled = arm=='candidate'
    record = OUT/arm
    record.mkdir()
    began = perf_counter()
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    watch = MemoryWatch(spec['memory_limit'])
    with watch.active():
        env,features,learner,prediction = initialize(spec,enabled,record,watch.check)
        try:
            torch.cuda.synchronize()
            watch.check()
            watch.sample()
            raw = env.action_manager.get_term('joint_pos')
            state = dict(observations={k:v for k,v in features.items()},critic_output=prediction,
                actor=learner.actor.state_dict(),critic=learner.critic.state_dict(),
                actor_adam=learner.actor_optimizer.state_dict(),critic_adam=learner.critic_optimizer.state_dict(),
                replay_rng=learner.generator.get_state(),anchor_rng=learner.anchor_generator.get_state(),
                qpos=env.sim.data.qpos,qvel=env.sim.data.qvel,control=env.sim.data.ctrl,ages=env.episode_length_buf,
                action_history=raw._raw_actions,applied=raw._processed_actions,initial_imu=env._imu_misalign_quat,
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(env.device),
                expanded={name:getattr(env.sim.model,name) for name in sorted(env.sim.expanded_fields)})
            write(record/'matched-state.json',signature(state))
            write(record/'result.json',dict(status='passed-zero-step',enabled=enabled,environments=env.num_envs,
                physical_substeps=env._sim_step_counter,actor_updates=learner.actor_steps,critic_updates=learner.critic_steps,
                peak_gpu=watch.summary(),wall_seconds=perf_counter()-began,protocol_sha256=sha(PLAN)))
        finally:
            env.close()
    print('EARLY_SUPPORT_PPO_ZERO_ARM',arm,watch.summary()['peak_mib'],flush=True)


def main():
    if sys.argv[1:]==['--freeze']:
        return freeze()
    spec = checked()
    if len(sys.argv)>1:
        assert len(sys.argv)==2 and sys.argv[1] in ('control','candidate')
        return zero(spec,sys.argv[1])
    assert os.isatty(1)
    runs = {}
    for arm in ('control','candidate'):
        check_project()
        argv = [sys.executable,'-B',str(Path(__file__).resolve()),arm]
        with (OUT/f'{arm}.log').open('x') as log:
            try:
                status = subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                        timeout=spec['arm_timeout_seconds']).returncode
            except subprocess.TimeoutExpired:
                status = 124
        runs[arm] = dict(argv=argv,exit_code=status)
        if status:
            write(OUT/'analysis.json',dict(status='stopped-review-required',runs=runs))
            raise SystemExit(status)
        runs[arm]['result'] = read(OUT/arm/'result.json')
    control,candidate = [read(OUT/arm/'matched-state.json') for arm in ('control','candidate')]
    mismatched = sorted(key for key in control if control[key]!=candidate[key])
    assert control.keys()==candidate.keys()
    write(OUT/'analysis.json',dict(status='passed-actual-matched-cuda-zero' if not mismatched else 'stopped-review-required',
        mismatched=mismatched,runs=runs,physical_substeps=0,learning_updates=0,heldout_used=False,
        protocol_sha256=sha(PLAN),common_state_signatures={arm:sha(OUT/arm/'matched-state.json') for arm in runs}))
    print('EARLY_SUPPORT_PPO_ZERO_PAIR',mismatched,flush=True)
    if mismatched:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
