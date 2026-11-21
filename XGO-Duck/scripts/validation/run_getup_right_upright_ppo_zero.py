"""Freeze the matched right-side PPO design and compare actual CUDA zero steps."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import torch

import check_getup_right_upright_native_contract as native
import prepare_getup_right_upright_retention as replay
import run_getup_right_upright_value_coverage as value
from getup_adaptive_ppo import AdaptiveGetupPPO
from getup_anchor_runtime import startup
from getup_release_fork_state import signature
from getup_right_upright_ppo_bridge import RightUprightPPOBridge
from getup_right_upright_progress import RightUprightProgress
from getup_value_bootstrap import networks
from record_io import write
from run_getup_early_support_ppo import initial_signature
from run_local_getup_ppo import MemoryWatch, check_project
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_right_upright_ppo_zero.json'
DRAFT = ROOT/'configs/experiments/getup_right_upright_ppo_draft.json'
OUT = ROOT/'records/reports/getup-right-upright-ppo-zero'
LOCAL = ROOT/'configs/experiments/local_getup_ppo.json'
PRIOR = ROOT/'configs/experiments/getup_side_to_supine_ppo.json'
STAGE = 'S3-right-side-upright-ppo-zero-development'
PREP = 'S3-right-side-upright-ppo-preparation'
ROLES = ('control','candidate')
SEED = 1101


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def freeze():
    from mjlab.scene import Scene
    from common_scene_contract import compare_arrays, model_arrays

    check_project()
    assert read(ROOT/'project.json')['staged_training_request']['current_stage'] == PREP
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    shared, draft, local, prior = value.checked(), read(DRAFT), read(LOCAL), read(PRIOR)
    result = value.OUT/'analysis.json'
    proof = read(result)
    bridge_file = ROOT/draft['required_matched_bridge']
    bridge = read(bridge_file)
    replay_file = replay.OUT/'analysis.json'
    retained = read(replay_file)
    assert proof['protocol_sha256'] == sha(value.PLAN)
    assert proof['both_value_gates_passed'] and proof['critic_updates'] == 50
    assert not any(proof[key] for key in ('heldout_used','release_mismatches','actor_updates'))
    assert bridge['status'] == 'passed-matched-cpu-ppo-bridge' and bridge['nonzero_candidate_samples'] > 0
    assert bridge['critic_features_bitwise_equal'] and bridge['control_contribution_exactly_zero']
    assert all(sha(ROOT/path) == digest for path, digest in bridge['source_sha256'].items())
    assert retained['status'] == 'passed-four-pose-parent-replay' and not retained['heldout_used']
    assert retained['rows'] == 900 and retained['cells'] == 24
    record = ROOT/proof['record']
    warm, windows = record/'critic-final.pt', record/'windows-view-verification.json'
    observation = read(windows)
    assert observation['status'] == 'passed' and observation['snapshots_with_complete_clip'] > 0
    assert sha(warm) == proof['critic_sha256']
    n = draft['proposed_controls']['num_envs_per_arm']
    assert n == local['arms']['throughput']['num_envs'] == 3072
    assert not any(f'_s{SEED}_' in path.name for path in (ROOT/'records/experiments').iterdir())
    p = deepcopy(shared['environment'])
    old = prior['environment']
    assert p['model_sha256'] == retained['model_sha256'] == old['model_sha256'] == sha(ROOT/p['model'])
    for key in ('anchor_data','anchor_cells','anchor_weight'):
        assert p[key] == old[key]
    p.update(seed=SEED, num_envs=n, focus_weight=1., anchor_seed=110104, retention_seed=110105,
             reward_scale=proof['reward_scale'], retention_data=relative(replay.OUT/'parent-replay.npz'),
             retention_cells=relative(replay.OUT/'sampling-cells.json'), retention_weight=old['retention_weight'],
             retention_samples_per_cell=old['retention_samples_per_cell'])
    p['split'] = dict(pose_codes=[0,1,2,3]*(n//4))
    p['learner'] = dict(old['learner'], minibatch_seed=110106)
    assert p['learner']['initial_std'] == p['initial_std'] == .1
    budget = draft['proposed_controls']
    assert n*budget['steps_per_update']//p['learner']['minibatches'] == 3072
    assert n*budget['steps_per_update']*budget['maximum_updates_per_arm'] == budget['maximum_transitions_per_arm']
    assert sha(ROOT/p['retention_data']) == retained['replay_sha256']
    assert sha(ROOT/p['retention_cells']) == retained['cells_sha256']
    pose_reference = ROOT/draft['pose_gate_reference']
    originals = read(pose_reference)['environment']
    gates = dict(seed=1102, worlds=320, criteria=originals['criteria'],
                 success_rate_min=originals['success_rate_min'], actual_release_minimum=originals['actual_release_minimum'],
                 applied_joint_limit_tolerance_deg=originals['applied_joint_limit_tolerance_deg'],
                 actual_right_net_gain_over_control_min=4, actual_right_net_gain_over_parent_min=0,
                 other_actual_poses_net_gain_over_control_min=0, other_actual_poses_net_gain_over_parent_min=0,
                 aggregate_net_gain_over_parent_min=0, final_holdout_used=False)
    paths = [value.PLAN, DRAFT, LOCAL, PRIOR, result, bridge_file, warm, windows, replay_file, pose_reference,
             native.SCENE, Path(__file__), Path(replay.__file__),
             ROOT/p['retention_data'], ROOT/p['retention_cells'], ROOT/p['anchor_data'], ROOT/p['anchor_cells']]
    paths.extend(ROOT/path for path in local['source_sha256'] if Path(path).suffix == '.py')
    sources = dict(shared['sources'])
    sources.update(bridge['source_sha256'])
    sources.update({relative(path): sha(path) for path in paths})
    spec = dict(status='frozen-zero-step-only', stage=STAGE, environment=p, training_budget=budget,
                memory_limit=budget['maximum_observed_gpu_memory_fraction'], arm_timeout_seconds=150,
                controls=0, learning_updates=0, hardware_io=False, heldout_used=False, automatic_retry=False,
                release_age=shared['release_age'], alignment=shared['alignment'], entry=shared['entry'],
                horizon=shared['horizon'], reward_coefficient=shared['reward_coefficient'],
                critic_checkpoint=relative(warm), critic_sha256=proof['critic_sha256'], evaluation_gates=gates,
                adaptive_settings=local['adaptive_settings'], step_scales=local['step_scales'],
                limits_rad=local['limits_rad'], observation_reference=relative(windows), sources=sources)
    cfg,_ = native.configuration(p)
    with np.load(native.SCENE) as arrays:
        compare_arrays(model_arrays(Scene(cfg.env.scene,device='cpu').compile()),
                       {key: arrays[key] for key in arrays.files})
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='passed-cpu-scene', protocol_sha256=sha(PLAN),
          physical_substeps=0, learning_updates=0, actor_width=61, critic_width=90, environments=n, seed=SEED))
    print('RIGHT_UPRIGHT_PPO_ZERO_FROZEN',sha(PLAN),flush=True)


def checked():
    spec = read(PLAN)
    assert spec['status'] == 'frozen-zero-step-only' and spec['controls'] == spec['learning_updates'] == 0
    assert read(OUT/'preflight.json')['protocol_sha256'] == sha(PLAN)
    value.checked()
    for path,digest in spec['sources'].items():
        assert sha(ROOT/path) == digest, path
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage'] == STAGE and stage['current_protocol_frozen'] and ROOT/stage['current_protocol'] == PLAN
    check_project()
    return spec


def initialize(spec,enabled,record,check_active):
    from mjlab.envs import ManagerBasedRlEnv

    p = spec['environment']
    cfg,_ = native.configuration(p)
    cfg.env.rewards[native.TERM].weight = spec['reward_coefficient']
    env = ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None)
    try:
        bridge = RightUprightPPOBridge(env,spec,enabled)
        obs,_ = env.reset(seed=p['seed'])
        actor,critic = networks(90,p,env.device)
        startup(env,obs,actor,p,record,native.SCENE)
        features = bridge.before_action(obs)
        assert features['actor'] is obs['actor']
        torch.testing.assert_close(features['critic'][:,:75],obs['critic'],rtol=0,atol=0)
        env.step = bridge.step_wrapper(env.step)
        bridge.term.func = bridge.reward
        assert sha(ROOT/spec['critic_checkpoint']) == spec['critic_sha256']
        warm = torch.load(ROOT/spec['critic_checkpoint'],map_location='cpu',weights_only=True)
        assert warm['reward_scale'] == p['reward_scale']
        critic.load_state_dict(warm['critic_state_dict'],strict=True)
        with np.load(ROOT/p['anchor_data']) as anchors, np.load(ROOT/p['retention_data']) as retained:
            learner = AdaptiveGetupPPO(actor,critic,warm['critic_optimizer_state_dict'],p['learner'],
                anchor_inputs=anchors['observations'],anchor_targets=anchors['teacher_targets'],
                anchor_cells=read(ROOT/p['anchor_cells']),anchor_weight=p['anchor_weight'],
                anchor_seed=p['anchor_seed'],focus_weight=p['focus_weight'],
                retention_replay=(retained['observations'],retained['targets'],read(ROOT/p['retention_cells'])),
                retention_weight=p['retention_weight'],retention_seed=p['retention_seed'],
                retention_samples_per_cell=p['retention_samples_per_cell'],adaptive_settings=spec['adaptive_settings'],
                reward_scale=p['reward_scale'],check_active=check_active,replay_limits=spec['limits_rad'],
                step_scales=spec['step_scales'])
        assert not learner.actor_optimizer.state
        assert {int(row['step']) for row in learner.critic_optimizer.state.values()} == {50}
        assert learner._within_limits(learner.replay_drift())
        assert env._sim_step_counter == env.common_step_counter == 0
        assert not bridge.state.release_known.any() and not bridge.state.supported.any()
        with torch.no_grad():
            prediction = critic(features)
        assert prediction.isfinite().all()
        return env,features,learner,prediction,bridge
    except BaseException:
        env.close()
        raise


def reward_probe(spec,env,features,bridge):
    """Use the actual registered term on an isolated fixture, leaving live state untouched."""
    live = bridge.state
    probe = RightUprightProgress(env.num_envs,env.device,spec['release_age'],spec['alignment'],
                                 spec['entry'],spec['horizon'])
    actor = features['actor'].clone()
    actor[:3,3:6] = actor.new_tensor([[0.,-1.,0.],[0.,-1.,0.],[0.,1.,0.]])
    ages = torch.zeros_like(env.episode_length_buf)
    ages[:3] = spec['release_age']
    tilt = actor.new_full((env.num_envs,),30.)
    feet = torch.zeros((env.num_envs,2),device=env.device,dtype=torch.bool)
    feet[:3] = True
    heads = feet[:,0].clone()
    with patch.object(env,'_right_upright_progress',probe), patch.object(env,'episode_length_buf',ages), +         patch.object(native,'inputs',lambda _: (-(tilt*(math.pi/180.)).cos(),feet,heads)):
        probe.observe_release(actor,ages)
        assert probe.right[:3].tolist() == [True,True,False]
        for _ in range(len(probe.tilts)):
            ages[:3] += 1
            assert not bridge.term.func(env).any()
        assert probe.trigger_age[:3].tolist() == [int(ages[0]),int(ages[0]),-1]
        ages[:3] += 1
        tilt[:3] = tilt.new_tensor([24.,36.,24.])
        raw = bridge.term.func(env)
        expected = [.2,-.2,0.] if bridge.enabled else [0.,0.,0.]
        torch.testing.assert_close(raw[:3],raw.new_tensor(expected),rtol=0,atol=1e-6 if bridge.enabled else 0)
        assert not raw[3:].any()
        native.reset_reward(env,torch.tensor([0,2],device=env.device))
        assert probe.release_known[:3].tolist() == [False,True,False]
        assert not probe.supported[:,[0,2]].any() and probe.trigger_age[[0,2]].tolist() == [-1,-1]
    assert env._right_upright_progress is live and not live.release_known.any()
    return dict(raw=raw[:3].tolist(), weighted=(raw[:3]*bridge.term.weight).tolist(),
                clock=signature(vars(probe)), reset_ids=[0,2])


def matched_signature(env,features,learner,prediction,bridge):
    return dict(initial_signature(env,features,learner,prediction),
                right_clock=signature(vars(bridge.state)), replay_inputs=signature(learner.retention_inputs),
                replay_targets=signature(learner.retention_targets), replay_cells=signature(learner.retention_cells),
                anchor_inputs=signature(learner.anchor_inputs), anchor_targets=signature(learner.anchor_targets))


def arm(spec,role):
    from mjlab.utils.torch import configure_torch_backends

    record = OUT/role
    record.mkdir()
    enabled = role == 'candidate'
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    watch = MemoryWatch(spec['memory_limit'])
    with watch.active():
        env,features,learner,prediction,bridge = initialize(spec,enabled,record,watch.check)
        try:
            matched = matched_signature(env,features,learner,prediction,bridge)
            probe = reward_probe(spec,env,features,bridge)
            assert matched == matched_signature(env,features,learner,prediction,bridge)
            torch.cuda.synchronize()
            watch.check()
            watch.sample()
            write(record/'matched-state.json',matched)
            write(record/'result.json',dict(status='passed-zero-step',environments=env.num_envs,enabled=enabled,
                  physical_substeps=env._sim_step_counter,actor_updates=learner.actor_steps,critic_updates=learner.critic_steps,
                  peak_gpu=watch.summary(),reward_probe=probe,protocol_sha256=sha(PLAN)))
        finally:
            env.close()
    print('RIGHT_UPRIGHT_PPO_ZERO_ARM',role,watch.summary()['peak_mib'],flush=True)


def pair(spec):
    runs = {}
    for role in ROLES:
        check_project()
        argv = [sys.executable,'-B',str(Path(__file__)),role]
        with (OUT/f'{role}.log').open('x') as log:
            try:
                status = subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                        timeout=spec['arm_timeout_seconds']).returncode
            except subprocess.TimeoutExpired:
                status = 124
        runs[role] = dict(argv=argv,exit_code=status)
        if status:
            write(OUT/'analysis.json',dict(status='stopped-review-required',runs=runs,protocol_sha256=sha(PLAN)))
            raise SystemExit(status)
        runs[role]['result'] = read(OUT/role/'result.json')
    control,candidate = [read(OUT/role/'matched-state.json') for role in ROLES]
    assert control.keys() == candidate.keys()
    mismatched = sorted(key for key in control if control[key] != candidate[key])
    if runs['control']['result']['reward_probe']['clock'] != runs['candidate']['result']['reward_probe']['clock']:
        mismatched.append('reward-probe-clock')
    write(OUT/'analysis.json',dict(status='passed-actual-matched-cuda-zero' if not mismatched else 'stopped-review-required',
          mismatched=mismatched,runs=runs,physical_substeps=0,learning_updates=0,heldout_used=False,
          protocol_sha256=sha(PLAN),common_state_signatures={role:sha(OUT/role/'matched-state.json') for role in ROLES}))
    print('RIGHT_UPRIGHT_PPO_ZERO_PAIR',mismatched,flush=True)
    if mismatched:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','pair',*ROLES))
    command = parser.parse_args().command
    if command == 'freeze':
        freeze()
    elif command == 'pair':
        pair(checked())
    else:
        arm(checked(),command)
