"""Matched local CUDA adaptation to the early upright band."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from check_getup_early_support_clock import config as original_config
from getup_adaptive_ppo import AdaptiveGetupPPO
from getup_anchor_runtime import startup
from getup_early_support_ppo_interface import SupportCriticObservations
from getup_leg_head_training import restrict_actor_to_leg_output
from getup_upright_band_reward import upright_band_increment
from getup_value_bootstrap import networks, td
from review_nominal_skill_composition_handoff import read
from run_getup_early_support_ppo import initial_signature
from run_nominal_supine_early_load import PROJECT, REPORT, ROOT, parent, rel, sha, write
from verify_getup_late_feasibility import exact
from getup_teacher_anchor_train import prepare, save_checkpoint
from inspect_delivery_checkpoint_loader import mlp_model
from monitored_ppo_authorization import AuthorizationMonitor
from run_local_getup_ppo import MemoryWatch
import getup_ppo_live as live
import native_ppo_observation as native
import run_getup_supine_training as rollout


KEY = 'nominal_supine_upright_band_training'
OUT = REPORT/'support-progress'/'training'
PLAN = ROOT/'configs/experiments/nominal_supine_upright_band_training.json'
ORIGINAL = ROOT/'configs/experiments/getup_latched_head_corrective_ppo_signature.json'
ORIGINAL_SHA = '6b9781f3277eb2f15bda189fb58510477b988a37a27a5ea6ed29ec259600cd4f'
ENTRY = Path(__file__).resolve()
TERM = 'bounded_upright_transfer'
ROLES = ('control','candidate')


def sources_checked(plan):
    for path, checksum in plan['integrity'].items():
        assert sha(ROOT/path) == checksum,path


def select(project, stage, execution, evidence):
    info = dict(current_protocol_execution=execution,current_evidence=rel(evidence))
    project.update(info)
    project['staged_training_request'].update(info,current_stage=stage,current_protocol=rel(PLAN),
                                            current_protocol_frozen=True)


def context(status):
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    plan = read(PLAN,node['protocol_sha256'])
    stage = plan['training_stage']
    assert parent.allowed(project,stage) and node['status']==status
    assert not project['staged_training_request']['simulation_running']
    sources_checked(plan)
    return project,node,plan


def retention_labels(source, warm, path):
    # The replay inputs and strata are unchanged. The targets come from
    # the exact starting actor used by both arms.
    actor = mlp_model('actor',warm,14)
    p = source['environment']
    with np.load(ROOT/p['anchor_data'],allow_pickle=False) as data:
        anchors = data['observations'].copy()
    with np.load(ROOT/p['retention_data'],allow_pickle=False) as data:
        fixed = data['observations'].copy()
    assert anchors.shape[1]==fixed.shape[1]==61 and len(fixed)==900
    with torch.inference_mode():
        a = actor(td('actor',torch.from_numpy(anchors))).numpy()
        b = actor(td('actor',torch.from_numpy(fixed))).numpy()
    np.savez_compressed(path, anchors=anchors, anchor_actions=a, retention=fixed, retention_actions=b)


def freeze():
    from getup_early_support_ppo_zero import SCENE

    project = read(PROJECT)
    goal = project['goal_execution']
    node = goal[KEY]
    draft = ROOT/node['design']
    proposed = read(draft)
    assert node['status']=='preparation-ready' and parent.allowed(project,proposed['preparation_stage'])
    assert not project['staged_training_request']['simulation_running'] and not OUT.exists() and not PLAN.exists()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    band_path = ROOT/goal['nominal_supine_upright_band']['preflight']
    band = read(band_path,proposed['archive_qualification_sha256'])
    assert band['qualifies_cuda_start']
    model_path = ROOT/goal['nominal_supine_support_progress']['preflight']
    model = read(model_path,band['parent_preflight_sha256'])['parent']
    old = read(ORIGINAL,ORIGINAL_SHA)
    p = deepcopy(old['environment'])
    assert p['num_envs']==proposed['environments']==3072 and p['reward_scale']==model['frozen_reward_scale']
    assert old['limit']['rollout_controls']==proposed['steps_per_update']==24
    assert old['memory_limit']==proposed['gpu_memory_fraction_limit'] and old['updates']==proposed['updates_per_arm']
    seed = proposed['training_seed']
    assert seed not in proposed['reserved_confirmations'] and not any(
        f'-s{seed}-' in path.name or f'_s{seed}_' in path.name for path in (ROOT/'records/experiments').iterdir())
    checkpoint = ROOT/model['checkpoint']
    assert sha(checkpoint)==model['checkpoint_sha256'] and sha(ROOT/model['onnx'])==model['onnx_sha256']
    warm = torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert warm['completed_updates']==model['completed_updates']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    runs = {role:f'duck-getup-upright-band-{role}-s{seed}-{stamp}' for role in ROLES}
    OUT.mkdir()
    targets = OUT/'parent-replay.npz'
    retention_labels(old,warm,targets)
    p.update(model=model['onnx'],model_sha256=model['onnx_sha256'],seed=seed,
             anchor_seed=seed*100+4,retention_seed=seed*100+5)
    p['learner']['minibatch_seed'] = seed*100+3
    assert p['split']['pose_codes']==[0,1,2,3]*(p['num_envs']//4)
    plan = dict(proposed,environment=p,release_age=old['release_age'],alignment=old['alignment'],reward=old['reward'],
        adaptive_settings=old['adaptive_settings'],step_scales=old['step_scales'],limits_rad=old['limits_rad'],
        term=band['term'],models=model,joint_order=model['joint_order'],runs=runs,
        limit=dict(old['limit'],wall_seconds_per_arm=proposed['wall_seconds_per_arm']),
        replay=rel(targets),cpu_scene=rel(SCENE))
    root_paths = (draft,ORIGINAL,band_path,model_path,checkpoint,ROOT/model['onnx'],ENTRY,
        SCENE,targets,*(ROOT/p[k] for k in ('anchor_data','anchor_cells','retention_data','retention_cells','probe_dataset')))
    modules = (
        'training/getup_upright_band_reward.py','training/getup_early_support_ppo_interface.py',
        'training/getup_adaptive_ppo.py','training/getup_success_step_control.py',
        'training/getup_masked_ppo.py','training/getup_leg_head_training.py','training/getup_early_support_reward.py',
        'training/getup_recovery_retention_ppo.py','training/run_getup_supine_training.py',
        'training/getup_ppo_live.py','training/native_ppo_observation.py','training/native_ppo_viewer.py',
        'training/native_roll_observation.py','training/monitored_ppo_authorization.py','training/getup_anchor_runtime.py',
        'training/getup_teacher_anchor_train.py','validation/check_getup_early_support_clock.py',
        'validation/run_getup_early_support_ppo.py','validation/getup_early_support_ppo_zero.py',
        'validation/inspect_delivery_checkpoint_loader.py','validation/verify_getup_late_feasibility.py')
    plan['integrity'] = {rel(path):sha(path) for path in (*root_paths,*(ROOT/'scripts'/name for name in modules))}
    plan['observer_preflight'] = parent.preflight()
    write(PLAN,plan)
    receipt = OUT/'frozen-preflight.json'
    write(receipt,dict(protocol_sha256=sha(PLAN),archive_sha256=sha(band_path),parent_sha256=sha(checkpoint),
                       targets_sha256=sha(targets),observer=plan['observer_preflight'],learning_updates=0,physical_substeps=0))
    node.update(status='cuda-zero-ready',design_sha256=sha(draft),protocol=rel(PLAN),
                protocol_sha256=sha(PLAN),runs=runs,preflight=rel(receipt))
    select(project,plan['training_stage'],'nominal-supine-upright-band-zero-ready',receipt)
    write(PROJECT,project)
    print('UPRIGHT_BAND_TRAINING_FROZEN',sha(PLAN),runs,flush=True)


def band_rate(env, enabled, **parameters):
    """Convert signed credit to a rate because the manager multiplies by dt."""
    state = env._upright_band
    up = -env.scene['robot'].data.projected_gravity_b[:,2]
    raw = upright_band_increment(state['previous'],up,env.episode_length_buf-1,
                                 env._early_support_release.supine,**parameters)
    state['previous'].copy_(up)
    state['delta'] = raw
    return (raw if enabled else torch.zeros_like(raw))/env.step_dt


def initialize(plan, role, record, check_active):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import RewardTermCfg

    p = plan['environment']
    cfg, _ = original_config(plan)
    enabled = role=='candidate'
    cfg.env.rewards[TERM] = RewardTermCfg(func=band_rate,weight=plan['upright_credit_weight'],
                                         params={**plan['term'],'enabled':enabled})
    env = ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None)
    try:
        observations,_ = env.reset(seed=p['seed'])
        features = SupportCriticObservations(env,plan['release_age'],plan['alignment'])
        actor, critic = networks(76,p,env.device)
        warm = torch.load(ROOT/plan['models']['checkpoint'],map_location='cpu',weights_only=True)
        actor.load_state_dict(warm['actor_state_dict'],strict=True)
        critic.load_state_dict(warm['critic_state_dict'],strict=True)
        startup(env,observations,actor,p,record,ROOT/plan['cpu_scene'])
        with np.load(ROOT/plan['replay'],allow_pickle=False) as data:
            anchor_inputs, anchor_targets, replay, actions = [data[key] for key in (
                'anchors','anchor_actions','retention','retention_actions')]
        learner = AdaptiveGetupPPO(actor,critic,warm['critic_optimizer_state_dict'],p['learner'],
            anchor_inputs=anchor_inputs, anchor_targets=anchor_targets, anchor_cells=read(ROOT/p['anchor_cells']),
            anchor_weight=p['anchor_weight'], anchor_seed=p['anchor_seed'], focus_weight=p['focus_weight'],
            retention_replay=(replay,actions,read(ROOT/p['retention_cells'])),
            retention_weight=p['retention_weight'], retention_seed=p['retention_seed'],
            retention_samples_per_cell=p['retention_samples_per_cell'], adaptive_settings=plan['adaptive_settings'],
            reward_scale=p['reward_scale'], check_active=check_active, replay_limits=plan['limits_rad'],
            step_scales=plan['step_scales'])
        learner.actor_optimizer.load_state_dict(warm['actor_optimizer_state_dict'])
        exact(warm['actor_optimizer_state_dict'],learner.actor_optimizer.state_dict())
        exact(warm['critic_optimizer_state_dict'],learner.critic_optimizer.state_dict())
        layer, rows = restrict_actor_to_leg_output(actor,plan['joint_order'])
        # Restored Adam moments must not move the four non-leg outputs
        # despite their gradient mask.
        for parameter in (layer.weight,layer.bias):
            state = learner.actor_optimizer.state[parameter]
            for key in ('exp_avg','exp_avg_sq'):
                assert not state[key][~rows].any(),key
        assert all(g['weight_decay']==0 for g in learner.actor_optimizer.param_groups)
        assert max(learner.replay_drift())<2e-5
        state = features.before_action(observations)
        env.step = features.step_wrapper(env.step)
        up = -env.scene['robot'].data.projected_gravity_b[:,2]
        env._upright_band = dict(previous=up.clone(),delta=torch.zeros_like(up))
        assert env.common_step_counter==env._sim_step_counter==0 and not features.latch.latched.any()
        with torch.no_grad():
            value = critic(state)
        assert torch.isfinite(value).all()
        receipt = initial_signature(env,state,learner,value)
        # JSON is also the exact representation saved on the zero-step route.
        receipt = json.loads(json.dumps(receipt,allow_nan=False))
        write(record/'matched-state.json',receipt)
        return env,state,learner
    except BaseException:
        env.close()
        raise


def zero(role):
    from mjlab.utils.torch import configure_torch_backends

    _,_,plan = context('cuda-zero-ready')
    record = OUT/'cuda-zero'/role
    record.mkdir(parents=True)
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    watch = MemoryWatch(plan['gpu_memory_fraction_limit'])
    with watch.active():
        env,inputs,learner = initialize(plan,role,record,watch.check)
        try:
            assert env.reward_manager._term_names[-1]==TERM and len(inputs['actor'])==plan['environments']
            watch.sample()
            proof = dict(role=role,physical_substeps=0,learning_updates=0,memory=watch.summary(),
                         parent_adam_loaded=True,matched_state_sha256=sha(record/'matched-state.json'),
                         actor_width=inputs['actor'].shape[1],critic_width=inputs['critic'].shape[1])
            write(record/'proof.json',proof)
            print('UPRIGHT_BAND_CUDA_ZERO',role,proof,flush=True)
        finally:
            env.close()


def match_zero():
    project,node,plan = context('cuda-zero-ready')
    results = {}
    for role in ROLES:
        here = OUT/'cuda-zero'/role
        proof = read(here/'proof.json')
        assert proof['physical_substeps']==proof['learning_updates']==0 and proof['parent_adam_loaded']
        assert proof['actor_width']==61 and proof['critic_width']==76
        assert sha(here/'matched-state.json')==proof['matched_state_sha256']
        results[role] = proof
    assert len({item['matched_state_sha256'] for item in results.values()})==1
    evidence = OUT/'cuda-zero'/'matched.json'
    write(evidence,dict(protocol_sha256=node['protocol_sha256'],proof=results,physics_steps=0,learning_updates=0))
    node.update(status='paired-training-ready',zero=rel(evidence),zero_sha256=sha(evidence))
    select(project,plan['training_stage'],'nominal-supine-upright-band-matched-zero',evidence)
    write(PROJECT,project)
    print('UPRIGHT_BAND_CUDA_MATCHED',sha(evidence),flush=True)


def audit_band(env, plan, role):
    manager = env.reward_manager
    assert manager._term_names[-1]==TERM and manager._scale_by_dt
    original = manager.compute
    active = role=='candidate'
    rows = []

    def compute(dt):
        rewards = original(dt)
        control = int(env.common_step_counter)
        age = (control-1)%400+1
        first,last = plan['term']['start_age'],plan['term']['stop_age']
        if first<=age<=last+1:
            delta = env._upright_band['delta']
            rate = (delta if active else torch.zeros_like(delta))/dt
            expected = torch.nan_to_num(rate*plan['upright_credit_weight']*dt,nan=0.,posinf=0.,neginf=0.)/dt
            torch.testing.assert_close(manager._step_reward[:,-1],expected,rtol=0,atol=0)
            values = torch.stack((delta.max(),delta.min(),(delta>0).sum(),(delta<0).sum(),
                                  env._early_support_release.supine.sum())).cpu().tolist()
            rows.append(dict(control=control,age=age,active_reward=active,max_delta=values[0],min_delta=values[1],
                             positive=int(values[2]),negative=int(values[3]),supine=int(values[4])))
        return rewards

    manager.compute = compute
    return rows


class RunObserver(native.NativePPOObservation):
    def check(self, *, force=False):
        from verify_forward_support_nominal_recovery import check
        with patch.object(native,'check_child',check):
            return super().check(force=force)


def train(role):
    from mjlab.utils.torch import configure_torch_backends

    project,node,plan = context('paired-training-ready')
    admission = read(ROOT/node['zero'],node['zero_sha256'])
    if role=='candidate':
        assert node.get('results',{}).get('control',{}).get('complete') is True
    assert role not in node.get('results',{})
    assert admission['protocol_sha256']==node['protocol_sha256']
    n, updates = plan['environments'],plan['updates_per_arm']
    stage = plan['training_stage']
    run_id = plan['runs'][role]
    owner = os.getpid()
    record = ROOT/'records/experiments'/run_id
    report = OUT/role
    report.mkdir()
    project['staged_training_request'].update(learning_running=True)
    node.update(active_role=role,worker_pid=owner)
    write(PROJECT,project)

    def authorized():
        item = read(PROJECT)
        task, goal = item['staged_training_request'],item['goal_execution']
        current = goal[KEY]
        return (goal['status']=='active' and not goal['pause_requested_by_user'] and not task['pause_requested']
                and task['current_stage']==stage and task['current_protocol']==rel(PLAN) and task['current_protocol_frozen']
                and task['learning_running'] and not task['deployment_authorized_now']
                and current.get('worker_pid')==owner and current.get('active_role')==role
                and sha(PLAN)==node['protocol_sha256'])

    watch = MemoryWatch(plan['gpu_memory_fraction_limit'])
    guard = AuthorizationMonitor(authorized)
    observer = RunObserver(run_id,report,guard.check,environments=n,planned_updates=updates,
                           purpose='仰卧起身｜原奖励' if role=='control' else '仰卧起身｜新增转正奖励')
    rows, audit = [], {}
    began = None
    status = dict(status='review-required',reason=None)
    p = dict(plan['environment'],run_id=run_id,arm=role,training_updates=updates)

    def check():
        watch.check()
        observer.require_active()

    def ready(_plan, output, _scene):
        _, display, metadata = prepare(p)
        env, inputs, learner = initialize(plan,role,output,check)
        try:
            expected = admission['proof'][role]['matched_state_sha256']
            assert sha(output/'matched-state.json')==expected
            save_checkpoint(output/'model-init.pt',learner,env,0)
            write(output/'display.json',display)
            audit.update(rows=audit_band(env,plan,role),env=env,learner=learner)
            watch.sample()
            return env,inputs,learner,metadata
        except BaseException:
            env.close()
            raise

    reference = OUT/'cuda-zero'/role

    def matched(output, expected):
        assert sha(output/'matched-state.json')==sha(expected/'matched-state.json')

    def publish(path, payload):
        nonlocal began
        if path==record/'command.json':
            payload = dict(payload,visible_tty=os.isatty(1),argv=sys.argv,pid=owner)
        write(path,payload)
        if path!=record/'state.json':
            return
        run_readme = record/'README.md'
        if not run_readme.exists():
            label = '原奖励' if role=='control' else '新增转正奖励'
            run_readme.write_text(f'# 仰卧起身配对训练 · {label}\n\n'
                '[冻结方案](../../../configs/experiments/nominal_supine_upright_band_training.json) · '
                '[逐次指标](observation.json) · [原片索引](clips.json)\n', encoding='utf-8')
        if began is None and payload['status']=='training':
            began = perf_counter()
        metric = payload.get('metrics')
        if metric and metric['update']>len(rows):
            assert metric['update']==len(rows)+1 and began is not None
            rows.append(dict(metric,net_transitions_per_second=metric['controls']*n/max(perf_counter()-began,.01)))
            if metric['update']==8:
                save_checkpoint(record/'model-8.pt',audit['learner'],audit['env'],8)
        log = rel(report/'training-command.log').removeprefix('records/')
        saved = live.publish(record,dict(payload,live_log=log),rows,n,updates,watch.summary())
        if payload['status'] in ('gpu-zero-step','training'):
            observer.publish(record,saved)

    try:
        guard.start()
        observer.start()
        configure_torch_backends(allow_tf32=False)
        torch.set_num_threads(4)
        with (watch.active(),patch.object(rollout,'initialize',ready),
              patch.object(rollout,'check_matched_start',matched),patch.object(rollout,'write',publish)):
            receipt = rollout.train(p,record,ROOT/plan['cpu_scene'],reference,node['protocol_sha256'],plan['limit'],check)
        assert receipt['status']=='collected-review-required' and len(rows)==updates
        assert any(row['positive']>0 and row['supine']>0 for row in audit['rows'])
        status = dict(status=receipt['status'],reason=None,complete=True,run_id=run_id,
                      physics_substeps=receipt['substeps'],transitions=receipt['controls']*n,updates=len(rows),
                      checkpoint_sha256=sha(record/'model-final.pt'),metrics_sha256=sha(record/'learning-metrics.jsonl'),
                      memory=watch.summary(),throughput_transitions_per_second=rows[-1]['net_transitions_per_second'])
    except BaseException as error:
        status.update(reason=repr(error),complete=False)
        raise
    finally:
        try:
            write(report/'reward-audits.json',dict(rows=audit.get('rows',[]),weight=plan['upright_credit_weight']))
            write(report/'result.json',status)
            write(report/'gpu-telemetry.json',dict(samples=watch.rows))
            observer.close(status)
        finally:
            guard.close()
            current = read(PROJECT)
            registry = current['goal_execution'][KEY]
            if registry.get('worker_pid')==owner:
                del registry['worker_pid'],registry['active_role']
                registry.setdefault('results',{})[role] = dict(status,record=rel(record),result=rel(report/'result.json'))
                if not status.get('complete'):
                    registry['status'] = 'review-required'
                elif role=='candidate':
                    registry['status'] = 'paired-evaluation-pending'
                current['staged_training_request']['learning_running'] = False
                select(current,stage,'nominal-supine-upright-band-'+registry['status'],report/'result.json')
                write(PROJECT,current)
    print('UPRIGHT_BAND_PPO_COMPLETE',role,status,flush=True)


def main():
    args = ArgumentParser(description=__doc__)
    args.add_argument('action',choices=('freeze','zero','match','train'))
    args.add_argument('role',choices=ROLES,nargs='?')
    spec = args.parse_args()
    assert (spec.role is not None)==(spec.action in ('zero','train'))
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'freeze':freeze,'zero':lambda:zero(spec.role),'match':match_zero,'train':lambda:train(spec.role)}[spec.action]()


if __name__=='__main__':
    main()
