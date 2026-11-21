"""Exercise the new support reward in the actual training plant without fitting."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import importlib
import os
from pathlib import Path
import shutil
import sys
from time import monotonic, perf_counter
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from s1_reference_inputs import ROOT, WORK, read, sha
from action_video import ActionVideo
from common_scene_contract import compare_arrays, model_arrays
from getup_anchor_runtime import RecordingSnapshot, startup
from getup_anchor_same_release import visible_log
from getup_early_support_reward import SupineReleaseLatch, component, early_support, inputs, reset_early_support
from getup_release_fork_state import signature
from getup_teacher_anchor_train import prepare
from getup_value_bootstrap import networks
from local_getup_scale import gpu_sample
from record_io import write
from run_local_getup_ppo import check_project
import training_live_page as live


PLAN = ROOT/'configs/experiments/getup_early_support_clock.json'
OUT = ROOT/'records/reports/getup-early-support-clock'
PAGE = ROOT/'records/getup-early-support-clock-live.html'
STAGE = 'S3-early-support-clock-development'
TZ = ZoneInfo('Asia/Shanghai')
TERM = 'early_supine_support'


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


def config(spec):
    from mjlab.managers import EventTermCfg, RewardTermCfg

    cfg, display, _ = prepare(spec['environment'])
    assert cfg.env.auto_reset and cfg.env.episode_length_s == 8. and cfg.env.decimation == 5
    assert TERM not in cfg.env.rewards and TERM not in cfg.env.events
    cfg.env.rewards[TERM] = RewardTermCfg(func=early_support, weight=1., params=spec['reward'])
    cfg.env.events[TERM] = EventTermCfg(func=reset_early_support, mode='reset')
    return cfg, display


def cpu_fixture(spec):
    latch = SupineReleaseLatch(5, 'cpu', spec['release_age'], spec['alignment'])
    x = torch.zeros(5,61)
    x[:,3:6] = torch.tensor([[-1.,0.,0.],[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.],[0.,0.,0.]])
    latch.observe(x,torch.full((5,),99))
    assert not latch.latched.any()
    latch.observe(x,torch.full((5,),100))
    assert latch.latched.all() and latch.supine.tolist()==[True,False,False,False,False]
    latch.observe(-x,torch.full((5,),100))
    assert latch.supine.tolist()==[True,False,False,False,False]
    latch.reset(torch.tensor([0,2]))
    assert latch.latched.tolist()==[False,True,False,True,True] and not latch.supine.any()
    x[:2,3:6] = torch.tensor([-1.,0.,0.])
    latch.observe(x,torch.tensor([100,100,0,0,0]))
    assert latch.supine.tolist()==[True,False,False,False,False]
    values = component(-torch.ones(5),torch.tensor([[True,True],[True,True],[True,False],[True,True],[True,True]]),
        torch.tensor([False,False,False,False,True]),torch.tensor([114,115,150,151,125]),torch.ones(5,dtype=torch.bool),
        **spec['reward'])
    torch.testing.assert_close(values,torch.tensor([0.,1.,.5,0.,0.]),atol=0,rtol=0)
    return dict(release_age_exclusive=True,selective_reset=True,latched_once=True,window_exact=True)


def checked():
    spec = read(PLAN)
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(PLAN)
    for base, items in ((ROOT,spec['references']),(ROOT,spec['source_sha256']),
                        (WORK,spec['environment']['training_source_sha256']),(Path('/'),spec['library_source_sha256'])):
        for name, expected in items.items():
            assert sha(base/name)==expected,name
    return spec


def freeze():
    from mjlab.scene import Scene

    assert not PLAN.exists() and not OUT.exists()
    source = ROOT/'configs/experiments/local_getup_scale.json'
    draft = ROOT/'configs/experiments/getup_supine_early_support_draft.json'
    signal = ROOT/read(draft)['saved_signal_screen']
    evidence = read(signal)
    assert evidence['status']=='passed-saved-development-review' and not evidence['heldout_used']
    assert all(row['same_release_latch']==320 and row['groups']['other_pose']['nonzero']==0 for row in evidence['cohorts'])
    p = deepcopy(read(source)['role'])
    assert p['model_sha256']==sha(ROOT/p['model'])
    n, seed = 128, 1081
    assert not any(f'_s{seed}_' in f.name for f in (ROOT/'records/experiments').iterdir())
    codes = np.concatenate(([1],np.random.default_rng(seed).permutation(np.repeat(np.arange(4),[32,31,32,32]))))
    p.update(seed=seed,num_envs=n,stage=STAGE,learning_updates=0)
    p['split']['pose_codes'] = codes.tolist()
    p['run_id'] = f"duck_getup_early-support-clock_development_s{seed}_{datetime.now(TZ).strftime('%Y%m%dT%H%M%S')}"
    assert p['criteria']['freeze_seconds']==2. and evidence['window_seconds']==[2.3,3.]
    scene = ROOT/'records/reports/getup-interface-value/cpu-scene-fields.npz'
    spec = dict(status='frozen-before-physics',stage=STAGE,environment=p,release_age=100,
        alignment=p['criteria']['release_alignment_min'],reward=dict(start_age=115,stop_age=150,
        tilt_limit_degrees=p['criteria']['tilt_max_deg']),controls=550,physics_substeps=2750,
        transitions=n*550,partial_reset_control=160,reward_audit_controls=[115,125,150,275,290,310,515,525,550],
        heldout_used=False,learning_updates=0,hardware_io=False,maximum_seconds=240,automatic_retry=False,
        references={path.relative_to(ROOT).as_posix():sha(path) for path in (source,draft,signal,scene,ROOT/p['model'],ROOT/p['standing_model'])})
    sources = dict(p['source_sha256'])
    for path in (Path(__file__).resolve(),Path(__file__).resolve().with_name('verify_getup_early_support_clock.py'),
                 ROOT/'scripts/training/getup_early_support_reward.py',ROOT/'scripts/training/training_live_page.py'):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    spec['source_sha256'] = sources
    spec['library_source_sha256'] = dict(p['library_source_sha256'])
    for name in ('mjlab.envs.manager_based_rl_env','mjlab.managers.reward_manager','mjlab.managers.event_manager'):
        path = Path(importlib.import_module(name).__file__)
        spec['library_source_sha256'][str(path)] = sha(path)
    for base, items in ((ROOT,sources),(WORK,p['training_source_sha256']),(Path('/'),spec['library_source_sha256'])):
        for path,expected in items.items():
            assert sha(base/path)==expected,path
    cfg, _ = config(spec)
    fields = model_arrays(Scene(cfg.env.scene,device='cpu').compile())
    with np.load(scene) as saved:
        compare_arrays(fields,{name:saved[name] for name in saved.files})
    fixture = cpu_fixture(spec)
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='passed-cpu-zero-step',protocol_sha256=sha(PLAN),scene_exact=True,
          fixture=fixture,physical_substeps=0,learning_updates=0))
    print('GETUP_SUPPORT_CLOCK_FROZEN',sha(PLAN),p['run_id'],flush=True)


def native_contact(env):
    """Independently reconstruct named contact pairs before the final forward call."""
    import mujoco
    import warp as wp

    model = env.sim.mj_model
    body = model.body('terrain').id
    ground = np.flatnonzero(model.geom_bodyid==body)
    assert ground.size
    geom_id = lambda name: mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'robot/'+name+'_collision')
    feet = [geom_id(f'{side}_foot') for side in ('left','right')]
    head, legs = geom_id('head'), [*feet,geom_id('left_hip_yaw'),geom_id('right_hip_yaw')]
    assert min([head,*legs])>=0
    slot = env.sim.wp_data.contact
    size = int(env.sim.wp_data.nacon.numpy()[0])
    assert 0<=size<slot.dist.shape[0]
    valid = (wp.to_torch(slot.dist)[:size]<0)&((wp.to_torch(slot.type)[:size]&1)!=0)
    a,b = wp.to_torch(slot.geom)[:size][valid].T
    worlds = wp.to_torch(slot.worldid)[:size][valid].long()

    def found(primary, secondaries):
        ids = torch.as_tensor(secondaries,device=env.device,dtype=a.dtype)
        matches = ((a==primary)&torch.isin(b,ids))|((b==primary)&torch.isin(a,ids))
        result = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        result[worlds[matches]] = True
        return result

    return torch.stack([found(foot,ground) for foot in feet],dim=1),found(head,legs)


def selective_reset(env, latch, record):
    ids = torch.nonzero(latch.supine).flatten()
    ids = ids[ids!=0][:2]
    assert len(ids)==2 and latch.latched[ids].all()

    def state():
        return dict(age=cpu(env.episode_length_buf),latched=cpu(latch.latched),supine=cpu(latch.supine),
                    qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel),action=cpu(env.action_manager._action))

    before = state()
    step = (env.common_step_counter,env._sim_step_counter)
    obs, _ = env.reset(env_ids=ids)
    assert step==(env.common_step_counter,env._sim_step_counter)
    after = state()
    mask = np.ones(env.num_envs,bool)
    mask[cpu(ids)] = False
    for key in before:
        np.testing.assert_array_equal(before[key][mask],after[key][mask])
    assert not (latch.latched[ids]|latch.supine[ids]).any() and not env.episode_length_buf[ids].any()
    np.savez_compressed(record/'selective-reset.npz',ids=cpu(ids),**{f'before_{k}':v for k,v in before.items()},
                        **{f'after_{k}':v for k,v in after.items()})
    return obs


def sample():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.os import dump_yaml

    assert os.isatty(1)
    spec = checked()
    p = spec['environment']
    record = ROOT/'records/experiments'/p['run_id']
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage']==STAGE and stage['current_protocol_frozen'] and ROOT/stage['current_protocol']==PLAN
    record.mkdir()
    shutil.copy2(PLAN,record/'protocol.json')
    (record/'README.md').write_text('# 起身奖励时钟\n\n[冻结方案](../../../configs/experiments/getup_early_support_clock.json) · '
        '[执行日志](execution.log) · [状态](state.json)\n',encoding='utf-8')
    rows, audits, rates = [], [], []
    video, env, snapshot = None, None, None
    last, instant = 0, perf_counter()
    controls = 0

    def publish(status):
        nonlocal last, instant
        now = perf_counter()
        if controls>last:
            rates.append(dict(control=controls,transitions_per_second=(controls-last)*p['num_envs']/(now-instant)))
            last, instant = controls, now
        clips = video.clips if video is not None else []
        complete = next((row['file'] for row in clips if row['complete_clip']),None)
        live.publish(record,dict(status=status,run_id=p['run_id'],purpose='仰卧早期支撑 · 奖励时钟',
            updated_at=datetime.now(TZ).isoformat(),controls=controls,maximum_controls=spec['controls'],
            transitions=controls*p['num_envs'],learning_updates=0,updates=rates,latest_clip=complete,
            frame='frame-step-0.jpg' if (record/'frame-step-0.jpg').is_file() else None))

    status = 'stopped-review-required'
    with (record/'execution.log').open('x',buffering=1) as log, visible_log(log), patch.object(live,'PAGE',PAGE):
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
            actor,_ = networks(75,p,env.device)
            startup(env,obs,actor,p,record,ROOT/'records/reports/getup-interface-value/cpu-scene-fields.npz')
            assert not latch.latched.any() and not latch.supine.any() and env._sim_step_counter==0
            state_hash = signature(actor.state_dict())
            gpu = gpu_sample()
            assert gpu['used_mib']/gpu['total_mib']<=.85
            write(record/'zero-step.json',dict(status='passed',physics_substeps=0,gpu=gpu))
            snapshot = RecordingSnapshot(env,record)
            video = ActionVideo(snapshot,'getup',record)
            video.sample()
            initial = cpu(env.sim.data.qpos[0])
            manager = env.reward_manager
            assert manager._scale_by_dt and manager._term_names[-1]==TERM
            original_reward = manager.compute
            gathered = None
            start = monotonic()

            def capture(dt):
                nonlocal gathered
                assert dt==.02 and gathered is None
                total = original_reward(dt)
                z, feet, head = inputs(env)
                terms = manager._step_reward
                if env.common_step_counter in spec['reward_audit_controls']:
                    actual_feet,actual_head = native_contact(env)
                    torch.testing.assert_close(feet,actual_feet,atol=0,rtol=0)
                    torch.testing.assert_close(head,actual_head,atol=0,rtol=0)
                    audits.append(dict(control=env.common_step_counter,feet_worlds=int(feet.any(dim=1).sum()),
                                       head_leg_worlds=int(head.sum()),feet_equal=True,head_leg_equal=True))
                gathered = dict(reward_age=cpu(env.episode_length_buf),reward_latch=cpu(latch.supine),
                    reward_gravity_z=cpu(z),feet=cpu(feet),head_leg=cpu(head),
                    term=cpu(terms[:,-1]),base_reward=cpu(terms[:,:-1].sum(dim=1)*dt),
                    total_reward=cpu(total),qpos_env0=cpu(env.sim.data.qpos[0]))
                return total

            with torch.inference_mode(), patch.object(manager,'compute',capture):
                for k in range(1,spec['controls']+1):
                    age = cpu(env.episode_length_buf)
                    actor_gravity = cpu(obs['actor'][:,3:6])
                    critic_input = cpu(obs['critic'])
                    latch.observe(obs['actor'],env.episode_length_buf)
                    action = actor(obs)
                    assert torch.isfinite(action).all()
                    obs,reward,terminated,timeout,_ = env.step(action)
                    controls += 1
                    assert gathered is not None and not terminated.any() and torch.isfinite(reward).all()
                    assert env._sim_step_counter==controls*5
                    rows.append(dict(gathered,actor_gravity=actor_gravity,pre_age=age,critic_input=critic_input,
                        post_age=cpu(env.episode_length_buf),post_latched=cpu(latch.latched),post_supine=cpu(latch.supine),done=cpu(timeout)))
                    gathered = None
                    if video.writer is not None:
                        video.sample()
                    if k==spec['partial_reset_control']:
                        obs = selective_reset(env,latch,record)
                    if k%25==0:
                        check_project()
                        assert monotonic()-start<spec['maximum_seconds']
                        publish('running')
                        print('GETUP_SUPPORT_CLOCK',k,'/',spec['controls'],flush=True)
            assert state_hash==signature(actor.state_dict()) and controls==spec['controls']
            np.savez_compressed(record/'reward-trace.npz',initial_qpos_env0=initial,**{key:np.stack([row[key] for row in rows]) for key in rows[0]})
            write(record/'reward-audits.json',audits)
            status = 'collected-review-required'
        finally:
            if video is not None:
                video.close()
            if snapshot is not None:
                snapshot.close()
            if env is not None:
                env.close()
            publish(status)


if __name__ == '__main__':
    check_project()
    assert sys.argv[1:] in (['--freeze'],['--sample'])
    (freeze if sys.argv[1]=='--freeze' else sample)()
