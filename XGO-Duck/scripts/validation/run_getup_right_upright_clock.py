"""Sample the right-side reward clock with a fixed actor on the real CUDA plant."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch

from s1_reference_inputs import ROOT, WORK, read, sha
from action_video import ActionVideo
from check_getup_early_support_clock import TZ, cpu, native_contact
from check_getup_right_upright_native_contract import (
    DRAFT, DEST as NATIVE_CPU, PARENT, SCENE, TERM, configuration, install_state,
)
from getup_anchor_runtime import RecordingSnapshot, startup
from getup_coverage_metrics import POSES
from getup_early_support_reward import inputs
from getup_release_fork_state import digest, signature
from getup_value_bootstrap import networks
from local_getup_scale import gpu_sample
from record_io import write
from run_local_getup_ppo import check_project
import training_live_page as live


PLAN = ROOT/'configs/experiments/getup_right_upright_clock.json'
OUT = ROOT/'records/reports/getup-right-upright-clock'
PAGE = ROOT/'records/getup-right-upright-clock-live.html'
STAGE = 'S3-right-side-upright-native-cuda-clock'
SEED = 1098


def freeze():
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    parent,draft,proof = read(PARENT),read(DRAFT),read(NATIVE_CPU)
    assert proof['status']=='passed-native-cpu-registration' and not proof['physical_substeps']
    assert draft['status']=='draft-not-executable' and not draft['heldout_used'] and not draft['hardware_io']
    for root,hashes in ((ROOT,proof['source_sha256']),(Path('/'),proof['library_sha256'])):
        for name,expected in hashes.items():
            assert sha(root/name)==expected,name
    project = read(ROOT/'project.json')['staged_training_request']
    assert project['current_stage']==draft['stage'] and not project['pause_requested']
    assert not any(f'_s{SEED}_' in f.name for f in (ROOT/'records/experiments').iterdir())
    plant = deepcopy(parent['environment'])
    assert tuple(POSES)==('prone','supine','left-side','right-side')
    requested = np.concatenate(([3],np.random.default_rng(SEED).permutation(np.repeat(np.arange(4),[32,32,64,127]))))
    plant.update(seed=SEED,num_envs=len(requested),stage=STAGE)
    plant['split'] = dict(pose_codes=requested.tolist(),groups={pose:np.flatnonzero(requested==i).tolist()
                                                               for i,pose in enumerate(POSES)})
    plant['run_id'] = f'duck_getup_right-upright-clock_development_s{SEED}_{datetime.now(TZ):%Y%m%dT%H%M%S}'
    history = ROOT/draft['trigger_transfer']
    sources = dict(parent['references'],**parent['source_sha256'])
    sources.update(proof['source_sha256'])
    files = [PARENT,DRAFT,NATIVE_CPU,SCENE,history,ROOT/plant['model'],ROOT/plant['standing_model'],Path(__file__)]
    files += [ROOT/'scripts/validation'/name for name in ('verify_getup_right_upright_clock.py',
             'run_getup_right_upright_clock.sh','verify_getup_early_support_clock.py','verify_walk_live_page.py')]
    sources.update({path.relative_to(ROOT).as_posix():sha(path) for path in files})
    controls = 550
    spec = dict(status='frozen-before-dynamics',stage=STAGE,environment=plant,controls=controls,
        physics_substeps=controls*5,transitions=controls*len(requested),release_age=parent['release_age'],
        alignment=parent['alignment'],horizon=draft['proposed_reward_window_controls'],entry=read(history)['reused_entry'],
        partial_reset_control=250,partial_reset_worlds=2,minimum_first_episode_triggers=1,
        partial_reset_selection='Triggered right-side worlds first, then other right-side worlds in index order, excluding camera world 0.',
        reward_audit_controls=[101,115,125,145,175,215,250,351,375,400,501,525,550],
        maximum_seconds=260,memory_fraction_limit=.85,learning_updates=0,hardware_io=False,heldout_used=False,
        automatic_retry=False,automatic_extension=False,sources=sources,
        library_sha256={**parent['library_source_sha256'],**proof['library_sha256']})
    OUT.mkdir()
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='frozen-after-native-cpu-admission',physical_substeps=0,
        protocol_sha256=sha(PLAN),run_id=plant['run_id'],page=PAGE.name,frozen_files=len(sources)))
    print('RIGHT_UPRIGHT_CLOCK_FROZEN',plant['run_id'],sha(PLAN),flush=True)


def checked():
    p = read(PLAN)
    assert p['stage']==STAGE and p['status']=='frozen-before-dynamics'
    assert not any(p[name] for name in ('learning_updates','hardware_io','heldout_used','automatic_retry','automatic_extension'))
    for root,hashes in ((ROOT,p['sources']),(WORK,p['environment']['training_source_sha256']),
                        (Path('/'),p['library_sha256'])):
        for name,expected in hashes.items():
            assert sha(root/name)==expected,name
    return p


def reset_selected(env,state,record,n):
    triggered = state.trigger_age>=0
    ids = torch.cat((torch.nonzero(state.right&triggered).flatten(),torch.nonzero(state.right&~triggered).flatten()))
    ids = ids[ids!=0][:n]
    assert len(ids)==n

    def view():
        return dict(age=cpu(env.episode_length_buf),known=cpu(state.release_known),right=cpu(state.right),
            trigger=cpu(state.trigger_age),previous=cpu(state.previous_potential),
            supported=cpu(state.supported.T),tilts=cpu(state.tilts.T),qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel))

    before = view()
    counters = (env.common_step_counter,env._sim_step_counter)
    observations,_ = env.reset(env_ids=ids)
    after = view()
    assert counters==(env.common_step_counter,env._sim_step_counter)
    keep = np.ones(env.num_envs,dtype=bool)
    keep[cpu(ids)] = False
    for name,value in before.items():
        np.testing.assert_array_equal(value[keep],after[name][keep],err_msg=name)
    assert all(not after[name][cpu(ids)].any() for name in ('age','known','right','previous','supported','tilts'))
    assert (after['trigger'][cpu(ids)]==-1).all()
    np.savez_compressed(record/'selective-reset.npz',ids=cpu(ids),**{f'{side}_{key}':value
                        for side,group in (('before',before),('after',after)) for key,value in group.items()})
    return observations


def sample():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    spec = checked()
    p = spec['environment']
    project = read(ROOT/'project.json')['staged_training_request']
    assert project['current_stage']==STAGE and project['current_protocol_frozen'] and ROOT/project['current_protocol']==PLAN
    assert read(OUT/'windows-entry.json')['protocol_sha256']==sha(PLAN)
    check_project()
    record = ROOT/'records/experiments'/p['run_id']
    record.mkdir()
    shutil.copy2(PLAN,record/'protocol.json')
    (record/'README.md').write_text('# 右侧起身奖励时钟\n\n[冻结方案](protocol.json) · [执行日志](execution.log) · [状态](state.json)\n',encoding='utf-8')
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg,display = configuration(p)
    dump_yaml(record/'env.yaml',asdict(cfg.env))
    write(record/'display.json',display)
    env = ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None)
    video,snapshot = None,None
    controls,last = 0,0
    now = monotonic()
    rate = []
    camera = None
    status = 'stopped-review-required'

    def publish(result):
        nonlocal now,last
        t = monotonic()
        if controls>last:
            rate.append(dict(control=controls,transitions_per_second=(controls-last)*env.num_envs/(t-now)))
            now,last = t,controls
        complete = next((clip['file'] for clip in video.clips if clip['complete_clip']),None) if video else None
        with patch.object(live,'PAGE',PAGE):
            live.publish(record,dict(status=result,run_id=p['run_id'],purpose='右侧起身 · 原生奖励时钟',
                updated_at=datetime.now(TZ).isoformat(),controls=controls,maximum_controls=spec['controls'],
                transitions=controls*p['num_envs'],learning_updates=0,updates=rate,latest_clip=complete,frame=camera))

    try:
        state = install_state(env,read(DRAFT),spec['release_age'],spec['alignment'])
        obs,_ = env.reset(seed=p['seed'])
        actor,_ = networks(75,p,env.device)
        startup(env,obs,actor,p,record,SCENE)
        actor_hash = digest(signature(actor.state_dict()))
        gpu = gpu_sample()
        assert gpu['used_mib']/gpu['total_mib']<=spec['memory_fraction_limit']
        assert env._sim_step_counter==0 and not state.release_known.any() and (state.trigger_age==-1).all()
        write(record/'zero-step.json',dict(status='passed',physical_substeps=0,gpu=gpu,actor_sha256=actor_hash))
        snapshot = RecordingSnapshot(env,record)
        video = ActionVideo(snapshot,'getup',record)
        video.sample()
        camera = 'frame-step-0.jpg'
        initial = cpu(env.sim.data.qpos[0])
        manager = env.reward_manager
        assert manager._scale_by_dt and manager._term_names[-1]==TERM
        term_config = manager._term_cfgs[-1]
        term,compute = term_config.func,manager.compute
        collected,rows,audits = {},[],[]
        raw_on_device = None
        start = monotonic()

        def measure_term(actual_env):
            nonlocal raw_on_device
            assert not collected and actual_env is env
            previous_trigger = state.trigger_age.clone()
            gravity,feet,head = inputs(env)
            raw = term(env)
            raw_on_device = raw
            k = env.common_step_counter
            new = state.trigger_age!=previous_trigger
            active = ~(env.termination_manager.terminated|env.termination_manager.time_outs)
            if k in spec['reward_audit_controls'] or new.any():
                native_feet,native_head = native_contact(env)
                torch.testing.assert_close(feet,native_feet,rtol=0,atol=0)
                torch.testing.assert_close(head,native_head,rtol=0,atol=0)
                audits.append(dict(control=k,feet_equal=True,head_leg_equal=True,trigger_worlds=int(new.sum()),
                                   contact_worlds=int(head.sum())))
            collected.update(reward_age=cpu(env.episode_length_buf),z=cpu(gravity),feet=cpu(feet),head_leg=cpu(head),
                active=cpu(active),raw=cpu(raw),reward_trigger=cpu(state.trigger_age),reward_right=cpu(state.right),
                qpos_env0=cpu(env.sim.data.qpos[0]))
            return raw

        def measure_compute(dt):
            assert dt==.02
            total = compute(dt)
            expected_rate = torch.nan_to_num(raw_on_device*1.*dt,nan=0.,posinf=0.,neginf=0.)/dt
            torch.testing.assert_close(manager._step_reward[:,-1],expected_rate,rtol=0,atol=0)
            collected.update(rate=cpu(manager._step_reward[:,-1]),total_reward=cpu(total),
                             base_reward=cpu(manager._step_reward[:,:-1].sum(dim=1)*dt))
            return total

        publish('running')
        with torch.inference_mode(),patch.object(term_config,'func',measure_term),patch.object(manager,'compute',measure_compute):
            for k in range(1,spec['controls']+1):
                age = cpu(env.episode_length_buf)
                pre = dict(pre_age=age,actor_gravity=cpu(obs['actor'][:,3:6]),critic_input=cpu(obs['critic']),
                           physical_gravity=cpu(env.scene['robot'].data.projected_gravity_b))
                state.observe_release(obs['actor'],env.episode_length_buf)
                action = actor(obs)
                assert torch.isfinite(action).all()
                obs,reward,terminated,timeout,_ = env.step(action)
                controls += 1
                assert collected and torch.isfinite(reward).all() and env._sim_step_counter==controls*5
                rows.append(dict(**pre,**collected,post_age=cpu(env.episode_length_buf),post_right=cpu(state.right),
                    post_known=cpu(state.release_known),post_trigger=cpu(state.trigger_age),done=cpu(terminated|timeout),
                    terminated=cpu(terminated),timeout=cpu(timeout)))
                collected.clear()
                if video.writer is not None:
                    video.sample()
                if k==spec['partial_reset_control']:
                    obs = reset_selected(env,state,record,spec['partial_reset_worlds'])
                if k%25==0:
                    check_project()
                    assert monotonic()-start<spec['maximum_seconds']
                    camera = snapshot.capture()['file']
                    publish('running')
                    line = f'RIGHT_UPRIGHT_CLOCK {controls}/{spec["controls"]} trigger_worlds={int((state.trigger_age>=0).sum())}'
                    with (record/'execution.log').open('a',encoding='utf-8') as log:
                        log.write(line+'\n')
                    print(line,flush=True)
        assert actor_hash==digest(signature(actor.state_dict())) and controls==spec['controls']
        np.savez_compressed(record/'reward-trace.npz',initial_qpos_env0=initial,**{
            key:np.stack([row[key] for row in rows]) for key in rows[0]})
        write(record/'reward-audits.json',audits)
        write(record/'runtime-contract.json',dict(controls=controls,transitions=controls*env.num_envs,
              physical_substeps=env._sim_step_counter,initial_actor_sha256=actor_hash,final_actor_sha256=actor_hash,
              learning_updates=0,native_roundtrip_bitwise_at_each_control=True))
        status = 'collected-review-required'
    finally:
        if video is not None:
            video.close()
        if snapshot is not None:
            snapshot.close()
        env.close()
        publish(status)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','sample'))
    args = parser.parse_args()
    (freeze if args.command=='freeze' else sample)()
