"""Fit one common critic on fresh matched right-side shaping returns."""
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
from check_getup_right_upright_native_contract import DRAFT, SCENE, TERM, configuration, install_state
from getup_anchor_runtime import RecordingSnapshot, startup
from getup_contact_training import discounted_returns
from getup_early_support_reward import inputs
from getup_release_fork_state import digest, signature
from getup_value_bootstrap import fit, networks, td
from record_io import write
from run_getup_early_support_value import scores
from run_getup_right_upright_clock import OUT as CLOCK_OUT, PLAN as CLOCK_PLAN, checked as clock_checked
from run_local_getup_ppo import MemoryWatch, check_project
import training_live_page as live


PLAN = ROOT/'configs/experiments/getup_right_upright_value.json'
OUT = ROOT/'records/reports/getup-right-upright-value'
PAGE = ROOT/'records/getup-right-upright-value-live.html'
STAGE = 'S3-right-side-upright-value-development'
PREPARATION = 'S3-right-side-upright-value-preparation'
SEED = 1099
WIDTH = 90


def freeze():
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    project = read(ROOT/'project.json')['staged_training_request']
    assert project['current_stage']==PREPARATION and not project['pause_requested']
    clock = clock_checked()
    receipt,signal_path = (CLOCK_OUT/name for name in ('analysis.json','value-signal-review.json'))
    proof,signal = read(receipt),read(signal_path)
    assert proof['status']=='passed-actual-cuda-reward-clock' and not proof['heldout_used']
    assert proof['complete_first_episode_triggers']>=1 and not proof['release_mismatches']
    assert signal['status']=='verified-value-signal-scale' and not signal['heldout_used']
    assert not signal['weights']['10']['contribution_larger_than_existing_count']
    assert signal['weights']['50']['contribution_larger_than_existing_count']>0
    for name,expected in signal['source_sha256'].items():
        assert sha(ROOT/name)==expected,name
    prior_path = ROOT/'configs/experiments/getup_early_support_value.json'
    prior = read(prior_path)['environment']
    p = deepcopy(clock['environment'])
    assert p['model_sha256']==prior['model_sha256']
    for name in ('critic_updates','minibatch_size','critic_learning_rate','gradient_norm_max',
                 'training_metric_samples','evaluation_updates','value_gate'):
        p[name] = prior[name]
    assert p['critic_updates']==50 and p['value_gate']['validation_explained_variance_min']==.5
    assert not any(f'_s{SEED}_' in path.name for path in (ROOT/'records/experiments').iterdir())
    rng = np.random.default_rng(SEED)
    codes = np.concatenate(([3],rng.permutation(np.repeat(np.arange(4),[128,128,256,511]))))
    p.update(seed=SEED,num_envs=len(codes),stage=STAGE,minibatch_seed=109903)
    p['split'] = dict(pose_codes=codes.tolist())
    p['run_id'] = f'duck_getup_right-upright-value_development_s{SEED}_{datetime.now(TZ):%Y%m%dT%H%M%S}'
    files = (CLOCK_PLAN,receipt,signal_path,prior_path,Path(__file__),ROOT/'scripts/validation/verify_getup_right_upright_value.py',
             ROOT/'scripts/validation/verify_getup_early_support_value.py',ROOT/'scripts/validation/run_getup_right_upright_value.sh',
             ROOT/'scripts/training/getup_value_bootstrap.py',ROOT/'scripts/validation/run_getup_early_support_value.py')
    sources = dict(clock['sources'])
    sources.update({path.relative_to(ROOT).as_posix():sha(path) for path in files})
    spec = dict(status='frozen-before-dynamics',stage=STAGE,environment=p,critic_width=WIDTH,actor_width=61,
        release_age=clock['release_age'],alignment=clock['alignment'],entry=clock['entry'],horizon=clock['horizon'],
        reward_coefficient=10.,controls=400,transitions=409600,physical_substeps=2000,
        validation_order=rng.permutation(len(codes)).tolist(),validation_per_pose=16,preferred_right_validation_triggers=8,
        minimum_validation_triggers=3,minimum_training_triggers=8,maximum_release_mismatches=0,
        validation_rule='Order by fixed permutation, reserve 16 per requested pose; within requested right prefer up to 8 observed trigger worlds. No reward or outcome used.',
        reward_audit_controls=[101,125,150,180,220,400],memory_limit=.85,maximum_sampling_seconds=260,
        actor_updates=0,hardware_io=False,heldout_used=False,automatic_retry=False,automatic_extension=False,
        sources=sources,library_sha256=clock['library_sha256'])
    OUT.mkdir()
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='frozen-before-dynamics',protocol_sha256=sha(PLAN),physical_substeps=0,
          run_id=p['run_id'],observation=PAGE.name,critic_width=WIDTH,actor_width=61,source_count=len(sources)))
    print('RIGHT_UPRIGHT_VALUE_FROZEN',p['run_id'],sha(PLAN),flush=True)


def checked():
    p = read(PLAN)
    assert p['status']=='frozen-before-dynamics' and p['stage']==STAGE
    assert not any(p[name] for name in ('actor_updates','hardware_io','heldout_used','automatic_retry','automatic_extension'))
    for root,items in ((ROOT,p['sources']),(WORK,p['environment']['training_source_sha256']),(Path('/'),p['library_sha256'])):
        for name,digest in items.items():
            assert sha(root/name)==digest,name
    return p


def features(state,age,observed):
    triggered = state.trigger_age>=0
    remaining = torch.where(triggered,(state.horizon-age+state.trigger_age).clamp(0,state.horizon),0)/state.horizon
    scalar = torch.stack((state.release_known,state.right,triggered,remaining,state.previous_potential),dim=1)
    # The write pointer indexes the oldest sample. Put the history in causal
    # order so that a model sees the same state during fitting and during PPO.
    def chronological(values):
        return torch.cat((values[state.slot:],values[:state.slot]),dim=0).T
    return torch.cat((observed,scalar,chronological(state.tilts)/180.,chronological(state.supported).float()),dim=1)


def select_worlds(spec,triggered):
    codes = np.asarray(spec['environment']['split']['pose_codes'])
    order = np.asarray(spec['validation_order'])
    valid = []
    for pose in range(4):
        group = order[codes[order]==pose]
        preferred = group[triggered[group]][:spec['preferred_right_validation_triggers']] if pose==3 else []
        rest = group[~np.isin(group,preferred)]
        valid.extend([*preferred,*rest[:spec['validation_per_pose']-len(preferred)]])
    train = np.flatnonzero(~np.isin(np.arange(len(codes)),valid))
    assert len(train)==960 and len(valid)==64
    return dict(pose_codes=codes.tolist(),train_ids=train.tolist(),validation_ids=[int(x) for x in valid])


def sample():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    spec = checked()
    p = deepcopy(spec['environment'])
    project = read(ROOT/'project.json')['staged_training_request']
    assert project['current_stage']==STAGE and project['current_protocol_frozen'] and ROOT/project['current_protocol']==PLAN
    assert read(OUT/'windows-entry.json')['protocol_sha256']==sha(PLAN)
    check_project()
    record = ROOT/'records/experiments'/p['run_id']
    record.mkdir()
    shutil.copy2(PLAN,record/'protocol.json')
    (record/'README.md').write_text('# 右侧起身价值起点\n\n[冻结方案](protocol.json) · [执行日志](execution.log) · [状态](state.json)\n',encoding='utf-8')
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg,display = configuration(p)
    cfg.env.rewards[TERM].weight = spec['reward_coefficient']
    dump_yaml(record/'env.yaml',asdict(cfg.env))
    write(record/'display.json',display)
    env,video,snapshot = None,None,None
    controls,updates,last = 0,0,0
    recent = monotonic()
    rate,camera = [],None
    status = 'stopped-review-required'
    watch = MemoryWatch(spec['memory_limit'])

    def publish(result):
        nonlocal recent,last
        now = monotonic()
        if controls>last:
            rate.append(dict(control=controls,transitions_per_second=(controls-last)*p['num_envs']/(now-recent)))
            recent,last = now,controls
        clip = next((row['file'] for row in video.clips if row['complete_clip']),None) if video else None
        with patch.object(live,'PAGE',PAGE):
            live.publish(record,dict(status=result,run_id=p['run_id'],purpose='右侧起身 · 价值起点',
                updated_at=datetime.now(TZ).isoformat(),controls=controls,maximum_controls=spec['controls'],
                transitions=controls*p['num_envs'],learning_updates=updates,updates=rate,latest_clip=clip,frame=camera))

    def log(line):
        with (record/'execution.log').open('a',encoding='utf-8') as stream:
            stream.write(line+'\n')
        print(line,flush=True)

    def on_fit(**row):
        nonlocal updates
        watch.check()
        updates = row['completed_updates']
        publish('running')
        log(f'RIGHT_UPRIGHT_VALUE_FIT {updates}/50')

    with watch.active():
        try:
            env = ManagerBasedRlEnv(cfg=cfg.env,device='cuda:0',render_mode=None)
            state = install_state(env,read(DRAFT),spec['release_age'],spec['alignment'])
            obs,_ = env.reset(seed=p['seed'])
            actor,critic = networks(WIDTH,p,env.device)
            startup(env,obs,actor,p,record,SCENE)
            original_actor = digest(signature(actor.state_dict()))
            assert env._sim_step_counter==0 and not state.release_known.any()
            watch.sample()
            write(record/'zero-step.json',dict(status='passed',physical_substeps=0,actor_width=61,critic_width=WIDTH,
                actor_sha256=original_actor,gpu=watch.summary()))
            snapshot = RecordingSnapshot(env,record)
            video = ActionVideo(snapshot,'getup',record)
            video.sample()
            camera = 'frame-step-0.jpg'
            q0 = cpu(env.sim.data.qpos[0])
            manager = env.reward_manager
            assert manager._term_names[-1]==TERM and manager._scale_by_dt
            term_cfg = manager._term_cfgs[-1]
            term,compute = term_cfg.func,manager.compute
            n = env.num_envs
            axes = dict(critic_obs=(n,WIDTH),actor_gravity=(n,3),physical_gravity=(n,3),z=(n,),feet=(n,2),head=(n,),
                signal=(n,),rate=(n,),control_reward=(n,),candidate_reward=(n,),trigger=(n,),done=(n,),qpos_env0=q0.shape)
            boolean = {'feet','head','done'}
            data = {key:torch.empty((400,*dims),device=env.device,dtype=torch.bool if key in boolean
                    else torch.long if key=='trigger' else torch.float32) for key,dims in axes.items()}
            audits = []
            raw = None
            i = 0
            began = monotonic()

            def measure_term(actual):
                nonlocal raw
                assert actual is env and raw is None
                z,feet,head = inputs(env)
                raw = term(env)
                if i+1 in spec['reward_audit_controls']:
                    expected_feet,expected_head = native_contact(env)
                    torch.testing.assert_close(feet,expected_feet,atol=0,rtol=0)
                    torch.testing.assert_close(head,expected_head,atol=0,rtol=0)
                    audits.append(i+1)
                for name,value in (('z',z),('feet',feet),('head',head),('signal',raw),
                                   ('trigger',state.trigger_age),('qpos_env0',env.sim.data.qpos[0])):
                    data[name][i].copy_(value)
                return raw

            def measure_compute(dt):
                nonlocal raw
                assert dt==.02
                total = compute(dt)
                scaled = torch.nan_to_num(raw*spec['reward_coefficient']*dt,nan=0.,posinf=0.,neginf=0.)/dt
                torch.testing.assert_close(manager._step_reward[:,-1],scaled,atol=0,rtol=0)
                data['rate'][i].copy_(manager._step_reward[:,-1])
                data['candidate_reward'][i].copy_(total)
                data['control_reward'][i].copy_(manager._step_reward[:,:-1].sum(dim=1)*dt)
                raw = None
                return total

            publish('running')
            with torch.no_grad(),patch.object(term_cfg,'func',measure_term),patch.object(manager,'compute',measure_compute):
                for i in range(400):
                    age = env.episode_length_buf
                    assert (age==i).all()
                    state.observe_release(obs['actor'],age)
                    data['critic_obs'][i].copy_(features(state,age,obs['critic']))
                    data['actor_gravity'][i].copy_(obs['actor'][:,3:6])
                    data['physical_gravity'][i].copy_(env.scene['robot'].data.projected_gravity_b)
                    action = actor(obs,stochastic_output=True)
                    assert torch.isfinite(action).all()
                    obs,_,terminated,timeout,_ = env.step(action)
                    controls += 1
                    assert not terminated.any() and bool(timeout.all() if controls==400 else ~timeout.any())
                    assert env._sim_step_counter==controls*5
                    data['done'][i].copy_(timeout)
                    if video.writer is not None:
                        video.sample()
                    if controls%25==0:
                        watch.check()
                        assert monotonic()-began<spec['maximum_sampling_seconds']
                        camera = snapshot.capture()['file']
                        publish('running')
                        log(f'RIGHT_UPRIGHT_VALUE_SAMPLE {controls}/400')
            arrays = {key:cpu(value) for key,value in data.items()}
            del data
            np.savez_compressed(record/'value-data.npz',initial_qpos_env0=q0,**arrays)
            write(record/'contact-audits.json',audits)
            triggered = (arrays['trigger']>=0).any(axis=0)
            p['split'] = select_worlds(spec,triggered)
            write(record/'split.json',p['split'])
            train,valid = (np.asarray(p['split'][name]) for name in ('train_ids','validation_ids'))
            exposure = dict(train_triggers=int(triggered[train].sum()),validation_triggers=int(triggered[valid].sum()))
            enough = (exposure['train_triggers']>=spec['minimum_training_triggers'] and
                      exposure['validation_triggers']>=spec['minimum_validation_triggers'])
            g = arrays['physical_gravity'][spec['release_age']].astype(np.float64)
            norm = np.linalg.norm(g,axis=1)
            actual_right = (norm>0)&(np.abs(g).argmax(axis=1)==1)&(g[:,1]<=-spec['alignment']*norm)
            mismatches = int((actual_right!=arrays['critic_obs'][spec['release_age'],:,76]).sum())
            if enough and mismatches<=spec['maximum_release_mismatches']:
                common = (arrays['control_reward'].astype(np.float64)+arrays['candidate_reward'].astype(np.float64))*.5
                fitted = fit(critic,dict(critic_obs=arrays['critic_obs'],rewards=common),p,record,on_fit)
                x = torch.from_numpy(arrays['critic_obs'][:,valid].reshape(-1,WIDTH)).to(env.device)
                with torch.no_grad():
                    predictions = torch.cat([critic(td('critic',part)).squeeze(-1) for part in x.split(4096)]).cpu().numpy().reshape(400,64)
                targets = {label:discounted_returns(torch.from_numpy(arrays[label+'_reward'][:,valid]),p['gamma']).numpy()/fitted['reward_scale']
                           for label in ('control','candidate')}
                result = scores(predictions,targets,p)
                assert updates==p['critic_updates']
                result.update(gates_passed=all(row['gate_passed'] for row in result.values()) and fitted['value_gate_passed'],
                              critic_sha256=sha(record/'critic-final.pt'))
            else:
                reason = 'release-observation-disagreement' if enough else 'insufficient-distinct-trigger-worlds'
                result = dict(gates_passed=False,reason=reason)
            assert original_actor==digest(signature(actor.state_dict()))
            watch.check()
            write(record/'fit-result.json',dict(status='collected-review-required',**result,**exposure,
                controls=controls,physical_substeps=env._sim_step_counter,transitions=controls*n,
                data_sha256=sha(record/'value-data.npz'),initial_actor_sha256=original_actor,final_actor_sha256=original_actor,
                critic_updates=updates,actor_updates=0,release_mismatches=mismatches,gpu=watch.summary()))
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


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','sample'))
    args = parser.parse_args()
    (freeze if args.command=='freeze' else sample)()
