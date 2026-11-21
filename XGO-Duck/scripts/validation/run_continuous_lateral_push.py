"""Run a matched no-push and two-direction lateral-push comparison."""
import argparse
from datetime import datetime
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch

from s1_reference_inputs import ROOT, WORK, read, sha
from record_io import write
from check_continuous_forward_fork_state import controller_snapshot, restore_controller
from check_continuous_lateral_push_cuda import OUT, PARENT, SEED, WARP_LIMIT
from check_continuous_lateral_push_geometry import DRAFT, recovery_window
from check_continuous_lateral_push_writer import apply_lateral_push
from getup_release_fork_state import digest, signature
from roll_entry_snapshot import RollEntrySnapshot
from roll_stand_walk_sequence import FORWARD, Sequence
from run_continuous_forward_yaw_pair import stop_boundary
from run_continuous_roll_projected_entry import ProjectedRollPolicy, baseline
from run_roll_entry_isolation import save_rows


PLAN = DRAFT.with_name('continuous_forward_lateral_push.json')
PAGE = ROOT/'records/continuous-forward-lateral-push-live.html'
STAGE = 'S5-roll-to-forward-lateral-perturbation-development'
LABELS = {'shared':'共同轨迹','reference':'无扰动','left-push':'向左扰动','right-push':'向右扰动'}
cpu = baseline.cpu


def freeze():
    assert not PLAN.exists() and not torch.cuda.is_initialized()
    p,draft,zero = read(PARENT),read(DRAFT),read(OUT/'cuda-zero.json')
    assert zero['status']=='passed-actual-cuda-zero-step' and zero['seed']==SEED and zero['physical_substeps']==0
    assert read(OUT/'zero/runtime-contract.json')['physics_substeps']==0
    assert not any(draft[key] for key in ('heldout_used','hardware_io','learning_updates','automatic_retry','automatic_extension'))
    for root,entries in ((ROOT,zero['source_sha256']),(OUT/'zero',zero['initial_archives'])):
        for name,expected in entries.items():
            assert sha(root/name)==expected,name
    independent = ROOT/draft['required_independent_result']
    proof = read(independent)
    prior_windows = ROOT/proof['record']/'windows-view-verification.json'
    assert proof[draft['required_independent_gate']] and read(prior_windows)['status']=='passed'
    assert sha(PARENT)==proof['protocol_sha256'] and p['worlds']==draft['proposed_worlds']==128
    assert p['minimum_continuous_successes']==p['minimum_roll_completion']==116
    assert p['forward_controls']==1000 and p['feedback']['maximum_yaw_input_rad_s']==.15
    assert p['feedback']['maximum_joint_correction_rad']==.03 and p['walking']['criteria']['yaw_error_max']==.2
    shared = draft['proposed_shared_fork_control']
    assert shared==sum(p[key] for key in ('initial_stand_controls','roll_controls','landing_stand_controls'))
    roles = draft['proposed_roles_body_horizontal_lateral_m_s']
    assert roles=={'reference':0.,'left-push':.08,'right-push':-.08}
    total = shared+len(roles)*(p['maximum_controls']-shared)
    assert total==draft['proposed_maximum_unique_controls']==3550
    assert not any(f'_s{SEED}_' in path.name for path in (ROOT/'records/evaluations').iterdir())
    files = [PARENT,DRAFT,independent,prior_windows,OUT/'cuda-zero.json',Path(__file__)]
    files += [ROOT/'scripts/validation'/name for name in ('check_continuous_lateral_push_cuda.py',
        'check_continuous_lateral_push_geometry.py','check_continuous_lateral_push_writer.py',
        'verify_continuous_lateral_push.py','verify_continuous_forward_yaw_pair.py','verify_walk_live_page.py',
        'run_continuous_lateral_push.sh')]
    sources = dict(p['frozen_sha256'],**zero['source_sha256'])
    sources.update({file.relative_to(ROOT).as_posix():sha(file) for file in files})
    run = f'duck_switch_forward-lateral-push_development_s{SEED}_{datetime.now(baseline.TZ):%Y%m%dT%H%M%S}'
    study = dict(shared=shared,roles=roles,maximum_unique_controls=total,maximum_control_transitions=total*p['worlds'],
                 maximum_physics_substeps=total*5,maximum_wall_seconds=660,
                 recovery_controls=100,gate_requires_full_and_recovery=True,observation_delay_controls=1)
    p.update(stage=STAGE,seed=SEED,roll=dict(p['roll'],seed=SEED),run_id=run,perturbation=study,
             maximum_warp_snapshot_bytes=WARP_LIMIT,frozen_sha256=sources)
    write(PLAN,p)
    write(OUT/'preflight.json',dict(status='frozen-after-cuda-zero-step',run_id=run,protocol_sha256=sha(PLAN),
        zero_sha256=sha(OUT/'cuda-zero.json'),frozen_files=len(sources),physical_substeps=0,observation=PAGE.name))
    print('CONTINUOUS_LATERAL_PUSH_FROZEN',run,sha(PLAN),flush=True)


def checked():
    p = read(PLAN)
    assert p['stage']==STAGE and p['status']=='frozen-before-dynamics'
    for root,key in ((ROOT,'frozen_sha256'),(WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
        for name,expected in p[key].items():
            assert sha(root/name)==expected,name
    zero = read(OUT/'cuda-zero.json')
    for name,expected in zero['initial_archives'].items():
        assert sha(OUT/'zero'/name)==expected,name
    return p


def collect(env,obs,p,folder,endpoint,networks,homes,sim_home,limits):
    spec = p['perturbation']
    shared,worlds,horizon = spec['shared'],p['worlds'],p['maximum_controls']
    robot = env.scene['robot']
    joints, order = robot.find_joints(p['walking']['joint_names'],preserve_order=True)
    feet, sites = robot.find_sites(('left_foot','right_foot'),preserve_order=True)
    assert order==p['walking']['joint_names'] and list(sites)==['left_foot','right_foot']
    head = robot.find_bodies('jaw_soft')[0][0]
    bounds = cpu(limits)
    tolerance = np.deg2rad(p['roll']['actual_joint_limit_tolerance_deg'])
    term = env.action_manager.get_term('joint_pos')
    seq = Sequence(p)
    policy = ProjectedRollPolicy(p,networks,homes,sim_home,limits)
    alive = np.ones(worlds,dtype=bool)
    snapshot = baseline.Snapshot(env,folder)
    render = snapshot.render
    role,view,cameras = 'shared',dict(run_id=p['run_id'],maximum_controls=spec['maximum_unique_controls'],
                                     learning_updates=0,updates=[]),{}
    pixels,last_render = None,None
    clock = start = monotonic()
    previous_count = 0
    original_step = env.sim.step

    def render_boundary():
        # Shared frames are rendered once and encoded in each original video.
        nonlocal pixels,last_render
        key = (role,env.common_step_counter)
        if last_render!=key:
            pixels,last_render = render(),key
        return pixels

    snapshot.render = render_boundary

    def publish(status='running',frame=True):
        nonlocal clock,previous_count
        k,n = env.common_step_counter,env.guard_steps
        now = monotonic()
        if n>previous_count:
            view['updates'].append(dict(control=n,transitions_per_second=(n-previous_count)*worlds/max(.01,now-clock)))
            clock,previous_count = now,n
        if frame:
            path = snapshot.capture()['file']
            view['frame'] = f'frame-{role}-{k}.jpg'
            (folder/path).replace(folder/view['frame'])
        view.update(status=status,purpose=f'前进横向扰动 · {LABELS[role]}',controls=n,transitions=n*worlds,
                    updated_at=datetime.now(baseline.TZ).isoformat())
        with patch.object(baseline.live,'PAGE',PAGE):
            baseline.live.publish(folder,view)
        message = f'CONTINUOUS_LATERAL_PUSH {role} unique={n} timeline={k} active={alive.sum()} forward={(seq.phase==FORWARD).sum()}'
        with (folder/'execution.log').open('a',encoding='utf-8') as stream:
            stream.write(message+'\n')
        print(message,flush=True)

    def take(k):
        state = baseline.capture(env,endpoint,alive,joints,head,feet,bounds,tolerance)
        assert all(np.isfinite(value).all() for value in state.values())
        seq.update(k,state)
        return dict(state,phase_next=seq.phase.copy())

    def step(k):
        observed = env.obs_buf['actor'].clone()
        target,row = policy.next(k,observed,seq)
        _,_,terminated,truncated,_ = env.step(target-sim_home)
        alive[:] &= ~cpu(terminated|truncated)
        torch.testing.assert_close(term._processed_actions,target,atol=1e-6,rtol=0)
        endpoint.capture()
        assert not endpoint.first_mismatches,endpoint.first_mismatches
        return dict(row,observed=cpu(observed),terminated=cpu(terminated),truncated=cpu(truncated),
                    actual_motor_target=cpu(term._processed_actions))

    def record(camera):
        if camera.writer is not None:
            camera.sample() if alive[0] else camera._sample_terminal()

    def clip_for(name):
        camera = cameras[name]
        if camera.writer is not None:
            camera._sample_terminal()
        clip, = camera.clips
        path = folder/'clips'/name/clip['file']
        return dict(clip,path=path.relative_to(folder).as_posix(),sha256=sha(path))

    (folder/'README.md').write_text('# 前进横向扰动\n\n'
        '三组共享前 550 步物理轨迹，随后比较无扰动、左推和右推。完整录像从共同起点开始。\n\n'
        '[方案](protocol.json) · [执行日志](execution.log) · [状态](state.json)\n',encoding='utf-8')
    try:
        for name in spec['roles']:
            branch = folder/'clips'/name
            branch.mkdir(parents=True)
            cameras[name] = baseline.ActionVideo(snapshot,'switch',branch,horizon*.02)
            cameras[name].sample()
        np.testing.assert_array_equal(cpu(env.obs_buf['actor']),cpu(obs['actor']))
        prefix,prefix_commands = [take(0)],[]
        publish()
        stop = None
        for k in range(1,shared+1):
            prefix_commands.append(step(k))
            prefix.append(take(k))
            stop = stop_boundary(seq,k,p)
            for camera in cameras.values():
                record(camera)
            if k%25==0:
                assert monotonic()-start<spec['maximum_wall_seconds']
                publish()
            if stop is not None:
                break
        save_rows(folder/'shared-trace.npz',prefix)
        save_rows(folder/'shared-commands.npz',prefix_commands)
        if stop is not None:
            endpoint.save(horizon+1)
            write(folder/'shared-stop.json',dict(stop=stop,clips={name:clip_for(name) for name in cameras}))
            publish('stopped-review-required')
            return
        eligible = alive&(seq.phase==FORWARD)
        assert int(eligible.sum())>=p['minimum_continuous_successes']
        assert (seq.forward_at[eligible]+p['forward_controls']>=shared+spec['recovery_controls']).all()
        saved_contacts = list(endpoint.rows)
        controller = controller_snapshot(seq)
        h = digest(signature(controller))
        np.savez_compressed(folder/'fork-controller.npz',**controller)
        fork = RollEntrySnapshot(env,policy,p['maximum_warp_snapshot_bytes'])
        receipt = fork.verify()
        assert receipt['counters']==[shared,shared*5]
        forward_worlds = np.flatnonzero(eligible)
        write(folder/'fork.json',dict(physics=receipt,field_names=sorted(fork.expected),controller_sha256=h,
                                     forward_worlds=forward_worlds.tolist()))
        first,results = None,{}
        for role,magnitude in spec['roles'].items():
            branch = folder/'clips'/role
            restored = fork.restore()
            restore_controller(seq,controller)
            assert restored==receipt and digest(signature(controller_snapshot(seq)))==h
            write(branch/'restored-fork.json',dict(physics=restored,controller_sha256=h))
            endpoint.rows,endpoint.folder,endpoint.first_mismatches = list(saved_contacts),branch,{}
            alive[:] = prefix[-1]['active']
            before = cpu(env.sim.data.qvel)
            np.testing.assert_array_equal(before,prefix[-1]['qvel'])
            injection = apply_lateral_push(env,eligible,magnitude)
            after = cpu(env.sim.data.qvel)
            observed = fork.live_signatures()
            changed = [name for name in fork.expected if observed[name]!=fork.expected[name]]
            assert changed==(['warp.qvel'] if magnitude else []),changed
            write(branch/'injection.json',dict(changed_fields=changed,magnitude_m_s=magnitude,control=shared))
            states,commands,motor,first_velocity = [prefix[-1]],[],[],[]
            stop = None

            def first_substeps(*args,**kwargs):
                if not motor:
                    first_velocity.append(cpu(env.sim.data.qvel))
                motor.append(cpu(env.sim.data.ctrl))
                return original_step(*args,**kwargs)

            try:
                for k in range(shared+1,horizon+1):
                    if k==shared+1:
                        env.sim.step = first_substeps
                    commands.append(step(k))
                    env.sim.step = original_step
                    if k==shared+1:
                        assert len(motor)==5 and len(first_velocity)==1
                        np.testing.assert_array_equal(first_velocity[0],after)
                        row = commands[-1]
                        values = tuple(row[name] for name in ('observed','actor_input','raw_policy_actions',
                                                             'feedback_yaw','applied_targets'))+(motor[0],)
                        if first is None:
                            first = values
                        for value,expected in zip(values,first,strict=True):
                            np.testing.assert_array_equal(value,expected)
                        np.savez_compressed(branch/'injection.npz',**injection,before=before,after=after,
                                            first_substep_qvel=first_velocity[0],substep_motor=np.stack(motor))
                    states.append(take(k))
                    record(cameras[role])
                    stop = stop_boundary(seq,k,p)
                    if k==shared+spec['recovery_controls'] and stop is None:
                        window = {key:np.stack([row[key] for row in states]) for key in ('active','velocity','joint_envelope_ok')}
                        recovered = eligible&recovery_window(window,0,p)
                        if int(recovered.sum())<p['minimum_continuous_successes']:
                            stop = dict(reason='two-second-recovery-threshold-impossible',control=k,
                                        possible=int(recovered.sum()),required=p['minimum_continuous_successes'])
                    if (k-shared)%25==0:
                        assert monotonic()-start<spec['maximum_wall_seconds']
                        publish()
                    if stop is not None:
                        break
                clip = clip_for(role)
                results[role] = dict(endpoint=k,controls=k-shared,stop=stop,video=clip)
                if clip['complete_clip']:
                    view['latest_clip'] = clip['path']
                publish()
            finally:
                env.sim.step = original_step
                save_rows(branch/'trace.npz',states)
                save_rows(branch/'commands.npz',commands)
                endpoint.save(horizon+1)
        total = shared+sum(row['controls'] for row in results.values())
        assert env.guard_steps==total<=spec['maximum_unique_controls'] and len(env.guard_resets)==1
        write(folder/'evaluation.json',dict(status='collected-review-required',protocol_sha256=sha(PLAN),
            shared=shared,controls=total,transitions=total*worlds,substeps=total*5,physical_resets=1,
            branches=results,learning_updates=0,heldout_used=False))
        with (folder/'README.md').open('a',encoding='utf-8') as stream:
            stream.write('\n'+' · '.join(f'[{LABELS[name]}原片]({row["video"]["path"]})' for name,row in results.items())+'\n')
    except BaseException:
        publish('stopped-review-required',frame=False)
        raise
    finally:
        for camera in reversed(list(cameras.values())):
            camera.close()
        snapshot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','sample'))
    args = parser.parse_args()
    if args.command=='freeze':
        return freeze()
    p = checked()
    with patch.multiple(baseline,PLAN=PLAN,OUT=OUT,PAGE=PAGE,checked=lambda:p,collect=collect),torch.no_grad():
        baseline.run(False)


if __name__=='__main__':
    main()
