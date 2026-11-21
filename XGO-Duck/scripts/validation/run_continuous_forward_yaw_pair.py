"""Compare forward yaw feedback caps after one shared continuous prefix."""
import argparse
from datetime import datetime
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import numpy as np
import torch

from check_continuous_forward_fork_state import controller_snapshot, restore_controller
from getup_release_fork_state import digest, signature
from record_io import write
from roll_entry_impossibility import maximum_possible
from roll_entry_snapshot import RollEntrySnapshot
from roll_stand_walk_sequence import FAILED, FORWARD, Sequence
from run_continuous_roll_projected_entry import ProjectedRollPolicy, baseline, checked as checked_parent
from run_roll_entry_isolation import save_rows
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/continuous_forward_yaw_pair.json'
DRAFT = PLAN.with_name('continuous_forward_yaw_pair_draft.json')
OUT = ROOT/'records/reports/continuous-forward-yaw-pair'
PAGE = ROOT/'records/continuous-forward-yaw-live.html'
ROLES = ('reference', 'wider-input')
LABELS = ('原反馈上限', '扩大反馈上限')
PRIOR = ROOT/'records/reports/continuous-roll-projected-entry'
cpu = baseline.cpu


def freeze():
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    draft, parent = read(DRAFT), checked_parent()
    assert draft['status']=='draft-not-executable' and not draft['heldout_used'] and not draft['hardware_io']
    assert not draft['learning_updates'] and not draft['automatic_extension'] and not draft['automatic_retry']
    evidence = ROOT/draft['parent_result']
    preflight = PRIOR/'forward-yaw-fork-controller-preflight.json'
    proof = read(preflight)
    assert proof['replayed_next_boundary_exact'] and proof['controller_fields']==15
    archive = PRIOR/'forward-yaw-fork-controller.npz'
    assert sha(archive)==proof['archive_sha256']
    for source,expected in proof['sources'].items():
        assert sha(ROOT/source)==expected,source
    parent_result = read(evidence)
    assert parent_result['status']=='passed-independent-verification' and not parent_result['passed_development_gate']
    assert sha(ROOT/draft['parent_protocol'])==parent_result['protocol_sha256']
    assert draft['roles']==dict(zip(ROLES,(.12,.15),strict=True))
    assert parent['feedback']['maximum_yaw_input_rad_s']==draft['roles']['reference']
    assert parent['feedback']['maximum_joint_correction_rad']==draft['joint_correction_cap_rad']==.03
    assert parent['walking']['criteria']['yaw_error_max']==draft['gate']['original_forward_yaw_error_max_rad_s']==.2
    assert parent['minimum_continuous_successes']==draft['gate']['minimum_complete_sequences']==116
    assert parent['maximum_controls']==draft['maximum_controls_per_complete_sequence']==1550
    assert draft['maximum_unique_controls']==2*parent['maximum_controls']
    assert draft['maximum_control_transitions']==draft['maximum_unique_controls']*draft['worlds']
    assert draft['maximum_physics_substeps']==draft['maximum_unique_controls']*5
    assert draft['worlds']==parent['worlds']==128 and parent['forward_controls']==1000
    seed = draft['development_seed']
    assert not any(f'_s{seed}_' in path.name for path in (ROOT/'records/evaluations').iterdir())
    sources = dict(parent['frozen_sha256'])
    paths = [DRAFT, ROOT/draft['parent_protocol'], evidence, ROOT/draft['localization'],
             ROOT/draft['static_range_screen'], preflight, archive, Path(__file__)]
    paths += [ROOT/'scripts/validation'/name for name in (
        'check_continuous_forward_fork_state.py','verify_continuous_forward_yaw_pair.py','roll_entry_snapshot.py',
        'getup_release_scene_state.py','getup_release_fork_state.py','run_roll_entry_isolation.py')]
    sources.update({path.relative_to(ROOT).as_posix():sha(path) for path in paths})
    p = dict(parent,status='frozen-before-dynamics',stage=draft['stage'],seed=seed,roll=dict(parent['roll'],seed=seed),
        run_id=f'duck_switch_forward-yaw-pair_development_s{seed}_{datetime.now(baseline.TZ):%Y%m%dT%H%M%S}',
        pair=draft,maximum_warp_snapshot_bytes=268435456,frozen_sha256=sources)
    OUT.mkdir()
    write(PLAN,p)
    write(OUT/'preflight.json',dict(status='passed-cpu-only',protocol_sha256=sha(PLAN),frozen_files=len(sources),
        controller_archive_sha256=sha(archive),physical_substeps=0,observation=PAGE.relative_to(ROOT).as_posix()))
    print('CONTINUOUS_FORWARD_YAW_PAIR_FROZEN',p['run_id'],sha(PLAN),flush=True)


def checked():
    # The parent checker pins the workspace dependency and installed library sources too.
    from s1_reference_inputs import WORK

    p = read(PLAN)
    assert p['status']=='frozen-before-dynamics' and p['projected_roll_without_slew']
    assert not any(p[key] for key in ('heldout_used','hardware_io','learning_updates','automatic_retry','automatic_extension'))
    for root,key in ((ROOT,'frozen_sha256'),(WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
        for path,expected in p[key].items():
            assert sha(root/path)==expected,path
    return p


def stop_boundary(sequence,k,p):
    if sequence.roll_begin<=k<=sequence.roll_end:
        possible,certified = maximum_possible(sequence.roll_latch,p['roll_controls'])
        if possible<p['minimum_roll_completion']:
            return dict(reason='original-roll-threshold-impossible',control=k,possible=possible,
                        certified=certified,required=p['minimum_roll_completion'])
    possible = int(((sequence.phase!=FAILED)|((sequence.forward_at>=0)&
                    (k>sequence.forward_at+p['forward_controls']))).sum())
    if possible<p['minimum_continuous_successes']:
        return dict(reason='sequence-threshold-impossible',control=k,possible=possible,
                    required=p['minimum_continuous_successes'])
    return None


def collect(env,obs,p,folder,endpoint,networks,homes,sim_home,limits):
    worlds, horizon = p['worlds'], p['maximum_controls']
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
    prefix, prefix_commands, results = [], [], {}
    started = sample_time = monotonic()
    sample_count = 0
    original_step = env.sim.step
    active_role = 'shared'
    view = dict(run_id=p['run_id'],maximum_controls=p['pair']['maximum_unique_controls'],learning_updates=0,updates=[])
    snapshot = baseline.Snapshot(env,folder)
    render = snapshot.render
    pixels, last_render = None, None
    cameras = {}

    def render_boundary():
        # The two encoders share the initial physical frames. Once forked,
        # the role is part of the key even for the same simulation clock.
        nonlocal pixels,last_render
        key = (active_role,env.common_step_counter)
        if key!=last_render:
            pixels,last_render = render(),key
        return pixels

    snapshot.render = render_boundary

    def publish(status='running',frame=True):
        nonlocal sample_time,sample_count
        count, now = env.guard_steps,monotonic()
        if count>sample_count:
            view['updates'].append(dict(control=count,transitions_per_second=(count-sample_count)*worlds/max(.01,now-sample_time)))
            sample_time,sample_count = now,count
        if frame:
            raw = snapshot.capture()['file']
            name = f'frame-{active_role}-{env.common_step_counter}.jpg'
            (folder/raw).replace(folder/name)
            view['frame'] = name
        label = '共同站立、滚翻、落地' if active_role=='shared' else LABELS[ROLES.index(active_role)]
        view.update(status=status,purpose=f'连续前进偏航对照 · {label}',updated_at=datetime.now(baseline.TZ).isoformat(),
                    controls=count,transitions=count*worlds)
        with patch.object(baseline.live,'PAGE',PAGE):
            baseline.live.publish(folder,view)
        row = f'CONTINUOUS_FORWARD_YAW {active_role} unique={count} timeline={env.common_step_counter} active={alive.sum()}'
        print(row,flush=True)
        with (folder/'execution.log').open('a',encoding='utf-8') as file:
            file.write(row+'\n')

    def observe():
        row = baseline.capture(env,endpoint,alive,joints,head,feet,bounds,tolerance)
        assert all(np.isfinite(value).all() for value in row.values())
        return row

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

    def take(k):
        state = observe()
        seq.update(k,state)
        return dict(state,phase_next=seq.phase.copy())

    def record(camera):
        if camera.writer is not None:
            camera.sample() if alive[0] else camera._sample_terminal()

    def clip_for(role):
        camera = cameras[role]
        camera._sample_terminal() if camera.writer is not None else None
        clip, = camera.clips
        path = folder/'clips'/role/clip['file']
        return dict(clip,path=path.relative_to(folder).as_posix(),sha256=sha(path))

    (folder/'README.md').write_text('# 连续前进偏航对照\n\n'
        '两组共用站立、滚翻、落地的物理轨迹，在首个场景前进前分支。录像从共同起点开始。\n\n'
        '[冻结方案](protocol.json) · [执行日志](execution.log) · [状态](state.json)\n',encoding='utf-8')
    try:
        for role in ROLES:
            branch = folder/'clips'/role
            branch.mkdir(parents=True)
            cameras[role] = baseline.ActionVideo(snapshot,'switch',branch,horizon*.02)
            cameras[role].sample()
        np.testing.assert_array_equal(cpu(env.obs_buf['actor']),cpu(obs['actor']))
        prefix.append(take(0))
        publish()
        k,stop = 0,None
        while not (seq.phase==FORWARD).any() and stop is None and k<horizon:
            k += 1
            prefix_commands.append(step(k))
            prefix.append(take(k))
            stop = stop_boundary(seq,k,p)
            for camera in cameras.values():
                record(camera)
            if k%25==0:
                assert monotonic()-started<p['maximum_sampling_wall_seconds']
                publish()
        save_rows(folder/'shared-trace.npz',prefix)
        save_rows(folder/'shared-commands.npz',prefix_commands)
        shared = k
        if stop is not None or not (seq.phase==FORWARD).any():
            endpoint.save(horizon+1)
            clips = {role:clip_for(role) for role in ROLES}
            write(folder/'shared-stop.json',dict(control=k,stop=stop,clips=clips))
            publish('stopped-review-required')
            return
        saved_endpoints = list(endpoint.rows)
        saved_controller = controller_snapshot(seq)
        controller_hash = digest(signature(saved_controller))
        np.savez_compressed(folder/'fork-controller.npz',**saved_controller)
        fork = RollEntrySnapshot(env,policy,p['maximum_warp_snapshot_bytes'])
        captured = fork.verify()
        assert captured['counters']==[shared,shared*5]
        write(folder/'fork.json',dict(physics=captured,field_names=sorted(fork.expected),controller_sha256=controller_hash,
            forward_worlds=np.flatnonzero(seq.phase==FORWARD).tolist()))
        first = None
        for role in ROLES:
            active_role = role
            branch = folder/'clips'/role
            restored = fork.verify() if role==ROLES[0] else fork.restore()
            assert restored==captured
            restore_controller(seq,saved_controller)
            assert digest(signature(controller_snapshot(seq)))==controller_hash
            write(branch/'restored-fork.json',dict(physics=restored,controller_sha256=controller_hash))
            endpoint.rows,endpoint.folder,endpoint.first_mismatches = list(saved_endpoints),branch,{}
            alive[:] = prefix[-1]['active']
            current = dict(observe(),phase_next=seq.phase.copy())
            for key,value in current.items():
                np.testing.assert_array_equal(value,prefix[-1][key],err_msg=key)
            policy.feedback.cap = p['pair']['roles'][role]
            k,stop = shared,None
            states,commands,motor = [current],[],[]

            def delayed_motor(*args,**kwargs):
                motor.append(cpu(env.sim.data.ctrl))
                return original_step(*args,**kwargs)

            try:
                while k<horizon and stop is None:
                    k += 1
                    if k==shared+1:
                        env.sim.step = delayed_motor
                    commands.append(step(k))
                    env.sim.step = original_step
                    if k==shared+1:
                        assert len(motor)==5
                        row = commands[-1]
                        actual = (row['observed'],row['actor_input'],row['raw_policy_actions'],motor[0])
                        if first is None:
                            first = actual
                        for value,expected in zip(actual,first,strict=True):
                            np.testing.assert_array_equal(value,expected)
                        np.savez_compressed(branch/'first-motor.npz',substeps=np.stack(motor))
                    states.append(take(k))
                    stop = stop_boundary(seq,k,p)
                    record(cameras[role])
                    if k%25==0:
                        assert monotonic()-started<p['maximum_sampling_wall_seconds']
                        publish()
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
        count = shared+sum(row['controls'] for row in results.values())
        assert env.guard_steps==count<=p['pair']['maximum_unique_controls'] and len(env.guard_resets)==1
        write(folder/'evaluation.json',dict(status='collected-review-required',protocol_sha256=sha(PLAN),shared=shared,
            controls=count,transitions=count*worlds,substeps=count*5,physical_resets=1,branches=results,
            learning_updates=0,heldout_used=False))
        with (folder/'README.md').open('a',encoding='utf-8') as file:
            file.write('\n'+' · '.join(f'[{label}原片]({results[role]["video"]["path"]})'
                       for role,label in zip(ROLES,LABELS,strict=True))+'\n')
    except BaseException:
        publish('stopped-review-required',frame=False)
        raise
    finally:
        for camera in reversed(list(cameras.values())):
            camera.close()
        snapshot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','zero','sample'))
    args = parser.parse_args()
    if args.command=='freeze':
        return freeze()
    p = checked()
    with patch.multiple(baseline,PLAN=PLAN,OUT=OUT,PAGE=PAGE,checked=lambda:p,collect=collect),torch.no_grad():
        baseline.run(args.command=='zero')


if __name__=='__main__':
    main()
