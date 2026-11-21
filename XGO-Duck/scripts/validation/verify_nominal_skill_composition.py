"""Read back the saved route against the component action and continuity gates."""

from hashlib import sha256

import imageio_ffmpeg
import numpy as np

from nominal_skill_composition_controller import COMMAND, FAILED, ROLLING, ROLL_STANDING
from roll_stand_walk_sequence import FORWARD as POST_ROLL_READY, STATE_FIELDS, Sequence
from routed_getup_sequence import GETUP, STAND
from routed_seven_command_score import command_score, recovery_and_standing
from run_nominal_skill_composition import KEY, PLAN, PROJECT, REPORT, ROOT, checked, onnx_session, read, rel, sha, write
from standup_metrics import projected_gravity
from verify_routed_turn_feedback_pair import PHYSICAL, trace as read_trace


def replay_roll(plan, trace):
    d=plan['composition']
    seq=Sequence(plan['roll_route'])
    first,start,roll_end,end = (d[name] for name in ('getup_to_stand_control','roll_start_control',
                                                     'roll_deadline_control','roll_to_base_stand_control'))
    if len(trace['mode'])<start:
        return None,None
    eligible=trace['mode'][start-1]==STAND
    roll_at_deadline=None
    for control in range(first,min(end,len(trace['mode']))+1):
        state={key:trace[key][control-1] for key in STATE_FIELDS}
        if control>=start:
            state['active'] = state['active'] & eligible
        seq.update(control-first,state)
        np.testing.assert_array_equal(seq.phase,trace['roll_phase'][control-1])
        if control==roll_end:
            roll_at_deadline=(seq.stand_at>=0).copy()
    ready=(seq.phase==POST_ROLL_READY) & (seq.forward_at>=0)
    if len(trace['mode'])>=end:
        np.testing.assert_array_equal(ready,trace['roll_ready'][end-1])
    return roll_at_deadline,ready if len(trace['mode'])>=end else None


def score_commands(spec, trace, physical, upstream, folder):
    d, seven, walking = (spec[key] for key in ('composition','seven_command','walking'))
    start=d['command_entry_control']
    ready=trace['mode'][start]==COMMAND
    np.testing.assert_array_equal(trace['mode'][start:]==COMMAND,
                                  np.broadcast_to(ready,trace['mode'][start:].shape))
    commands, complete = {},np.zeros(len(ready),bool)
    post = slice(start,None)
    joints = physical['joint_positions'][post]
    low,high = np.asarray(walking['joint_limits']).T
    tol = np.deg2rad(walking['criteria']['joint_tolerance_deg'])
    for name,assigned in seven['groups'].items():
        ids = np.asarray(assigned)[ready[assigned]]
        q = physical['quaternion'][post,ids]
        fallen = -projected_gravity(q)[...,2] < seven['minimum_root_up_z_for_walk_survival']
        active = np.ones(fallen.shape,bool)
        active[1:] = np.logical_and.accumulate(~fallen[:-1],axis=0)
        positions = joints[:,ids]
        envelope = ((positions>=low-tol)&(positions<=high+tol)).all(axis=-1)
        sample=dict(active=active,xy=trace['qpos'][post,ids,:2],height=physical['height'][post,ids],
            velocity=trace['velocity'][post,ids],yaw_rate=trace['angular_velocity'][post,ids,2],
            joint_positions=positions,root_quaternion=q,foot_positions=trace['foot_positions'][post,ids],
            foot_speed=trace['foot_speed'][post,ids],contact=trace['original_feet'][post,ids],
            terminated=trace['terminated'][post,ids],fell_over=fallen,nan_state=np.zeros_like(active),
            out_of_bounds=np.zeros_like(active),joint_envelope_ok=envelope)
        result=command_score(sample,trace['qpos'][start-1,ids,:2],name,dict(walking,groups={name:ids}))
        complete[ids] = upstream[ids] & [row['success'] for row in result['episode_scores']]
        commands[name]=dict(assigned=len(assigned),entered=len(ids),complete=int(complete[assigned].sum()),
            minimum_complete=seven['minimum_by_command'][name],original_command_score=result)
    return commands,complete


def check_boundaries(plan, folder, record, trajectory):
    d=plan['composition']
    switches={d['getup_to_stand_control']:'getup-to-stand',d['roll_start_control']:'stand-to-roll',
              d['roll_to_base_stand_control']:'official-stand-to-base',d['command_entry_control']:'base-to-command'}
    models,homes=plan['models'],plan['policy_home']
    observed_home=np.asarray(plan['environment']['model_home'],np.float32)
    report={}
    for crumb in record['boundaries']:
        before=crumb['control']
        path=folder/crumb['file']
        assert sha(path)==crumb['sha256']
        with np.load(path,allow_pickle=False) as item:
            row=dict(item)
        np.testing.assert_array_equal(row['actual_target'],row['expected_articulation_target'])
        np.testing.assert_array_equal(row['mode'],trajectory['mode'][before])
        assert row['motor'].shape==(5,d['worlds'],14) and np.isfinite(row['motor']).all()
        if before not in switches:
            continue
        modes=row['mode']
        if before==d['roll_start_control']:
            selected=modes==ROLLING
            assert not row['bridge'].any() and not row['actor_input'][selected,34:48].any()
            assignments={'official-forward-roll':selected}
        elif before==d['command_entry_control']:
            selected=modes==COMMAND
            groups=plan['seven_command']
            roles={'base':'stand-and-base','official':'forward-official','ppo-candidate':'turn-right'}
            assignments={role:selected & np.isin(np.arange(len(modes)),
                [i for name,ids in groups['groups'].items() if roles[groups['routes'][name]]==role for i in ids])
                for role in roles.values()}
            assert not row['actor_input'][assignments['forward-official']|assignments['turn-right'],34:48].any()
        else:
            selected=modes==STAND
            assert not row['actor_input'][selected,34:48].any()
            assignments={'stand-and-base':selected}
        changed=float(abs(row['actual_target'][selected]-row['previous_target'][selected]).max(initial=0.))
        if before!=d['roll_start_control']:
            assert row['bridge'][selected].all() and changed<=.0400002
        errors={}
        for role,use in assignments.items():
            indices=np.flatnonzero(use)[:32]
            if not len(indices):
                continue
            actual_home=np.asarray(homes[role],np.float32)
            obs=row['actor_input'][indices]
            np.testing.assert_allclose(obs[:,6:20]+actual_home,row['observed'][indices,6:20]+observed_home,atol=2e-7,rtol=0)
            session=onnx_session(ROOT/models[role]['model'])
            actual=np.stack([session.run(None,{'obs':x[None]})[0][0] for x in obs])
            np.testing.assert_allclose(actual,row['raw'][indices],atol=2e-5,rtol=2e-5)
            errors[role]=float(abs(actual-row['raw'][indices]).max())
        report[switches[before]]=dict(entered=int(selected.sum()),maximum_target_step_rad=changed,
            actual_articulation_target_exact=True,first_substeps=5,onnx_model_max_abs_error=errors)
    return report


def check_video(folder, record, trajectory, initial):
    video=record['video']
    path=folder/video['file']
    steps=video['common_steps']
    assert sha(path)==video['sha256'] and steps[0]==0 and steps[-1]==record['controls']
    assert video['full_action']==(record['status']=='collected') and not video['extra_physics_steps']
    snapshots={item['common_step']:item['qpos_sha256'] for item in read(folder/'recording-states.json')}
    for t in steps:
        qpos=initial['qpos'][0] if not t else trajectory['qpos'][t-1,0]
        assert sha256(qpos.tobytes()).hexdigest()==snapshots[t]
    frames=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24')
    meta=next(frames)
    assert meta['fps']==25 and sum(1 for _ in frames)==len(steps)==video['frames']
    return dict(path=rel(path),sha256=video['sha256'],frames=len(steps),actual_seconds=steps[-1]*.02,
                full_route_duration=video['full_action'],camera_command=record['camera_command'],camera_world=0)


def score():
    project,plan=checked('score-pending')
    state=project['goal_execution'][KEY]
    folder=ROOT/plan['record']
    path=folder/'collection.json'
    assert sha(path)==state['collection_sha256'] and sha(folder/'protocol.json')==sha(PLAN)
    record=read(path)
    d=plan['composition']
    controls=record['controls']
    assert record['resets']==1 and record['physics_substeps']==5*controls and record['transitions']==d['worlds']*controls
    detail=read(folder/'endpoint-contact.json')
    endpoint=folder/'endpoint-contact.npz'
    assert detail['complete'] and detail['controls']==controls+1 and not detail['first_mismatches'] and sha(endpoint)==detail['sha256']
    fields=tuple(dict.fromkeys((*PHYSICAL,*STATE_FIELDS,'roll_phase','roll_ready','base_stand_reads','original_feet')))
    trace=read_trace(folder,record['chunks'],1,controls,fields,sha)
    with np.load(endpoint,allow_pickle=False) as contacts:
        np.testing.assert_array_equal(contacts['fresh'],contacts['constraint_points']>0)
        np.testing.assert_array_equal(contacts['fresh'][1:],np.concatenate((trace['ground_contact'][...,None],
            trace['head_contact'][...,None],trace['foot_contact']),axis=-1))
    assert sha(folder/'initial.npz')==record['initial_sha256']
    with np.load(folder/'initial.npz',allow_pickle=False) as loaded:
        initial=dict(loaded)
    boundaries=check_boundaries(plan,folder,record,trace)
    video=check_video(folder,record,trace,initial)
    roll, rolling_stand = replay_roll(plan,trace)
    getup=standing=details=None
    physical=None
    if controls>=d['roll_start_control']:
        assert sha(folder/'release-poses.npz')==record['release_sha256']
        joints=trace['qpos'][...,record['qpos_joint_addresses']]
        limits=np.asarray(plan['environment']['joint_limits'])
        tol=np.deg2rad(plan['environment']['applied_joint_limit_tolerance_deg'])
        joint_ok=((joints>=limits[:,0]-tol)&(joints<=limits[:,1]+tol)).all(axis=-1)
        physical=dict(trace,foot_contact=trace['original_feet'],joint_positions=joints,joint_envelope_ok=joint_ok)
        getup,standing,details=recovery_and_standing(plan,folder,trace,initial,physical)
    complete=commands=None
    eligible=None
    if controls>=d['command_entry_control']:
        eligible=getup & standing & rolling_stand & (trace['base_stand_reads'][d['command_entry_control']-1]>=25)
    if controls==d['maximum_controls']:
        commands,complete=score_commands(plan,trace,physical,eligible,folder)
    seven=plan['seven_command']
    passed=bool(complete is not None and details['functional_coverage_gate'] and roll.sum()>=plan['roll_route']['minimum_roll_completion']
        and complete.sum()>=seven['minimum_overall'] and all(row['complete']>=row['minimum_complete'] and
            row['original_command_score']['functional_development_pass'] for row in commands.values()))
    result=dict(protocol_sha256=sha(PLAN),collection_sha256=sha(path),status=record['status'],reason=record.get('reason'),
        controls=controls,physical_substeps=record['physics_substeps'],transitions=record['transitions'],worlds=d['worlds'],
        release_counts=record.get('actual_release_counts'),getup=details,pre_roll_standing=None if standing is None else int(standing.sum()),
        roll_complete=None if roll is None else int(roll.sum()),roll_gate=plan['roll_route']['minimum_roll_completion'],
        post_roll_standing=None if rolling_stand is None else int(rolling_stand.sum()),
        possible_completions_after_upstream=None if eligible is None else int(eligible.sum()),
        command_admission=record.get('command_admission'),commands=commands,
        complete=None if complete is None else int(complete.sum()),minimum_complete=seven['minimum_overall'],
        passed_development_gate=passed,verified_boundaries=boundaries,video=video,learning_updates=0,hardware_io=False)
    dest=REPORT/'analysis.json'
    assert not dest.exists()
    write(dest,result)
    state.update(status='reviewed',result=rel(dest),result_sha256=sha(dest),combined_physics_verified=True,
                 passed_development_gate=passed,actual_controls=controls)
    project['staged_training_request'].update(current_protocol_execution='nominal-skill-composition-action-reviewed',current_evidence=rel(dest))
    write(PROJECT,project)
    print('NOMINAL_SKILL_COMPOSITION_SCORED',sha(dest),controls,result['roll_complete'],result['complete'],passed,flush=True)
