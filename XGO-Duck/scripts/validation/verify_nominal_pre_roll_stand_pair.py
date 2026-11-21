"""Verify the matched snapshot, the single policy change, and the original gates."""

import numpy as np

from collect_nominal_pre_roll_stand_pair import CANDIDATE, REFERENCES, ROLES
from roll_stand_walk_sequence import STATE_FIELDS
from routed_getup_sequence import STAND
from routed_seven_command_score import recovery_and_standing
from run_nominal_pre_roll_stand_pair import KEY, PLAN, PROJECT, REPORT, ROOT, checked, read, rel, sha, write
from standup_metrics import projected_gravity
from verify_nominal_skill_composition import (PHYSICAL, check_boundaries, check_video, onnx_session, read_trace,
    replay_roll, score_commands)


def standing_checks(plan, folder, record, trajectory):
    d=plan['composition']
    pair=plan['standing_pair']
    start,end=pair['candidate_interval_controls']
    path=folder/'pre-roll-standing.npz'
    assert sha(path)==record['standing_sha256']
    with np.load(path,allow_pickle=False) as stored:
        block=dict(stored)
    mode=block['mode']
    np.testing.assert_array_equal(mode,trajectory['mode'][start:end])
    assert mode.shape==(end-start,d['worlds'])
    stands=mode[0]==STAND
    assert stands.any() and np.all((mode==STAND)==stands)
    inputs,raw,targets=(block[key][:,stands] for key in ('actor_input','raw','actual_target'))
    assert not inputs[:,:,48:].any()
    np.testing.assert_array_equal(inputs[1:,:,34:48],raw[:-1])
    candidate=folder.name==CANDIDATE
    model='forward-official' if candidate else 'stand-and-base'
    if candidate:
        assert not inputs[0,:,34:48].any()
    start_record,=(row for row in record['boundaries'] if row['control']==start)
    first=folder/start_record['file']
    assert sha(first)==start_record['sha256']
    with np.load(first,allow_pickle=False) as f:
        beginning=dict(f)
    assert not beginning['bridge'].any()
    np.testing.assert_array_equal(inputs[0],beginning['actor_input'][stands])
    np.testing.assert_array_equal(raw[0],beginning['raw'][stands])
    np.testing.assert_array_equal(targets[0],beginning['actual_target'][stands])
    home=np.asarray(plan['policy_home'][model],np.float32)
    observed_home=np.asarray(plan['environment']['model_home'],np.float32)
    np.testing.assert_allclose(inputs[0,:,6:20]+home,beginning['observed'][stands,6:20]+observed_home,atol=2e-7,rtol=0)
    limits=np.asarray(plan['environment']['joint_limits'],np.float32)
    np.testing.assert_allclose(targets,np.clip(raw+home,limits[:,0],limits[:,1]),atol=2e-7,rtol=0)
    session=onnx_session(ROOT/plan['models'][model]['model'])
    indices=np.linspace(0,int(stands.sum())-1,8,dtype=int)
    sample=inputs[:,indices].reshape(-1,inputs.shape[-1])
    actual=np.stack([session.run(None,{'obs':item[None]})[0][0] for item in sample])
    expected=raw[:,indices].reshape(-1,raw.shape[-1])
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    return dict(actor=model,standing_worlds=int(stands.sum()),verified_controls=end-start,zero_command_tail=True,
        first_history_empty=bool(not inputs[0,:,34:48].any()),subsequent_history_is_raw=True,
        onnx_samples=len(sample),onnx_max_abs_error=float(abs(actual-expected).max())),beginning


def verify_branch(plan, folder, record, prefix, common, initial, fork):
    d,seven=plan['composition'],plan['seven_command']
    shared=plan['standing_pair']['common_snapshot_control']
    controls=record['controls']
    assert controls in (d['roll_deadline_control'],d['command_entry_control'],d['maximum_controls'])
    restored=folder/'restored-fork.json'
    assert sha(restored)==record['restore_sha256']
    assert read(restored)=={key:fork[key] for key in ('physical','controller_sha256')}
    branch=read_trace(folder,record['chunks'],shared+1,controls,tuple(prefix),sha)
    trace={name:np.concatenate((prefix[name],branch[name])) for name in prefix}
    detail=read(folder/'endpoint-contact.json')
    endpoint=folder/'endpoint-contact.npz'
    assert detail['complete'] and detail['controls']==controls+1 and not detail['first_mismatches']
    assert detail['zero_matches_standalone'] and sha(endpoint)==detail['sha256']==record['contact_sha256']
    with np.load(endpoint,allow_pickle=False) as contacts:
        np.testing.assert_array_equal(contacts['fresh'],contacts['constraint_points']>0)
        np.testing.assert_array_equal(contacts['fresh'][1:],np.concatenate((trace['ground_contact'][...,None],
            trace['head_contact'][...,None],trace['foot_contact']),axis=-1))
    boundaries=check_boundaries(plan,folder.parent/'shared',common,trace)
    boundaries.update(check_boundaries(plan,folder,record,trace))
    standing,first=standing_checks(plan,folder,record,trace)
    video=check_video(folder,dict(record,camera_command=common['camera_command']),trace,initial)
    roll,rolling_stand=replay_roll(plan,trace)
    joints=trace['qpos'][...,common['qpos_joint_addresses']]
    limits=np.asarray(plan['environment']['joint_limits'])
    tol=np.deg2rad(plan['environment']['applied_joint_limit_tolerance_deg'])
    joint_ok=((joints>=limits[:,0]-tol)&(joints<=limits[:,1]+tol)).all(axis=-1)
    physical=dict(trace,foot_contact=trace['original_feet'],joint_positions=joints,joint_envelope_ok=joint_ok)
    getup,base_standing,details=recovery_and_standing(plan,folder.parent,trace,initial,physical)
    candidate=folder.name==CANDIDATE
    if controls==d['roll_deadline_control']:
        assert record['reason']=='original-roll-gate-unreachable' and roll.sum()<plan['roll_route']['minimum_roll_completion']
    arrived=eligible=complete=commands=None
    if controls>=d['command_entry_control']:
        arrived=np.isin(np.arange(d['worlds']),record['command_entry_worlds'])
        assert int(arrived.sum())==len(record['command_entry_worlds'])
        assert record['command_admission']=={name:int(arrived[ids].sum()) for name,ids in seven['groups'].items()}
        eligible=getup & base_standing & rolling_stand & (trace['base_stand_reads'][d['command_entry_control']-1]>=25) & arrived
        if controls==d['command_entry_control']:
            assert record['reason']=='complete-command-gate-unreachable-at-entry'
            assert (int(arrived.sum())<seven['minimum_overall'] or
                    any(record['command_admission'][name]<seven['minimum_by_command'][name] for name in seven['groups']))
    if controls==d['maximum_controls']:
        assert arrived is not None and record['status']=='collected'
        commands,complete=score_commands(plan,trace,physical,eligible,folder.parent)
    passed=bool(complete is not None and details['functional_coverage_gate'] and
        roll.sum()>=plan['roll_route']['minimum_roll_completion'] and complete.sum()>=seven['minimum_overall'] and
        all(row['complete']>=row['minimum_complete'] and row['original_command_score']['functional_development_pass']
            for row in commands.values()))
    start=d['roll_start_control']
    entering=trace['mode'][start-1]==STAND
    g=projected_gravity(trace['quaternion'][start-1,entering])
    tilt=np.rad2deg(np.arccos(np.clip(-g[:,2],-1.,1.)))
    result=dict(candidate=candidate,controls=controls,status=record['status'],reason=record.get('reason'),
        getup=details,pre_roll_standing=int(base_standing.sum()),roll_complete=int(roll.sum()),
        roll_entry_tilt_median_deg=float(np.median(tilt)),roll_entry_tilt_p95_deg=float(np.quantile(tilt,.95)),
        post_roll_standing=None if rolling_stand is None else int(rolling_stand.sum()),
        command_admission=record.get('command_admission'),eligible_at_entry=None if eligible is None else int(eligible.sum()),
        commands=commands,complete=None if complete is None else int(complete.sum()),passed_development_gate=passed,
        standing_policy=standing,verified_boundaries=boundaries,video=video)
    population=dict(roll=roll,command_admission=arrived,complete=complete)
    return result,population,first


def compare(a,b,groups):
    result={}
    for name,x in a.items():
        y=b[name]
        if x is None or y is None:
            result[name]=None
            continue
        gained=x & ~y
        lost=y & ~x
        result[name]=dict(gained=int(gained.sum()),lost=int(lost.sum()),net=int(gained.sum()-lost.sum()),
            net_by_assigned_command={key:int(x[ids].sum()-y[ids].sum()) for key,ids in groups.items()})
    return result


def score():
    project,plan=checked('score-pending')
    state=project['goal_execution'][KEY]
    folder=ROOT/plan['record']
    path=folder/'collection.json'
    assert sha(path)==state['collection_sha256'] and sha(folder/'protocol.json')==sha(PLAN)
    record=read(path)
    d=plan['composition']
    shared=plan['standing_pair']['common_snapshot_control']
    assert record['status']=='collected' and record['actor_weights_unchanged'] and record['resets']==1
    assert set(record['branches'])==set(ROLES) and record['prefix']['controls']==shared
    assert record['controls']==shared+sum(item['controls']-shared for item in record['branches'].values())
    assert record['physics_substeps']==5*record['controls'] and record['transitions']==d['worlds']*record['controls']
    assert record['transitions']<=plan['maximum_control_transitions']
    assert sha(folder/'zero/proof.json')==record['process_local_zero_sha256']
    assert sha(folder/'fork.json')==record['fork_sha256']
    fork=read(folder/'fork.json')
    assert fork['physical']['counters']==[shared,shared*5] and fork['physical']['fields']==len(fork['fields'])
    assert fork['fields']==sorted(fork['field_signatures'])
    assert sha(folder/'initial.npz')==record['initial_sha256'] and sha(folder/'release-poses.npz')==record['release_sha256']
    with np.load(folder/'initial.npz',allow_pickle=False) as loaded:
        initial=dict(loaded)
    fields=tuple(dict.fromkeys((*PHYSICAL,*STATE_FIELDS,'roll_phase','roll_ready','base_stand_reads','original_feet')))
    prefix=read_trace(folder/'shared',record['prefix']['chunks'],1,shared,fields,sha)
    common=dict(record['prefix'],qpos_joint_addresses=record['qpos_joint_addresses'],camera_command=record['camera_command'])
    results,population,first={},{},{}
    for role in ROLES:
        results[role],population[role],first[role]=verify_branch(plan,folder/role,record['branches'][role],prefix,common,initial,fork)
    a,b=REFERENCES
    for field in ('observed','previous_target','mode'):
        for role in (b,CANDIDATE):
            np.testing.assert_array_equal(first[a][field],first[role][field])
    for field in ('actor_input','raw','actual_target'):
        np.testing.assert_array_equal(first[a][field],first[b][field])
    for role in (b,CANDIDATE):
        np.testing.assert_array_equal(first[a]['motor'][0],first[role]['motor'][0])
    groups=plan['seven_command']['groups']
    differences={role:compare(population[CANDIDATE],population[role],groups) for role in REFERENCES}
    reference_repeat=compare(population[b],population[a],groups)
    chosen=results[CANDIDATE]['passed_development_gate']
    result=dict(status='verified-paired-development-evidence',protocol_sha256=sha(PLAN),collection_sha256=sha(path),
        seed=plan['environment']['seed'],worlds=d['worlds'],shared_controls=shared,actual_controls=record['controls'],
        physics_substeps=record['physics_substeps'],transitions=record['transitions'],resets=record['resets'],
        frozen_gates=plan['standing_pair']['original_gates'],verified_fork=fork['physical'],
        exact_reference_first_target=True,exact_first_motor_after_delay=True,
        results=results,matched_candidate_differences=differences,matched_reference_repeat=reference_repeat,
        candidate_eligible_for_independent_development=chosen,reserved_seed_used=False,hardware_io=False,learning_updates=0)
    dest=REPORT/'analysis.json'
    assert not dest.exists()
    write(dest,result)
    state.update(status='reviewed',result=rel(dest),result_sha256=sha(dest),candidate_advanced=chosen)
    project['staged_training_request'].update(current_protocol_execution='nominal-pre-roll-stand-pair-action-reviewed',
        current_evidence=rel(dest))
    write(PROJECT,project)
    print('PRE_ROLL_STAND_PAIR_SCORED',sha(dest),{key:(row['roll_complete'],row['complete']) for key,row in results.items()},flush=True)
