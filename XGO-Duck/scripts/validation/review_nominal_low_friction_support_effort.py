"""Recompute the original get-up and roll-entry gates and read the solved loads."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from getup_coverage_metrics import POSES
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence
from routed_getup_sequence import STAND
from routed_seven_command_score import recovery_and_standing
from standup_metrics import projected_gravity
from verify_getup_anchor_same_release import independent_holds
from verify_nominal_skill_composition import PHYSICAL, check_video, read_trace
from run_nominal_low_friction_support_effort import (
    KEY, PLAN, PROJECT, REPORT, ROOT, STAGE, parent, read, rel, sha, write,
)


def tilt_deg(quat):
    return np.rad2deg(np.arccos(np.clip(-projected_gravity(quat)[...,2], -1., 1.)))


def stats(values):
    values = np.asarray(values)
    if not len(values):
        return None
    lo, mid, hi = np.quantile(values, [.25,.5,.75])
    return dict(q25=float(lo), median=float(mid), q75=float(hi))


def read_windows(plan, raw, folder, trace):
    info = raw['support_effort']
    meta_path = folder/info['path']
    meta = read(meta_path, info['sha256'])
    spec = plan['support_effort']
    assert meta['contact_channels']==['ground','head','left_foot','right_foot']
    assert set(meta['joint_order'])==set(plan['walking']['joint_names']) and len(meta['joint_order'])==14
    limits = np.asarray(meta['solver_torque_limit_nm'], np.float32)
    assert meta['pwm_limit']>0. and meta['maximum_torque_readback_error_nm']<=5e-6
    windows = {}
    for defined, row in zip(spec['windows'],meta['windows'],strict=True):
        first, last = (defined[k]*5 for k in ('start','stop'))
        assert row['window']==defined['name'] and row['reached_substep']==row['planned_last_substep']==last
        assert row['start_substep']==first+1 and row['rows']==last-first
        file = folder/row['file']
        assert sha(file)==row['sha256']
        with np.load(file, allow_pickle=False) as source:
            data = dict(source)
        np.testing.assert_array_equal(data['substep'],np.arange(first+1,last+1))
        for key, cut, controls in (
            ('orientation_before', slice(0,None,5), np.arange(defined['start'],defined['stop'])),
            ('orientation_after', slice(4,None,5), np.arange(defined['start']+1,defined['stop']+1)),
        ):
            np.testing.assert_array_equal(data[key][cut], trace['quaternion'][controls-1], err_msg=key)
        np.testing.assert_array_equal(data['orientation_before'][1:],data['orientation_after'][:-1])
        normal, duty, request, actual = (data[name] for name in ('normal_force_n','voltage_duty',
                        'requested_torque_nm','applied_torque_nm'))
        worlds = plan['composition']['worlds']
        assert normal.shape==(last-first,worlds,4) and duty.shape==request.shape==actual.shape==(last-first,worlds,14)
        assert all(np.isfinite(value).all() for value in data.values())
        assert (normal>=0.).all() and np.all(normal[...,0]+5e-5>=normal[...,1:].sum(axis=-1))
        assert np.abs(duty).max() <= meta['pwm_limit']+1e-6 and (data['supply_v']>0.).all()
        np.testing.assert_allclose(actual, np.clip(request,-limits,limits), atol=5e-6, rtol=0)
        windows[row['window']] = data
    assert sum(len(data['substep']) for data in windows.values())==info['sampled_substeps']
    return meta, windows


def compare_release(early, trace, matches, spec, weight):
    # Compare actual supine scenes using release gravity and first 20 ms response.
    start = spec['windows'][0]['start']
    feet = early['normal_force_n'][...,2:4]
    support = (feet>spec['nominal_load_fraction']*weight).all(axis=-1)&(feet.sum(axis=-1)>=spec['both_feet_fraction']*weight)
    dwell = spec['minimum_load_solves']
    window = np.lib.stride_tricks.sliding_window_view(support,dwell,axis=0).all(axis=-1)
    first = np.where(window.any(axis=0), window.argmax(axis=0)+dwell-1, -1)
    t = tilt_deg(early['orientation_before'])
    pose = projected_gravity(trace['quaternion'][start-1])
    motion = projected_gravity(trace['quaternion'][start])-pose
    cohort_a, cohort_b = matches
    features = np.concatenate((pose, motion), axis=-1)
    cost = np.linalg.norm(features[cohort_a,None]-features[None,cohort_b], axis=-1)
    take_a, take_b = linear_sum_assignment(cost)
    assert len(take_a)==len(cohort_a) and len(set(take_b.tolist()))==len(take_b)

    def event(world):
        idx = first[world]
        if idx<0:
            return dict(world=int(world), confirmed=False)
        aligned = {}
        for ms in (-80,0,80):
            slot = idx+round(ms/(1000*spec['physics_dt']))
            if 0<=slot<len(t):
                aligned[str(ms)] = dict(tilt_deg=float(t[slot,world]),head_n=float(early['normal_force_n'][slot,world,1]),
                                        feet_n=float(feet[slot,world].sum()))
        return dict(world=int(world),confirmed=True,confirmation_elapsed_ms=round((idx+1)*spec['physics_dt']*1000),
                    aligned=aligned)

    pairs = [dict(missed=event(cohort_a[i]),admitted=event(cohort_b[j]),distance=float(cost[i,j])) for i,j in zip(take_a,take_b,strict=True)]
    contrasts = {}
    for ms in (-80,0,80):
        key = str(ms)
        success = [pack for pack in pairs if all(entry.get('aligned',{}).get(key) is not None
                   for entry in (pack['missed'],pack['admitted']))]
        contrasts[key] = dict(pairs=len(success),tilt_missed_minus_admitted_deg=stats([
            item['missed']['aligned'][key]['tilt_deg']-item['admitted']['aligned'][key]['tilt_deg'] for item in success]))
    return dict(matches=pairs,comparisons=contrasts,release_confirmed_missed=int((first[cohort_a]>=0).sum()),
                release_confirmed_admitted=int((first[cohort_b]>=0).sum()),release_missed_count=len(cohort_a),
                release_admitted_count=len(cohort_b),confirmation_missed_ms=stats((first[cohort_a][first[cohort_a]>=0]+1)*4),
                confirmation_admitted_ms=stats((first[cohort_b][first[cohort_b]>=0]+1)*4))


def summarize_settled(data, ids, spec, meta):
    slo, shi = spec['settled_controls']
    in_scope = (data['substep']>5*slo)&(data['substep']<=5*shi)
    support = data['normal_force_n'][in_scope][:,ids]
    duty = data['voltage_duty'][in_scope][:,ids]
    requested = data['requested_torque_nm'][in_scope][:,ids]
    actual = data['applied_torque_nm'][in_scope][:,ids]
    tilt = tilt_deg(data['orientation_before'][in_scope][:,ids])
    supply = data['supply_v'][in_scope][:,ids]
    weight = meta['nominal_weight_n']
    fraction = spec['saturation_of_limit_fraction']
    duty_saturated = abs(duty)>=meta['pwm_limit']*fraction
    torque_saturated = abs(actual)>=np.asarray(meta['solver_torque_limit_nm'])*fraction
    clamped = abs(actual-requested)>5e-6
    limits = spec['nominal_load_fraction']*weight
    records = []
    for i, world in enumerate(ids):
        records.append(dict(world=int(world), head_load_fraction=float(np.median(support[:,i,1])/weight),
            feet_load_fraction=float(np.median(support[:,i,2:].sum(axis=-1))/weight), tilt_deg=float(np.median(tilt[:,i])),
            head_loaded_share=float((support[:,i,1]>limits).mean()),
            all_three_loaded_share=float((support[:,i,1:]>limits).all(axis=-1).mean()),
            voltage_saturation_share_per_joint=duty_saturated[:,i].mean(axis=0).tolist(),
            solver_torque_saturation_share_per_joint=torque_saturated[:,i].mean(axis=0).tolist(),
            pwm_max_fraction_per_joint=(abs(duty[:,i]).max(axis=0)/meta['pwm_limit']).tolist(),
            torque_peak_nm_per_joint=abs(actual[:,i]).max(axis=0).tolist(),
            supply_median_v=float(np.median(supply[:,i]))))
    return dict(worlds=len(ids), substeps_per_world=int(in_scope.sum()), records=records,
        head_load_fraction=stats([row['head_load_fraction'] for row in records]),
        foot_load_fraction=stats([row['feet_load_fraction'] for row in records]),tilt_deg=stats([row['tilt_deg'] for row in records]),
        head_loaded_share=stats([row['head_loaded_share'] for row in records]),
        all_three_loaded_share=stats([row['all_three_loaded_share'] for row in records]),
        pwm_saturated_actuator_intervals=int(duty_saturated.sum()), torque_saturated_actuator_intervals=int(torque_saturated.sum()),
        solver_clipped_actuator_intervals=int(clamped.sum()), peak_pwm_fraction=float(abs(duty).max(initial=0.)/meta['pwm_limit']),
        total_actuator_intervals=int(duty.size))


def analyze(save):
    project,plan = read(PROJECT),read(PLAN)
    node, task = project['goal_execution'][KEY],project['staged_training_request']
    assert parent.allowed(project,STAGE) and not task['simulation_running'] and node['status'] in ('score-pending','reviewed')
    assert sha(PLAN)==node['protocol_sha256']
    parent.base.shared.assert_sources(plan)
    collection = ROOT/node['collection']
    raw = read(collection,node['collection_sha256'])
    assert raw['status']=='collected' and raw['resets']==1 and raw['actor_weights_unchanged']
    assert not (raw['reserved_seed_used'] or raw['hardware_io'] or raw['learning_updates'])
    folder,d,env,spec = collection.parent,plan['composition'],plan['environment'],plan['support_effort']
    end,n = d['roll_start_control'],d['worlds']
    assert (raw['controls'],raw['physics_substeps'],raw['transitions'])==(end,5*end,n*end)
    fields = tuple(dict.fromkeys((*PHYSICAL,*STATE_FIELDS,'original_feet','roll_phase')))
    trace = read_trace(folder,raw['chunks'],1,end,fields,sha)
    assert sha(folder/'initial.npz')==raw['initial_sha256'] and sha(folder/'release-poses.npz')==raw['release_sha256']
    with np.load(folder/'initial.npz',allow_pickle=False) as saved:
        initial = dict(saved)
    with np.load(folder/'release-poses.npz',allow_pickle=False) as saved:
        poses = saved['poses']
    joints = trace['qpos'][...,raw['qpos_joint_addresses']]
    limits = np.asarray(env['joint_limits'])
    tol = np.deg2rad(env['applied_joint_limit_tolerance_deg'])
    physical = dict(trace,foot_contact=trace['original_feet'],joint_positions=joints,
                    joint_envelope_ok=((joints>=limits[:,0]-tol)&(joints<=limits[:,1]+tol)).all(axis=-1))
    getup,_,summary = recovery_and_standing(plan,folder,trace,initial,physical)
    handoff = d['getup_to_stand_control']
    timely,terminal = independent_holds({key:val[:handoff] for key,val in physical.items()},env['criteria'])
    np.testing.assert_array_equal(getup, timely&terminal&np.isin(poses,POSES))
    sequence = Sequence(plan['roll_route'])
    for control in range(handoff,end+1):
        frame = {key:trace[key][control-1] for key in STATE_FIELDS}
        if control==end:
            frame['active'] = frame['active'] & (trace['mode'][control-1]==STAND)
        sequence.update(control-handoff,frame)
        np.testing.assert_array_equal(sequence.phase,trace['roll_phase'][control-1])
    entry = sequence.phase==ROLL
    meta, windows = read_windows(plan, raw, folder, trace)
    actual = poses=='supine'
    groups={'roll_entry_missed':np.flatnonzero(actual&~entry),'roll_entry_admitted':np.flatnonzero(actual&entry)}
    a,b = groups.values()
    early = compare_release(windows['release'],trace,(a,b),spec,meta['nominal_weight_n'])
    late = {label:summarize_settled(windows['before-deadline'],ids,spec,meta) for label,ids in groups.items()}
    video = check_video(folder,raw,trace,initial)
    result = dict(status='verified-low-friction-support-effort',seed=env['seed'],foot_sliding_friction=d['foot_friction'],
        protocol_sha256=sha(PLAN), collection_sha256=sha(collection), instrument_summary=meta,
        control_transitions=raw['transitions'], batched_substeps=raw['physics_substeps'],
        measured_world_substeps=n*sum(row['rows'] for row in meta['windows']), original_getup=summary,
        original_timely_hold=int(timely.sum()), original_getup_passed=int(getup.sum()),
        original_getup_miss_true_supine=sorted(np.flatnonzero(actual&~getup).tolist()),
        roll_entry=int(entry.sum()), roll_entry_by_command={key:int(entry[ids].sum()) for key,ids in plan['seven_command']['groups'].items()},
        release_support=early, settled_support=late, world_groups={key:ids.tolist() for key,ids in groups.items()},
        peak_gpu_mib=max(row['used_mib'] for row in raw['gpu_samples']), video=video, learning_updates=0,
        additional_physical_steps_in_review=0, reserved_seed_used=False, hardware_io=False)
    dest=REPORT/'analysis.json'
    if save:
        assert node['status']=='score-pending' and not dest.exists()
        write(dest,result)
        node.update(status='reviewed',result=rel(dest),result_sha256=sha(dest),learning_updates=0)
        info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(dest))
        task.update(info,next_gate='Use the observed load-transfer window and effort margin to freeze the next candidate.')
        project.update(info)
        write(PROJECT,project)
    else:
        assert result==read(dest,node['result_sha256'])
    print('LOW_FRICTION_SUPPORT_EFFORT_SCORED', result['roll_entry'], result['roll_entry_by_command']['turn-left'],
          len(a), len(b), result['measured_world_substeps'],
          {label:{key:details[key] for key in ('head_load_fraction','foot_load_fraction','pwm_saturated_actuator_intervals',
                                              'torque_saturated_actuator_intervals')} for label,details in late.items()},
          sha(dest), flush=True)


if __name__=='__main__':
    import sys

    analyze(save='--check' not in sys.argv[1:])
