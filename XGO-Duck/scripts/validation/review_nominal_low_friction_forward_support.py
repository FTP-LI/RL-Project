"""Verify the 4ms force, orientation and actuator readback for the early get-up."""

from pathlib import Path

import numpy as np

import run_nominal_low_friction_forward_support as experiment
from review_nominal_low_friction_first_support_effort import tilt
from verify_nominal_skill_composition import check_video, read_trace


ROOT, REPORT, ROLES = experiment.ROOT, experiment.REPORT, experiment.ROLES
read, rel, sha, write = experiment.read, experiment.rel, experiment.sha, experiment.write
START, END = experiment.START, experiment.END
CHANNELS = ['ground','head','left_foot','right_foot']


def first_streak(flags, count, start=0):
    if len(flags)-start < count:
        return None
    ready = np.lib.stride_tricks.sliding_window_view(flags[start:],count).all(axis=-1)
    indices = np.flatnonzero(ready)
    return int(indices[0]+start+count-1) if len(indices) else None


def summary(signals, world, scope, metadata):
    force = signals['normal_force_n'][:,world]
    tau = signals['applied_torque_nm'][:,world]
    duty = signals['voltage_duty'][:,world]
    after = tilt(signals['orientation_after'][:,world])
    before = tilt(signals['orientation_before'][:,world])
    weight = metadata['nominal_weight_n']
    foot_limit = scope['nominal_load_fraction'] * weight
    dwell = scope['minimum_load_solves']
    dt = scope['physics_dt']
    feet = (force[:,2:] > foot_limit).all(axis=1) & (force[:,2:].sum(axis=1) >= scope['both_feet_fraction'] * weight)
    head = force[:,1] > foot_limit
    stable = feet & ~head
    support = first_streak(feet,dwell)
    head_clear = first_streak(stable,dwell)
    head_back = None if head_clear is None else first_streak(head,dwell,head_clear+1)
    upright = first_streak(stable & (after <= scope['diagnostic_upright_threshold_deg']),dwell)
    def timing(event):
        return None if event is None else round((event+1)*dt*1000)
    result = dict(first_bilateral_support_ms=timing(support), first_bilateral_without_head_load_ms=timing(head_clear),
        next_sustained_head_load_ms=timing(head_back), first_upright_bilateral_without_head_load_ms=timing(upright),
        terminal_tilt_deg=float(after[-1]), late_sustained_head_and_feet=bool((feet[-dwell:] & head[-dwell:]).all()))
    if support is None:
        return result
    half = round(scope['comparison_window_ms']/(dt*1000))
    names = metadata['joint_order']
    result['support_sample'] = dict(tilt_deg=float(after[support]), head_force_n=float(force[support,1]),
        left_foot_force_n=float(force[support,2]), right_foot_force_n=float(force[support,3]))
    windows = {}
    for lo,hi in ((-half,0),(0,half),(half,2*half)):
        start,stop = support+1+lo,support+1+hi
        if start < 0 or stop > len(after):
            continue
        seconds = slice(start,stop)
        load = force[seconds]
        windows[f'{round(lo*dt*1000)}:{round(hi*dt*1000)}'] = dict(
            tilt_change_toward_upright_deg=float(before[start]-after[stop-1]),
            mean_head_weight_fraction=float(load[:,1].mean()/weight),
            mean_feet_weight_fraction=float(load[:,2:].sum(axis=1).mean()/weight),
            signed_motor_torque_impulse_nms={name:float(tau[seconds,j].sum(dtype=np.float64)*dt) for j,name in enumerate(names)},
            peak_absolute_duty_fraction={name:float(np.abs(duty[seconds,j]).max()/metadata['pwm_limit']) for j,name in enumerate(names)})
    result['aligned_windows_ms'] = windows
    return result


def verify_role(role, batch, plan, folder, fork, worlds, prefix, initial):
    scope = plan['forward_support']
    path = folder/role
    part = batch['branches'][role]
    assert part['status']=='collected' and part['controls']==END
    assert read(path/'restored-fork.json',part['restore_sha256'])=={k:fork[k] for k in ('physical','controller_sha256')}
    receipt = batch['forward_support']['roles'][role]
    meta = read(path/receipt['path'],receipt['sha256'])
    assert meta['contact_channels']==CHANNELS and set(meta['joint_order'])==set(plan['walking']['joint_names'])
    assert meta['maximum_torque_readback_error_nm'] <= 5e-6
    assert meta['compiled_limits']['effective_range_equals_final_force_range']
    window, = meta['windows']
    file = path/window['file']
    assert sha(file)==window['sha256'] and window['rows']==receipt['sampled_substeps']==(END-START)*5
    with np.load(file,allow_pickle=False) as saved:
        signals = dict(saved)
    np.testing.assert_array_equal(signals['substep'],np.arange(5*START+1,5*END+1))
    assert all(np.isfinite(x).all() for x in signals.values())
    normals, duty, command, applied = (signals[k] for k in ('normal_force_n','voltage_duty','requested_torque_nm','applied_torque_nm'))
    assert normals.shape == ((END-START)*5,len(worlds),4)
    assert duty.shape == command.shape == applied.shape == ((END-START)*5,len(worlds),14)
    assert (normals>=0).all() and (normals[...,0]+5e-5>=normals[...,1:].sum(axis=-1)).all()
    assert float(abs(duty).max())<=meta['pwm_limit']+1e-6 and (signals['supply_v']>0).all()
    limit = np.asarray(meta['solver_torque_limit_nm'],np.float32)
    np.testing.assert_allclose(applied, np.clip(command,-limit,limit),atol=5e-6,rtol=0)
    q = read_trace(path,part['chunks'],START+1,END,('quaternion','qpos'),sha)
    # The substep meter reads the raw solver qpos. The endpoint's world-frame
    # link quaternion is a separately computed field, not a byte-level match.
    boundary = np.concatenate((prefix['qpos'][-1:,:,3:7],q['qpos'][...,3:7]))
    np.testing.assert_array_equal(signals['orientation_before'][::5],boundary[:-1,worlds])
    np.testing.assert_array_equal(signals['orientation_after'][4::5],boundary[1:,worlds])
    np.testing.assert_array_equal(signals['orientation_before'][1:],signals['orientation_after'][:-1])
    link_quaternion = np.concatenate((prefix['quaternion'][-1:],q['quaternion']))
    representation_error = float(abs(boundary[:,worlds]-link_quaternion[:,worlds]).max())
    archive = dict(qpos=np.concatenate((prefix['qpos'],q['qpos'])))
    clip = check_video(path,dict(part,camera_command=batch['camera_command']),archive,initial)
    assert clip['actual_seconds']==4. and clip['frames']==101
    mark, = (row for row in part['boundaries'] if row['control']==START)
    target = path/mark['file']
    assert sha(target)==mark['sha256']
    with np.load(target,allow_pickle=False) as data:
        first = {key:data[key] for key in ('observed','actor_input','mode','raw','actual_target','previous_target','motor')}
    getup = {str(w):summary(signals,i,scope,meta) for i,w in enumerate(worlds)}
    result = dict(source=rel(file),source_sha256=window['sha256'],metadata=rel(path/receipt['path']),
        metadata_sha256=receipt['sha256'],solved_substeps=window['rows'],clip=clip,selected_world_readback=getup,
        raw_solver_endpoint_quaternion_exact=True,link_frame_quaternion_component_max_difference=representation_error)
    return result, first, signals


def analyze(plan, node):
    folder = ROOT/plan['record']
    batch = read(folder/'collection.json',node['collection_sha256'])
    assert batch['status']=='collected' and batch['actor_weights_unchanged'] and batch['resets']==1
    assert batch['learning_updates']==0 and not batch['reserved_seed_used'] and not batch['hardware_io']
    assert (batch['controls'],batch['physics_substeps'],batch['transitions'])==(500,2500,448000)
    assert batch['verified_low_friction_startup_sha256']==sha(ROOT/plan['low_friction_trial']['startup_fields'])
    assert batch['protocol_sha256']==node['protocol_sha256']==sha(folder/'protocol.json')
    scope = plan['forward_support']
    worlds = scope['selected_worlds']
    assert batch['forward_support']['worlds']==worlds and len(worlds)==151
    assert sha(folder/'initial.npz')==batch['initial_sha256']
    assert sha(folder/'release-poses.npz')==batch['release_sha256']
    assert sha(folder/'release-routing.npz')==batch['release_routing_sha256']
    with (np.load(folder/'initial.npz',allow_pickle=False) as data,
          np.load(folder/'release-poses.npz',allow_pickle=False) as actual,
          np.load(folder/'release-routing.npz',allow_pickle=False) as route):
        initial,poses = dict(data),actual['poses']
        swapped = route['latched_supine'] & ~route['right']
    focus = scope['focus_worlds']
    assert all(poses[w]=='supine' and swapped[w] for w in focus)
    prefix = read_trace(folder/'shared',batch['prefix']['chunks'],1,START,('quaternion','qpos'),sha)
    fork = read(folder/'fork.json',batch['fork_sha256'])
    assert fork['physical']['counters']==[START,5*START]
    groups, repeated, first_ref = {},None,None
    for i,role in enumerate(ROLES):
        result, first, signals = verify_role(role,batch,plan,folder,fork,worlds,prefix,initial)
        groups[role] = result
        if i==0:
            first_ref, repeated = first, signals
            continue
        for key in ('observed','actor_input','mode','previous_target'):
            np.testing.assert_array_equal(first[key],first_ref[key],err_msg=f'{role}/{key}')
        if i==1:
            for key in ('raw','actual_target'):
                np.testing.assert_array_equal(first[key],first_ref[key],err_msg=f'{role}/{key}')
            variability = dict(repeated_policy_same_release_action=True,
                first_20ms_torque_max_abs_nm=float(abs(first['motor']-first_ref['motor']).max()),
                first_20ms_orientation_max_abs_quaternion_component=float(abs(
                    signals['orientation_after'][:5]-repeated['orientation_after'][:5]).max()))
        else:
            np.testing.assert_array_equal(first['raw'][~swapped],first_ref['raw'][~swapped])
            assert float(abs(first['raw'][swapped]-first_ref['raw'][swapped]).max())>1e-5
    target_worlds = list(dict.fromkeys((*focus, *scope['matched_archive_worlds'].values(), *scope['repeatability_worlds'],672,863)))
    selected = {str(w):dict(actual_release=str(poses[w]),
                       groups={role:groups[role]['selected_world_readback'][str(w)] for role in ROLES}) for w in target_worlds}
    return dict(status='complete-mechanism-sample',protocol_sha256=node['protocol_sha256'],
        collection_sha256=node['collection_sha256'],seed=1257,foot_sliding_friction=.8,release_control=START,
        stop_control=END,verified_restore_fields=len(fork['fields']),same_policy_numerical_variability=variability,
        selected_worlds=len(worlds),target_readback=selected,roles=groups,transitions=batch['transitions'],
        batched_solver_launches=batch['physics_substeps'],peak_gpu_mib=max(row['used_mib'] for row in batch['gpu_samples']),
        learning_updates=0,heldout_used=False,hardware_io=False)


def main():
    project = read(experiment.PROJECT)
    node = project['goal_execution'][experiment.KEY]
    task = project['staged_training_request']
    assert experiment.parent.allowed(project,experiment.STAGE) and not task['simulation_running']
    assert node['status']=='score-pending'
    plan = read(experiment.PLAN,node['protocol_sha256'])
    experiment.parent.base.shared.assert_sources(plan)
    outcome = analyze(plan,node)
    output = REPORT/'analysis.json'
    assert not output.exists()
    write(output,dict(outcome,analysis_script_sha256=sha(Path(__file__))))
    node.update(status='reviewed',result=rel(output),result_sha256=sha(output))
    experiment.prior.training.earlier.selected(project,experiment.STAGE,experiment.PLAN,experiment.STAGE+'-reviewed',output,
        'Use the common-release measured forces and servo effort to choose a protected, event-aligned recovery intervention.')
    write(experiment.PROJECT,project)
    print('FORWARD_SUPPORT_VERIFIED',sha(output),outcome['transitions'],flush=True)
    for world in plan['forward_support']['focus_worlds']:
        rows = outcome['target_readback'][str(world)]['groups']
        print(world,{role:{k:row[k] for k in ('first_bilateral_support_ms','first_bilateral_without_head_load_ms',
              'next_sustained_head_load_ms','first_upright_bilateral_without_head_load_ms','terminal_tilt_deg')}
              for role,row in rows.items()},flush=True)


if __name__=='__main__':
    main()
