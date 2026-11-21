"""Compare the stopped rolling prefix against its prior same-seed candidate."""

from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from roll_screen_stop_guard import check_online, check_state
from roll_metrics import progress
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence


ROOT = Path(__file__).resolve().parents[2]
CASE = 's1248-friction-low-low-friction'
REPORT = ROOT/'records/reports/roll-forward-input-integration/remaining'/CASE
OLDER = ROOT/'configs/experiments/roll_friction_paired_development'/f'{CASE}.json'
CURRENT = ROOT/'configs/experiments/roll_forward_input_integration/remaining'/f'{CASE}.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name:data[name] for name in data.files}


def measure(trace, plan):
    sequence = Sequence(plan)
    latch = sequence.roll_latch
    worlds = plan['worlds']
    entered = np.zeros(worlds,dtype=bool)
    held = np.zeros(worlds,dtype=np.int32)
    peak_supported = np.zeros(worlds)
    peak_forward = np.zeros(worlds)
    first_head = np.full(worlds,-1,dtype=np.int32)
    first_supported = first_head.copy()
    first_stop = None
    for step, phase in enumerate(trace['phase_next']):
        state = {key:trace[key][step] for key in STATE_FIELDS}
        assert not check_state(state,step,plan)['stop']
        rolling = sequence.phase==ROLL
        sequence.update(step,state)
        np.testing.assert_array_equal(sequence.phase,phase)
        entered |= sequence.phase==ROLL
        peak_supported[rolling] = np.maximum(peak_supported[rolling],latch.supported[rolling])
        peak_forward[rolling] = np.maximum(peak_forward[rolling],latch.angle[rolling])
        held[rolling] = np.maximum(held[rolling],latch.dwell[rolling])
        first_head[(first_head<0)&latch.seen_head] = step
        first_supported[(first_supported<0)&latch.finished] = step
        outcome = check_online(sequence,state,step,plan)
        if first_stop is None and outcome['stop']:
            first_stop = outcome
    passed = sequence.stand_at>=0
    reason = np.full(worlds,'passed',dtype=object)
    reason[~passed] = 'landing-hold-or-deadline'
    reason[~passed & (first_supported<0)] = 'insufficient-ground-supported-rotation'
    reason[~passed & (first_head<0)] = 'no-qualified-head-contact'
    reason[~passed & ~entered] = 'initial-stand'
    failed = {label:count for label,count in Counter(reason.tolist()).items() if label!='passed'}
    metrics = [dict(world=i, original_roll_pass=bool(passed[i]), first_head_control=int(first_head[i]),
                    first_supported_completion_control=int(first_supported[i]), longest_valid_landing_controls=int(held[i]),
                    max_supported_rotation_deg=float(np.rad2deg(peak_supported[i])),
                    max_forward_rotation_deg=float(np.rad2deg(peak_forward[i])), reason=reason[i]) for i in range(worlds)]
    return dict(passed=passed,stop=first_stop,failure_counts=failed,worlds=metrics)


def differences(before, after, field, shared):
    source, current = before[field], after[field]
    assert source.shape==current.shape
    compare = np.abs(source.astype(np.float64)-current.astype(np.float64))
    world = compare.reshape(compare.shape[:2]+(-1,)).max(axis=-1)
    different = np.flatnonzero((world>0).any(axis=1))
    return dict(first_control_with_any_difference=int(different[0]) if different.size else None,
                differs_at_initial_control=bool((world[0]>0).any()),
                worlds_differing_through_shared_prefix=int((world[:shared+1]>0).any(axis=0).sum()),
                max_abs_through_shared_prefix=float(world[:shared+1].max()),
                max_abs_through_cutoff=float(world.max()))


def support_windows(trace, start, measured, passed):
    sampled = {name:trace[name][start:] for name in ('quaternion','ground_contact')}
    geometric, supported, orientation = progress(sampled)
    positive = np.maximum(np.diff(geometric,axis=0,prepend=0.),0.)
    ground = sampled['ground_contact']
    flat = np.clip((.866-np.abs(orientation[...,2,1]))/(.866-.5),0.,1.)
    no_ground = positive*~ground
    lateral = positive*ground*(1.-flat*flat*(3.-2.*flat))
    np.testing.assert_allclose(np.cumsum(no_ground+lateral,axis=0), geometric-supported, atol=1e-9,rtol=0.)
    lost = []
    for world in np.flatnonzero(~passed):
        steps = np.flatnonzero(geometric[:,world]>=np.deg2rad(330.))
        assert steps.size, 'This diagnostic requires a geometric roll'
        end = int(steps[0])
        head = measured['worlds'][world]['first_head_control']-start
        assert 0<=head<end
        hits = np.flatnonzero((~ground[head+1:end+1,world]) & (positive[head+1:end+1,world]>0))+head+1
        assert hits.size, 'No post-head gap in the scored data'
        intervals = np.split(hits,np.flatnonzero(np.diff(hits)!=1)+1)
        main = max(intervals,key=lambda steps:no_ground[steps,world].sum())
        lost.append(dict(world=int(world), first_head_control=head+start, geometric_threshold_control=end+start,
            scored_ground_gap_deg=float(np.rad2deg(no_ground[:end+1,world].sum())),
            scored_lateral_loss_deg=float(np.rad2deg(lateral[:end+1,world].sum())),
            scored_ground_gap_after_head_deg=float(np.rad2deg(no_ground[head+1:end+1,world].sum())),
            primary_gap_control_start=int(main[0])+start, primary_gap_control_end=int(main[-1])+start,
            primary_gap_deg=float(np.rad2deg(no_ground[main,world].sum()))))
    keys = ('scored_ground_gap_deg','scored_lateral_loss_deg','scored_ground_gap_after_head_deg','primary_gap_deg')
    return dict(worlds=lost, score_contact_period_ms=20,
        median={key:float(np.median([row[key] for row in lost])) for key in keys},
        primary_gap_start_range=[min(row['primary_gap_control_start'] for row in lost),max(row['primary_gap_control_start'] for row in lost)],
        primary_gap_end_range=[min(row['primary_gap_control_end'] for row in lost),max(row['primary_gap_control_end'] for row in lost)])


def main():
    earlier, plan = read(OLDER), read(CURRENT)
    assert earlier['seed']==plan['seed']==1248 and earlier['parameter_factors']==plan['parameter_factors']
    assert all(earlier[skill]['model_sha256']==plan[skill]['model_sha256'] for skill in ('roll','walking'))
    assert earlier['roll']['criteria']==plan['roll']['criteria'] and earlier['walking']['criteria']==plan['walking']['criteria']
    old_folder,new_folder = (ROOT/'records/evaluations'/row['run_id'] for row in (earlier,plan))
    original_trace, original_receipt = old_folder/'trace.npz', old_folder/'guarded-evaluation.json'
    assert sha(original_trace)==read(original_receipt)['trace_sha256']
    stop_path, runtime_path = new_folder/'paired-stop.json',new_folder/'runtime-contract.json'
    stop,runtime = read(stop_path),read(runtime_path)
    cutoff,shared = plan['initial_stand_controls']+plan['roll_controls'],plan['integration']['shared_control_boundary']
    assert read(new_folder/'protocol.json')==plan and sha(new_folder/'protocol.json')==sha(CURRENT)
    assert runtime['control_steps']==stop['physical_controls']==stop['episode_control']==cutoff==500
    assert runtime['control_transitions']==cutoff*plan['worlds'] and runtime['physics_substeps']==cutoff*5
    assert runtime['interface_checks']==1 and not (runtime['capacity_reached'] or runtime['extra_observation_or_render_steps'])
    role = stop['role']
    assert role=='original-forward-input' and set(stop['branches'])=={role} and stop['detail']['reason']=='roll-gate-impossible'
    branch = new_folder/'clips'/role
    common_file,suffix_file = new_folder/'shared/trace.npz',branch/'trace.npz'
    branch_receipt = stop['branches'][role]
    assert sha(suffix_file)==branch_receipt['trace_sha256']['trace.npz'] and branch_receipt['controls']==cutoff-shared
    case_arrays = arrays(common_file),arrays(suffix_file)
    for key, values in case_arrays[0].items():
        np.testing.assert_array_equal(values[-1], case_arrays[1][key][0])
    before = {key:value[:cutoff+1] for key,value in arrays(original_trace).items()}
    after = {key:np.concatenate((value,case_arrays[1][key][1:])) for key,value in case_arrays[0].items()}
    assert len(after['active'])==len(before['active'])==cutoff+1
    receipt = read(branch/'endpoint-contact.json')
    assert not receipt['first_mismatches'] and receipt['zero_matches_standalone']
    contact_file = branch/'endpoint-contact.npz'
    assert sha(contact_file)==receipt['sha256']==branch_receipt['contact_sha256']
    contact = arrays(contact_file)
    observed = np.concatenate((after['ground_contact'][...,None],after['head_contact'][...,None],after['foot_contact']),axis=-1)
    np.testing.assert_array_equal(observed,contact['fresh'])
    np.testing.assert_array_equal(observed,contact['constraint_points']>0)
    np.testing.assert_array_equal(contact['control'],np.arange(cutoff+1))
    np.testing.assert_array_equal(contact['substep'],5*np.arange(cutoff+1))
    archived, stopped = measure(before,earlier),measure(after,plan)
    old_pass,new_pass = archived.pop('passed'),stopped.pop('passed')
    assert archived['stop'] is None and stopped['stop']==stop['detail'] and int(new_pass.sum())==stop['detail']['possible']
    gained,lost = np.flatnonzero(~old_pass&new_pass).tolist(),np.flatnonzero(old_pass&~new_pass).tolist()
    restored = read(branch/'restored-fork.json')
    fork = read(new_folder/'fork.json')
    assert restored=={key:fork[key] for key in ('physical','controller_sha256')}
    other = REPORT.parents[1]/'s1247-friction-low-low-friction'
    other_result = read(other/'trajectory-readback.json')
    other_receipt = ROOT/other_result['source_receipt']
    assert sha(other_receipt)==other_result['source_receipt_sha256']
    proof = read(other_receipt)
    other_folder = ROOT/'records/evaluations'/other_result['run_id']
    other_plan = read(other_folder/'protocol.json')
    other_common, other_suffix = other_folder/'shared/trace.npz',other_folder/'clips'/role/'trace.npz'
    assert sha(other_common)==proof['prefix_sha256']['trace.npz'] and sha(other_suffix)==proof['branches'][role]['trace_sha256']['trace.npz']
    other_prefix, other_tail = arrays(other_common),arrays(other_suffix)
    second_seed = {key:np.concatenate((value,other_tail[key][1:cutoff-shared+1])) for key,value in other_prefix.items()}
    independent = measure(second_seed,other_plan)
    other_passed = independent.pop('passed')
    assert independent['stop'] is None and int(other_passed.sum())==other_result['arms'][role]['roll_completed']
    windows = {key:support_windows(track,plan['initial_stand_controls'],row,passed)
                    for key,track,row,passed in (('1248-earlier',before,archived,old_pass),('1248-stopped',after,stopped,new_pass),
                                                 ('1247-original-input',second_seed,independent,other_passed))}
    report = dict(seed=plan['seed'], friction=plan['parameter_factors']['foot_friction'], original_gate=plan['minimum_roll_completion'],
        earlier_run=earlier['run_id'],stopped_run=plan['run_id'], earlier_roll_success=int(old_pass.sum()), stopped_roll_success=int(new_pass.sum()),
        gained_worlds=gained,lost_worlds=lost,earlier=archived,stopped=stopped,shared_snapshot_control=shared,
        adapted_forward_branch_started=False, restored_snapshot_receipt_matches=True, physical_transitions=runtime['control_transitions'],
        position_differences=differences(before,after,'qpos',shared),velocity_differences=differences(before,after,'qvel',shared),
        support_windows=windows,
        source_sha256={item.relative_to(ROOT).as_posix():sha(item) for item in
                       (OLDER,CURRENT,original_trace,original_receipt,stop_path,runtime_path,common_file,suffix_file,contact_file,
                        other_receipt,other_common,other_suffix,other/'trajectory-readback.json')},
        added_physical_steps=0,learning_updates=0,reserved_seed_used=False)
    (REPORT/'roll-prefix-readback.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('ROLL_PREFIX_REPLAYED',report['earlier_roll_success'],report['stopped_roll_success'],len(gained),len(lost),
          report['position_differences']['first_control_with_any_difference'],stopped['failure_counts'])


if __name__=='__main__':
    main()
