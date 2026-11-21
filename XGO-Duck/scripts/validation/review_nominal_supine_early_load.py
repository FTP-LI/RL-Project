"""Align early solver load with unmodified roll-entry scores in both seed runs."""

from itertools import product
from pathlib import Path

import numpy as np

from review_nominal_skill_composition_handoff import read
from roll_contact_clock_capture import LABELS
from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence
from routed_getup_sequence import STAND
from run_nominal_supine_early_load import KEY, PROJECT, REPORT, ROOT, paths, rel, sha, write
from verify_nominal_skill_composition import check_video, read_trace


def first_event(mask, steps):
    return np.where(mask.any(axis=0), steps[mask.argmax(axis=0)], -1)


def confirmed(mask, length):
    totals = np.zeros(mask.shape[1], dtype=np.int32)
    reads = np.empty_like(mask)
    for i, tick in enumerate(mask):
        totals = np.where(tick, totals+1, 0)
        reads[i] = totals >= length
    return reads


def roll_admission(plan, trace):
    route = Sequence(plan['roll_route'])
    d = plan['composition']
    start, end = d['getup_to_stand_control'], d['roll_start_control']
    for k in range(start,end+1):
        frame = {name:trace[name][k-1] for name in STATE_FIELDS}
        if k == end:
            frame['active'] = frame['active'] & (trace['mode'][k-1] == STAND)
        route.update(k-start, frame)
        np.testing.assert_array_equal(route.phase, trace['roll_phase'][k-1])
    return route.phase == ROLL


def read_run(project, seed, spec):
    key, _, protocol, _ = paths(seed)
    node = project['goal_execution'][key]
    assert node['status'] == 'score-pending'
    plan = read(protocol, node['protocol_sha256'])
    path = ROOT/node['collection']
    collection = read(path, node['collection_sha256'])
    folder = path.parent
    n, end = spec['worlds_per_seed'], spec['end_control']
    assert collection['status'] == 'collected' and collection['protocol_sha256'] == node['protocol_sha256']
    assert sha(folder/'protocol.json') == node['protocol_sha256']
    assert collection['controls'] == end and collection['physics_substeps'] == end*5
    assert collection['transitions'] == n*end and collection['resets'] == 1 and not collection['reserved_seed_used']
    assert collection['hardware_io'] is False and collection['learning_updates'] == 0
    original = ROOT/plan['contact_load']['initial_cuda_proof']
    old = read(original, next(item['zero_sha256'] for item in spec['sources'] if item['seed'] == seed))
    actual = read(folder/'zero/proof.json',collection['process_local_zero_sha256'])
    verified = read(folder/'zero/load-instrument-zero.json')
    assert actual['composition_boundary'] == old['composition_boundary'] and verified['archived_reset_matches']
    assert actual['physical_substeps'] == verified['physical_substeps'] == 0
    encoded = folder/'release-poses.npz'
    assert sha(encoded) == collection['release_sha256']
    with np.load(encoded, allow_pickle=False) as release:
        pose = release['poses']
    supine = np.flatnonzero(pose == 'supine')
    assert len(supine) == collection['actual_release_counts']['supine']
    source = collection['contact_load']
    assert source['substeps'] == collection['physics_substeps']
    meta = read(ROOT/source['meta'],source['meta_sha256'])
    first,last = spec['capture_controls']
    assert meta['complete'] and meta['recorded_substeps'] == source['measured_substeps'] == 5*(last-first)
    assert tuple(meta['labels']) == LABELS and meta['force_threshold_n'] == verified['threshold_n']
    load_table = folder/'contact-clock.npz'
    assert sha(load_table) == meta['sha256']
    with np.load(load_table,allow_pickle=False) as data:
        clock = dict(data)
    res = np.arange(5*first+1,5*last+1)
    np.testing.assert_array_equal(clock['substep'], res)
    np.testing.assert_array_equal(clock['before'][1:], clock['after'][:-1])
    fields = ('mode', 'roll_phase', 'qpos', *STATE_FIELDS)
    trace = read_trace(folder, collection['chunks'], 1, end, fields, sha)
    np.testing.assert_array_equal(trace['qpos'][first-1:last-1,:,3:7], clock['before'][::5])
    np.testing.assert_array_equal(trace['qpos'][first:last,:,3:7], clock['after'][4::5])
    force = clock['normal_force_n']
    assert force.shape == clock['points'].shape == (len(res), n, len(LABELS))
    assert (clock['points'] >= 0).all() and (force >= 0.).all() and np.isfinite(force).all()
    assert (force[:,:,1:].sum(axis=2) <= force[:,:,0] + 1e-5).all()
    assert sha(folder/'initial.npz') == collection['initial_sha256']
    with np.load(folder/'initial.npz',allow_pickle=False) as initial:
        clip = check_video(folder,collection,trace,dict(initial))
    passed = roll_admission(plan,trace)
    chosen = np.isin(supine, plan['seven_command']['groups']['forward'])
    summary = dict(seed=seed, run_id=collection['run_id'], collection=node['collection'],
        collection_sha256=node['collection_sha256'], protocol_sha256=node['protocol_sha256'],
        output_clock=rel(load_table), clock_sha256=meta['sha256'], clock_meta_sha256=source['meta_sha256'],
        nominal_weight_n=meta['nominal_weight_n'], control_transitions=collection['transitions'],
        batch_solved_substeps=len(res), solved_scene_rows=len(res)*n, video=clip,
        passes_whole_group=int(passed.sum()), total_worlds=n, passes_by_release={name:{
            'admission':int((passed & (pose==name)).sum()), 'population':int((pose==name).sum())}
            for name in collection['actual_release_counts']}, all_forward_admissions=int(passed[plan['seven_command']['groups']['forward']].sum()),
        gpu_memory_peak_mib=max(row['used_mib'] for row in collection['gpu_samples']))
    live = read(folder/'state.json')
    assert live['run_id'] == collection['run_id'] and live['controls'] == end
    previous = 0
    for update in live['updates']:
        control = update['control']
        if first <= previous and control <= last:
            summary['within_load_capture_transitions_per_s'] = dict(first_control=previous,
                last_control=control, transitions_per_second=update['transitions_per_second'])
        previous = control
    return dict(clock=clock, trace=trace, worlds=supine, passed=passed[supine], forward=chosen, summary=summary)


def measure(batch, spec):
    clock, trace, worlds = (batch[name] for name in ('clock','trace','worlds'))
    steps = clock['substep']
    weight = batch['summary']['nominal_weight_n']
    f = clock['normal_force_n'][:,worlds] / weight
    feet = f[:,:,2:].sum(axis=2)
    contact = clock['points'][:,worlds,2:].any(axis=2)
    release = spec['release_control']*5
    after_release = (steps>release)[:,None]
    above = f[:,:,2:] > spec['any_foot_load_fraction']
    any_load = above.any(axis=2) & after_release
    bilateral = above.all(axis=2) & (feet >= spec['bilateral_sum_load_fraction']) & after_release
    touch = first_event(contact & after_release, steps)
    loaded = first_event(any_load,steps)
    lasting = first_event(confirmed(any_load,spec['minimum_consecutive_loaded_solves']),steps)
    sustained = first_event(confirmed(bilateral,spec['minimum_consecutive_bilateral_solves']),steps)
    hit_head = f[:,:,1] > spec['any_foot_load_fraction']
    chosen = np.maximum(np.searchsorted(steps,sustained),0)
    angle = np.rad2deg(np.arccos(np.clip(rotation(clock['after'][:,worlds])[...,2,2],-1.,1.)))
    first_angle = angle[chosen,np.arange(len(worlds))]
    event_head = hit_head[chosen,np.arange(len(worlds))]
    look = spec['recent_load_window_solves']
    features = {}
    for k in spec['decision_controls']:
        tick = k*5
        at, = np.flatnonzero(steps==tick)
        upright = rotation(trace['quaternion'][k-1,worlds])[...,2,:]
        sine = np.linalg.norm(upright[:,:2],axis=1)
        gyro = trace['angular_velocity'][k-1,worlds]
        toward = np.rad2deg((upright[:,0]*gyro[:,1]-upright[:,1]*gyro[:,0]) / np.maximum(sine,.05))
        features[k] = dict(available=trace['active'][k-1,worlds] & (sine>=.05) & ((sustained<0) | (sustained>tick)),
            lead_to_confirmation=np.where(sustained<0,steps[-1]-tick,sustained-tick),
            tilt=angle[at], slow_upright=toward, low_foot_load=feet[at-look+1:at+1].mean(axis=0))
    return dict(touch=touch, load=loaded, sustained_any=lasting, sustained_both=sustained,
                support_tilt=first_angle, head_loaded=event_head, feet_fraction=feet, features=features, angle=angle)


def timing_summary(events, mask, release, period):
    observed = events[mask & (events>=0)]
    ms = (observed-release) * period * 1_000.
    return dict(measured=len(ms), unobserved=int(mask.sum())-len(ms), median_ms=float(np.median(ms)) if len(ms) else None,
                range_ms=[float(ms.min()),float(ms.max())] if len(ms) else None)


def summarize(batch, result, spec):
    release, dt = spec['release_control']*5, spec['physics_period_s']
    passed, worlds, forward = (batch[name] for name in ('passed','worlds','forward'))
    summary = {}
    for label, mask in (('admitted',passed), ('missed',~passed), ('forward_missed',~passed & forward)):
        events = {key:timing_summary(result[key],mask,release,dt) for key in ('touch','load','sustained_any','sustained_both')}
        valid = mask & (result['sustained_both']>=0)
        tilt = result['support_tilt'][valid]
        summary[label] = dict(worlds=int(mask.sum()), timing=events,
            sustained_both_by_previous_control_131=int((valid & (result['sustained_both']<=655)).sum()),
            head_loaded_at_sustained_both=int((valid & result['head_loaded']).sum()),
            tilt_at_sustained_both_median_deg=float(np.median(tilt)) if len(tilt) else None,
            individual_failed_worlds=worlds[mask].tolist() if label!='admitted' else [])
    return summary


OPERATORS = {'tilt':np.greater_equal, 'slow_upright':np.less_equal, 'low_foot_load':np.less_equal}


def evaluate(batch, values, rule, spec):
    passed, worlds = batch['passed'], batch['worlds']
    trigger = values['available'].copy()
    for field, limit in rule.items():
        trigger &= OPERATORS[field](values[field],limit)
    sufficient = values['lead_to_confirmation'] >= 5*spec['minimum_lead_to_bilateral_confirmation_controls']
    covered = trigger & ~passed & sufficient
    exposed = trigger & passed
    misses, admissions = int((~passed).sum()), int(passed.sum())
    return dict(missed=misses, admitted=admissions, early_hits=int(covered.sum()),
                success_exposure=int(exposed.sum()), coverage=float(covered.sum()/misses) if misses else None,
                exposure=float(exposed.sum()/admissions) if admissions else None, early_hit_worlds=worlds[covered].tolist(),
                early_hit_forward_worlds=worlds[covered & batch['forward']].tolist())


def select_rule(batch, result, spec):
    if batch['passed'].all():
        return None
    choices = []
    q = np.asarray(spec['threshold_quantiles'])
    for control, values in result['features'].items():
        successful = batch['passed'] & values['available']
        if not successful.any():
            continue
        grids = [np.unique(np.quantile(values[name][successful],1.-q if name=='tilt' else q))
                 for name in OPERATORS]
        for cutoffs in product(*([None,*grid] for grid in grids)):
            rule = {name:float(limit) for name,limit in zip(OPERATORS,cutoffs,strict=True) if limit is not None}
            if not rule:
                continue
            score = evaluate(batch,values,rule,spec)
            if score['exposure'] is not None and score['exposure'] <= spec['maximum_original_success_exposure']:
                choices.append(dict(control=control,rule=rule,score=score))
    return min(choices, default=None, key=lambda row:(-row['score']['early_hits'],row['score']['success_exposure'],
        len(row['rule']),row['control'],tuple(row['rule'].items())))


def chronology(batch, result, spec):
    release = spec['release_control']*5
    steps = batch['clock']['substep']
    steps = steps[steps>release]
    time = np.concatenate(([0.], (steps-release)*spec['physics_period_s']*1_000.))
    def cumulative(event):
        return np.concatenate(([0.], ((event>=0) & (event<=steps[:,None])).mean(axis=1))).tolist()
    groups = {}
    for name, mask in (('admitted',batch['passed']),('missed',~batch['passed'])):
        if mask.any():
            force = result['feet_fraction'][-len(steps):,mask]
            groups[name] = dict(n=int(mask.sum()),
                sustained_bilateral_fraction=cumulative(result['sustained_both'][mask]),
                foot_load_quartiles=np.quantile(force,(.25,.5,.75),axis=1).tolist())
    return dict(time_ms=time.tolist(), cohort_curves=groups, first_touch_fraction=cumulative(result['touch']))


def main():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    design = ROOT/node['design']
    spec = read(design,node['design_sha256'])
    assert node['status'] == 'score-pending'
    assert not project['staged_training_request']['simulation_running']
    batches = [read_run(project,seed,spec) for seed in node['seeds']]
    measured = [measure(batch,spec) for batch in batches]
    cohorts = [dict(batch['summary'], cohorts=summarize(batch,result,spec), chronology=chronology(batch,result,spec))
               for batch,result in zip(batches,measured,strict=True)]
    choice = select_rule(batches[0],measured[0],spec)
    validation = None if choice is None else evaluate(batches[1],measured[1]['features'][choice['control']],choice['rule'],spec)
    qualification = validation is not None and all(
        row['coverage'] is not None and row['coverage'] >= spec['minimum_miss_coverage']
        and row['exposure'] is not None and row['exposure'] <= spec['maximum_original_success_exposure']
        for row in (choice['score'],validation))
    file = REPORT/'analysis.json'
    assert not file.exists()
    write(file,dict(design_sha256=sha(design),analysis_script_sha256=sha(Path(__file__)),
        seed_results=cohorts, development_rule=choice, unchanged_rule_validation=validation,
        qualifies_observed_load_window=qualification, learning_updates=0, hardware_io=False,
        reserved_seeds_used=False))
    for seed in node['seeds']:
        project['goal_execution'][paths(seed)[0]]['status'] = 'reviewed'
    node.update(status='reviewed', analysis=rel(file), analysis_sha256=sha(file),
                qualifies_observed_load_window=qualification)
    task = project['staged_training_request']
    task.update(current_protocol_execution='nominal-supine-early-load-reviewed', current_evidence=rel(file))
    project.update(current_protocol_execution=task['current_protocol_execution'], current_evidence=task['current_evidence'])
    write(PROJECT,project)
    print('SUPINE_EARLY_LOAD_ANALYSIS',sha(file),choice,validation,qualification,flush=True)
    for seed in cohorts:
        print('SUPINE_EARLY_LOAD_TIMING', seed['seed'], seed['passes_whole_group'], seed['cohorts'], flush=True)


if __name__ == '__main__':
    main()
