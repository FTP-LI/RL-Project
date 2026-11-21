"""Measure solved normal load through the saved low-friction loss window."""

import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import run_roll_stand_forward_diagnostic as direct
from native_nominal_baseline import verify_sources
from roll_metrics import progress
from roll_stand_walk_sequence import Sequence
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha


ENTRY = Path(__file__).resolve()
STAGE = 'roll-ground-support-window-review'
KEY = 'roll_support_window_probe'
CASE = 's1248-friction-low-low-friction'
PARENT = ROOT/'configs/experiments/roll_forward_input_integration/remaining'/f'{CASE}.json'
REPORT = ROOT/'records/reports/roll-forward-input-integration/support-window-force'
PLANS = ROOT/'configs/experiments/roll_forward_input_integration/support-window-force'
MASTER = PLANS/'master.json'
REVIEW = REPORT.parent/'remaining'/CASE/'roll-prefix-readback.json'


def instrument(factory, base, may_continue):
    import guarded_roll_collector as guarded
    from roll_contact_clock_capture import ContactClock

    # The original single-route collector is captured by the caller and
    # continues to enforce all roll and contact boundary checks.
    original_collector = factory(base, may_continue)

    def collect(env, obs, plan, folder, *args):
        clock = ContactClock(env, plan['force_capture'])
        step = env.sim.step
        def publish(record, state):
            return observer(record, dict(state,purpose='低摩擦滚翻支撑载荷复核'))
        observer = base.live.publish
        try:
            with (patch.object(env.sim,'step',lambda:clock.record(step)),
                  patch.object(guarded,'Sequence',Sequence), patch.object(base.live,'publish',publish)):
                return original_collector(env,obs,plan,folder,*args)
        finally:
            clock.save(folder)

    return collect


def freeze():
    from verify_roll_contact_clock import check_alignment_fixture

    project, review = read(P), read(REVIEW)
    task, goals = project['staged_training_request'],project['goal_execution']
    parent_state = goals['roll_forward_input_remaining']
    require(direct.pair.trial.allowed(project,STAGE) and not task['simulation_running'] and not task['learning_running']
            and parent_state['status']=='closed-at-original-roll-gate' and not PLANS.exists() and not REPORT.exists()
            and review['stopped_roll_success']==112 and not review['reserved_seed_used']
            and all(sha(ROOT/name)==digest for name,digest in review['source_sha256'].items()), 'The roll review or authorization changed')
    parent = read(PARENT)
    origin = read(PARENT.with_name('master.json'))
    require(sha(PARENT)==origin['protocols'][CASE] and sha(PARENT.with_name('master.json'))==parent_state['master_sha256']
            and parent['worlds']==128 and parent['seed']==1248 and parent['roll_controls']==450
            and parent['initial_stand_controls']==50 and parent['minimum_roll_completion']==116,
            'The frozen roll identity or original gate differs')
    windows = review['support_windows'].values()
    starts = [row['primary_gap_start_range'][0] for row in windows]
    ends = [row['primary_gap_end_range'][1] for row in windows]
    first, last = 75,100
    require(first<min(starts) and last>max(ends), 'The selected 1.50–2.00 s interval excludes a known gap')
    check_alignment_fixture()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    horizon = parent['initial_stand_controls']+parent['roll_controls']
    plan = dict(parent,stage=STAGE,run_id=f'duck_roll_support_force_s1248_{stamp}',route_scope='One original-input roll prefix',
        maximum_controls=horizon,maximum_control_transitions=horizon*128,maximum_physics_substeps=horizon*5,
        maximum_sampling_wall_seconds=300, guarded_output=rel(REPORT/CASE),page=rel(REPORT/CASE/'route-view.html'),
        force_capture=dict(capture_seconds=[first*.02,last*.02],first_control=first,last_control=last,
                           maximum_batched_substeps=horizon*5,load_threshold_nominal_weight_fraction=.01))
    sources = [ENTRY,PARENT,REVIEW,*(ENTRY.with_name(name) for name in (
        'roll_contact_clock_capture.py','verify_roll_contact_clock.py','guarded_roll_collector.py','score_roll_friction_route.py',
        'review_roll_forward_input_stop.py'))]
    plan['frozen_sha256'].update({rel(path):sha(path) for path in sources})
    verify_sources(plan)
    PLANS.mkdir()
    (REPORT/CASE).mkdir(parents=True)
    protocol = PLANS/f'{CASE}.json'
    put(protocol,plan)
    put(REPORT/CASE/'windows-entry.json',dict(protocol_sha256=sha(protocol),evidence='native-file-tracker'))
    put(MASTER,dict(stage=STAGE,order=[CASE],protocols={CASE:sha(protocol)},reserved_seed=2246,
                    reserved_seed_used=False,learning_updates=0,hardware_io=False,automatic_retry=False))
    goals[KEY] = dict(status='ready',master=rel(MASTER),master_sha256=sha(MASTER),scored={})
    task.update(current_protocol=rel(protocol),current_protocol_frozen=True,current_protocol_execution='roll-support-load-ready',
                next_gate='Measure the original 4 ms normal forces from 1.50 to 2.00 s with the original roll guard.')
    direct.pair.trial.write(P,project)
    print('ROLL_SUPPORT_FORCE_FROZEN',sha(protocol),horizon,flush=True)


def sample():
    import guarded_roll_collector as guarded
    import native_roll_observation_bounded as observer
    from verify_forward_support_nominal_recovery import RecordedObserver

    original = guarded.make_guarded_collector
    with (patch.object(guarded,'make_guarded_collector',lambda base, guard:instrument(original,base,guard)),
          patch.object(observer,'NativeRollObservationBounded',RecordedObserver)):
        direct.collect(CASE)


def score():
    from score_roll_friction_route import score as score_route
    from verify_continuous_roll_stand_walk import arrays

    project, protocol, plan = direct.checked(CASE,'score-pending')
    folder = ROOT/'records/evaluations'/plan['run_id']
    collected = read(REPORT/CASE/'collection-result.json')
    recorded = folder/'guarded-evaluation.json'
    require(sha(recorded)==collected['evidence_sha256'] and collected['run_id']==plan['run_id'], 'Collection evidence changed')
    route = score_route(plan,folder)
    require(route['controls']==500 and not any(route['safety'].values()) and route['roll_world_pass'] is not None,
            'The original roll prefix or safety evidence is incomplete')
    clock_file = folder/'contact-clock.npz'
    meta, receipt = read(folder/'contact-clock.json'),read(recorded)
    measure, spec = arrays(clock_file),plan['force_capture']
    start,end = spec['first_control'],spec['last_control']
    require(meta['complete'] and sha(clock_file)==meta['sha256'] and meta['labels']==['ground','head','left_foot','right_foot']
            and meta['force_threshold_n']==meta['nominal_weight_n']*spec['load_threshold_nominal_weight_fraction'], 'Force channel metadata differs')
    trace_file = folder/'trace.npz'
    require(sha(trace_file)==receipt['trace_sha256'], 'Route pose evidence changed')
    trace = arrays(trace_file)
    np.testing.assert_array_equal(measure['substep'],np.arange(5*start+1,5*end+1))
    np.testing.assert_array_equal(measure['before'][::5],trace['qpos'][start:end,:,3:7])
    np.testing.assert_array_equal(measure['after'][4::5],trace['qpos'][start+1:end+1,:,3:7])
    np.testing.assert_array_equal(measure['after'][:-1],measure['before'][1:])
    force = measure['normal_force_n']
    require(force.shape==(5*(end-start),128,4) and np.isfinite(force).all() and (force>=0).all(), 'Invalid normal force array')
    force = force.reshape(end-start,5,128,4)
    roll_start = plan['initial_stand_controls']
    active = trace['active'][start+1:end+1]
    roll = trace['phase_next'][start+1:end+1]==1
    angular, _, _ = progress({key:trace[key][roll_start:] for key in ('quaternion','ground_contact')})
    positive = np.diff(angular,axis=0,prepend=0.)[start+1-roll_start:end+1-roll_start]>0
    gap = ~trace['ground_contact'][start+1:end+1] & active & roll & positive
    loaded = (force[...,0]>meta['force_threshold_n']).sum(axis=1)
    groups = {}
    for label, group in (('roll-pass',route['roll_world_pass']),('roll-fail',np.logical_not(route['roll_world_pass']))):
        mask = gap & np.asarray(group)[None,:]
        counts = loaded[mask]
        groups[label] = dict(worlds=int(np.asarray(group).sum()), gap_worlds=int(mask.any(axis=0).sum()),
            endpoint_gap_samples=len(counts), load_count_histogram=np.bincount(counts,minlength=6).tolist(),
            any_loaded_samples=int((counts>0).sum()), fully_unloaded_samples=int((counts==0).sum()),
            by_control=[dict(control=k+start+1,scoring_gaps=int(row.sum()),
                             any_loaded=int((loaded[k,row]>0).sum())) for k,row in enumerate(mask)])
    result = REPORT/'force-readback.json'
    put(result,dict(run_id=plan['run_id'], force_window_seconds=spec['capture_seconds'],threshold_n=meta['force_threshold_n'],
        substep_seconds=.004,score_control_seconds=.02,original_roll_success=route['roll_completion_at_common_prefix'],
        minimum_roll_success=plan['minimum_roll_completion'],physical_transitions=500*128,physical_substeps=500*5,
        sampled_physics_times=len(measure['substep']),groups=groups, video=route['clips'],
        observation=rel(REPORT/CASE/'native-view/native-status.txt'),
        protocol_sha256=sha(protocol),source_sha256={rel(path):sha(path) for path in (clock_file,trace_file,recorded)},
        original_counting_preserved=True,heldout_used=False,learning_updates=0,hardware_io=False))
    state = project['goal_execution'][KEY]
    state.update(status='reviewed',result=rel(result),result_sha256=sha(result),scored={CASE:rel(result)})
    project['staged_training_request'].update(current_evidence=rel(result),current_protocol_execution='roll-support-load-reviewed',
        next_gate='Choose a targeted action change after distinguishing full unloading from the contact-boundary tally.')
    direct.pair.trial.write(P,project)
    print('ROLL_SUPPORT_FORCE_REPLAYED',route['roll_completion_at_common_prefix'],groups['roll-fail']['load_count_histogram'],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','sample','score'))
    args = parser.parse_args()
    if args.action=='freeze':
        freeze()
    else:
        with patch.multiple(direct,STAGE=STAGE,KEY=KEY,PLANS=PLANS,REPORT=REPORT,MASTER=MASTER,ENTRY=ENTRY):
            {'sample':sample,'score':score}[args.action]()
