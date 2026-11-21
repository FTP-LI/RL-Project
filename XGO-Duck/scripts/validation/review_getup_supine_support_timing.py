"""Audit causal trigger confirmation against contact timing on saved supine traces."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import classify_release, projected_gravity


PLAN = ROOT/'configs/experiments/getup_supine_support_timing.json'
OUT = ROOT/'records/reports/getup-supine-support-timing'
FIELDS = ('time', 'quaternion', 'foot_contact', 'native_head_leg_contact', 'height', 'angular_velocity')
SIGNALS = ('rebound', 'foot_loss')


def first_run(mask, width):
    starts = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(mask,width).all(axis=-1))
    return (int(starts[0]),int(starts[0])+width-1) if starts.size else None


def event(pair, controls):
    return dict(onset_control=int(controls[pair[0]]), decision_control=int(controls[pair[1]])) if pair else None


def events(tilt, feet, contact, height, angular, controls, spec, dt):
    start,end = (round(t/dt) for t in spec['decision_window_s'])
    valid = (controls>start)&(controls<=end)
    width = spec['consecutive_samples']
    bilateral = feet.all(axis=-1)
    ready = valid & (tilt<=spec['near_upright_deg']) & bilateral & ~contact
    eligible = np.flatnonzero(ready)
    result = dict(near_upright_clear=None, rebound=None, foot_loss=None, recontact=None,
                  precontact_rebound=None, precontact_foot_loss=None,
                  lead_controls={key:None for key in SIGNALS})
    if not eligible.size:
        return result
    first = int(eligible[0])
    result['near_upright_clear'] = dict(control=int(controls[first]),tilt_deg=float(tilt[first]),
        height_m=float(height[first]),angular_speed_rad_s=float(np.linalg.norm(angular[first])))
    after = np.arange(len(controls))>first
    # The running best only includes samples since first clear bilateral support.
    best = np.minimum.accumulate(np.where(np.arange(len(controls))>=first,tilt,np.inf))
    rebound = (tilt-best)>=spec['rebound_from_running_minimum_deg']
    contact_event = first_run(after&contact,width)
    result['recontact'] = event(contact_event,controls)
    for key,mask in (('rebound',rebound),('foot_loss',~bilateral)):
        candidate = after&valid&mask
        result[key] = event(first_run(candidate,width),controls)
        decision = first_run(candidate&~contact,width)
        result[f'precontact_{key}'] = event(decision,controls)
        if decision and contact_event:
            # Compare the time when the online signal is confirmed with the
            # first sample of the eventual contact run, without backdating it.
            result['lead_controls'][key] = int(controls[contact_event[0]]-controls[decision[1]])
    return result


def summary(rows, spec):
    successes = [row for row in rows if row['passed']]
    misses = [row for row in rows if not row['passed']]
    result = dict(scenarios=len(rows), passes=len(successes), failures=len(misses),
                  clear_bilateral_passes=sum(row['near_upright_clear'] is not None for row in successes),
                  clear_bilateral_failures=sum(row['near_upright_clear'] is not None for row in misses))
    result['failure_order'] = {}
    for key in SIGNALS:
        ordered = dict(signal_before_recontact=0, simultaneous=0, recontact_before_signal=0, no_pair=0)
        for row in misses:
            first, second = row[key],row['recontact']
            if not first or not second:
                ordered['no_pair'] += 1
                continue
            diff = first['onset_control']-second['onset_control']
            ordered['signal_before_recontact' if diff<0 else 'recontact_before_signal' if diff>0 else 'simultaneous'] += 1
        result['failure_order'][key] = ordered
        hits = [row['world'] for row in misses if row[f'precontact_{key}']]
        useful = [row['world'] for row in misses if row['lead_controls'][key] is not None and
                  row['lead_controls'][key]>=spec['minimum_lead_controls_before_recontact']]
        protected = [row['world'] for row in successes if row[f'precontact_{key}']]
        result[key] = dict(failure_activations=hits,useful_failures=useful,success_activations=protected,
            useful_failure_fraction=len(useful)/len(misses) if misses else None,
            success_activation_fraction=len(protected)/len(successes) if successes else None,
            passes_gate=(len(misses)>=spec['minimum_failures_per_seed'] and bool(successes) and
                         len(useful)/len(misses)>=spec['minimum_useful_failure_coverage'] and
                         len(protected)/len(successes)<=spec['maximum_success_activation_rate']))
    return result


def examine(seed, spec):
    source, expected = spec['archives'][str(seed)]
    source = ROOT/source
    assert sha(source)==expected
    proof = read(source)
    assert proof['status']=='passed-independent-verification' and not proof['heldout_used'] and not proof['learning_updates']
    record = ROOT/proof['record']
    protocol = record/'protocol.json'
    assert sha(protocol)==proof['protocol_sha256']
    plan = read(protocol)
    criteria = plan['environment']['criteria']
    assert plan['environment']['seed']==seed and criteria['freeze_seconds']==spec['decision_window_s'][0]
    assert plan['policies'][spec['primary_role']]['model_sha256']==spec['parent_sha256']
    prefix = record/'trace-shared-prefix.npz'
    assert sha(prefix)==proof['prefix_sha256']
    with np.load(prefix) as saved:
        release_quaternion = saved['quaternion'][-1]
        assert float(saved['time'][-1])==criteria['freeze_seconds']
    actual = classify_release(projected_gravity(release_quaternion),criteria['release_alignment_min'])
    dt = criteria['step_dt']
    output = {}
    for role in (spec['primary_role'],spec['secondary_role']):
        arm = proof['arms'][role]
        episodes = arm['timing']['episodes']
        n = len(episodes)
        assert [row['env_index'] for row in episodes]==list(range(n))
        assert actual==[row['release_pose'] for row in episodes]
        passed = np.array([row['hold_start_seconds'] is not None for row in episodes]) & arm['end_hold']['stable_by_environment']
        assert int((passed & (np.array(actual)!='other')).sum())==arm['combined_successes']
        worlds = np.flatnonzero(np.array(actual)=='supine')
        branch = record/f'trace-{role}.npz'
        assert sha(branch)==arm['trace_sha256']
        with np.load(branch) as saved:
            trace = {name:saved[name] if name=='time' else saved[name][:,worlds] for name in FIELDS}
        controls = np.rint(trace['time']/dt).astype(int)
        np.testing.assert_allclose(trace['time'],controls*dt,atol=1e-8,rtol=0)
        np.testing.assert_array_equal(controls,np.arange(round(criteria['freeze_seconds']/dt)+1,
                                                         round(criteria['max_seconds']/dt)+1))
        tilt = np.rad2deg(np.arccos(np.clip(-projected_gravity(trace['quaternion'])[...,2],-1.,1.)))
        rows = []
        for col,world in enumerate(worlds):
            report = events(tilt[:,col],trace['foot_contact'][:,col],trace['native_head_leg_contact'][:,col],
                            trace['height'][:,col],trace['angular_velocity'][:,col],controls,spec,dt)
            rows.append(dict(world=int(world),requested=episodes[world]['requested_pose'],passed=bool(passed[world]),
                             **report))
        assert len(rows)==arm['timing']['actual']['supine']['episodes']
        assert sum(row['requested']=='supine' for row in rows)==32
        output[role] = dict(trace_sha256=arm['trace_sha256'],rows=rows,
            all_actual_supine=summary(rows,spec),
            requested_supine=summary([row for row in rows if row['requested']=='supine'],spec),
            from_side=summary([row for row in rows if row['requested'] in ('left-side','right-side')],spec))
    return dict(seed=seed,record=proof['record'],source_sha256=expected,criteria=criteria,roles=output)


def main():
    assert not OUT.exists()
    spec = read(PLAN)
    assert spec['status']=='frozen-saved-traces-only'
    assert not any(spec[key] for key in ('physical_substeps','learning_updates','heldout_used','hardware_io'))
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage']==spec['stage'] and ROOT/stage['current_protocol']==PLAN
    for path,expected in spec['source_receipts'].items():
        assert sha(ROOT/path)==expected
    seeds = [*spec['reference_seeds'],spec['transfer_seed']]
    with ThreadPoolExecutor(max_workers=len(seeds)) as workers:
        batches = list(workers.map(lambda seed:examine(seed,spec),seeds))
    assert all(batch['criteria']==batches[0]['criteria'] for batch in batches)
    primary = [batch['roles'][spec['primary_role']]['all_actual_supine'] for batch in batches]
    admitted = {signal:all(group[signal]['passes_gate'] for group in primary) for signal in SIGNALS}
    OUT.mkdir()
    write(OUT/'analysis.json',dict(status='passed-saved-timing-review',protocol_sha256=sha(PLAN),
          script_sha256=sha(Path(__file__)),batches=batches,signal_admission=admitted,
          candidate_dynamic_execution_authorized=False,physical_substeps=0,learning_updates=0,heldout_used=False))
    print('SUPINE_SUPPORT_TIMING',[(batch['seed'],group['failures'],
        {signal:[len(group[signal]['useful_failures']),len(group[signal]['success_activations'])]
         for signal in SIGNALS}) for batch,group in zip(batches,primary,strict=True)],admitted,flush=True)


if __name__=='__main__':
    main()
