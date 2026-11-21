"""Measure the head-to-foot support interval using pinned 4 ms samples."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from roll_metrics import progress


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'configs/experiments/roll_forward_input_integration/head_to_foot_handoff_review.json'
RESULT = ROOT/'records/reports/roll-forward-input-integration/endpoint-progress-training/head-to-foot-handoff.json'


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require_hash(path, expected):
    if sha(path)!=expected:
        raise ValueError(f'Source evidence changed: {path}')
    return path


def load_role(raw, role, manifest, sources, plan, design):
    branch = raw/'clips'/role
    trace = branch/'trace.npz'
    clock = branch/'contact-clock.npz'
    marker = branch/'contact-clock.json'
    meta = read(require_hash(marker,sources[marker.relative_to(ROOT).as_posix()]))
    assert meta['labels']==['ground','head','left_foot','right_foot'] and meta['complete']
    threshold = meta['nominal_weight_n']*design['load_threshold_fraction_of_nominal_weight']
    assert meta['force_threshold_n']==threshold
    expected = sources[clock.relative_to(ROOT).as_posix()]
    assert expected==meta['sha256']
    split = plan['checkpoint_selection']['plant']['shared_snapshot_control']
    start, end = (plan['force_capture'][key] for key in ('first_control','last_control'))
    first_pose = plan['initial_stand_controls']-split
    with np.load(require_hash(trace,manifest['trace_sha256']['trace.npz'])) as trajectory, np.load(require_hash(clock,expected)) as archive:
        quaternions = trajectory['qpos'][...,3:7]
        np.testing.assert_array_equal(quaternions[start-split:end-split],archive['before'][::5])
        np.testing.assert_array_equal(quaternions[start-split+1:end-split+1],archive['after'][4::5])
        np.testing.assert_array_equal(archive['substep'],np.arange(start*5+1,end*5+1))
        prefix = quaternions[first_pose:start-split+1]
        q = np.concatenate((prefix,archive['after']))
        angle = np.rad2deg(progress(dict(quaternion=q,ground_contact=np.ones(q.shape[:2],bool)))[0][len(prefix):])
        force = archive['normal_force_n']
    assert force.shape==angle.shape+(4,) and np.isfinite(force).all() and (force>=0.).all()
    result = []
    low,high = design['observation_phase_degrees']
    duration = design['consecutive_loaded_foot_substeps']
    dt = design['physics_period_ms']
    for angles,loads in zip(angle.T,force.transpose(1,0,2),strict=True):
        onset = np.flatnonzero(angles>=low)
        exit_at = np.flatnonzero(angles>=high)
        last = int(exit_at[0])+1 if len(exit_at) else len(angles)
        result.append({key:handoff(loads,start=int(onset[0]) if len(onset) else last,end=last,threshold=threshold,
                                  duration=duration,dt=dt,bilateral=key=='both-feet')
                       for key in design['foot_support_definitions']})
    return result


def handoff(loads, *, start, end, threshold, duration, dt, bilateral):
    if end-start<duration:
        return dict(status='phase-not-observed')
    feet = loads[start:end,2:] > threshold
    bearing = feet.all(axis=1) if bilateral else feet.any(axis=1)
    first = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(bearing,duration).all(axis=1))
    if not len(first):
        return dict(status='foot-arrival-right-censored')
    onset = start+int(first[0])
    head = np.flatnonzero(loads[:onset+1,1]>threshold)
    if not len(head):
        return dict(status='head-load-missing')
    last_head = int(head[-1])
    gap = loads[last_head+1:onset,0] <= threshold
    return dict(status='measured',foot_arrival_after_180_ms=(onset-start)*dt,
        head_to_foot_gap_ms=len(gap)*dt,unloaded_ground_ms=int(gap.sum())*dt,
        head_loaded_on_foot_arrival=last_head==onset,first_foot_mask=np.flatnonzero(loads[onset,2:]>threshold).tolist())


def paired(candidate, reference, definitions, mask):
    result = {}
    for name in definitions:
        selected = [(a[name],b[name]) for a,b,use in zip(candidate,reference,mask,strict=True) if use]
        pairs = [(a,b) for a,b in selected if a['status']==b['status']=='measured']
        medians = {key:float(np.median([a[key]-b[key] for a,b in pairs])) if pairs else None
                   for key in ('foot_arrival_after_180_ms','head_to_foot_gap_ms','unloaded_ground_ms')}
        result[name] = dict(measured_pairs=len(pairs),selected_worlds=len(selected),candidate_minus_reference=medians,
            candidate_not_measured=int(sum(a['status']!='measured' for a,b in selected)),
            reference_not_measured=int(sum(b['status']!='measured' for a,b in selected)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    project=read(ROOT/'project.json')
    state=project['goal_execution']['roll_head_to_foot_handoff']
    assert sha(PROTOCOL)==state['protocol_sha256']
    design=read(PROTOCOL)
    rows={}
    for name,source in design['sources'].items():
        comparison=read(require_hash(ROOT/source['comparison'],source['comparison_sha256']))
        pinned=read(require_hash(ROOT/source['source_inventory'],source['source_inventory_sha256']))['sources_sha256']
        assert not comparison['reserved_seed_used'] and comparison['learning_updates']==0
        cases={}
        for case,item in comparison['arms'].items():
            raw=ROOT/'records/evaluations'/item['run_id']
            manifest=read(require_hash(raw/'guarded-evaluation.json',item['collection_sha256']))
            plan=read(require_hash(raw/'protocol.json',item['protocol_sha256']))
            assert plan['minimum_roll_completion']==design['original_roll_gate'] and plan['worlds']==128
            a,b,c=item['arms']
            passes=[np.asarray(item['arms'][role]['roll_pass'],bool) for role in (a,b,c)]
            masks=[np.ones(plan['worlds'],bool),passes[0]&passes[1]&~passes[2]]
            observed={role:load_role(raw,role,manifest['branches'][role],pinned,plan,design) for role in (a,b,c)}
            by_cohort={cohort:{ref:paired(observed[c],observed[ref],design['foot_support_definitions'],mask) for ref in (a,b)}
                       for cohort,mask in zip(design['cohorts'],masks,strict=True)}
            cases[case]=dict(seed=plan['seed'],candidate=c,regression_worlds=np.flatnonzero(masks[1]).tolist(),
                             paired=by_cohort,per_world=observed)
        assert [case['seed'] for case in cases.values()]==design['development_seeds']
        rows[name]=cases
    payload=dict(protocol_sha256=sha(PROTOCOL),script_sha256=sha(Path(__file__)),groups=rows,
                 new_simulation_steps=0,new_learning_updates=0,reserved_seed_used=False,hardware_io=False)
    encoded=json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.check:
        assert RESULT.read_text(encoding='utf-8')==encoded
    else:
        with RESULT.open('x',encoding='utf-8',newline='\n') as out:
            out.write(encoded)
    episode_count=sum(len(items) for cases in rows.values() for case in cases.values() for items in case['per_world'].values())
    print('HEAD_TO_FOOT_HANDOFF_VERIFIED',sha(RESULT),sum(len(cases) for cases in rows.values()),episode_count,flush=True)


if __name__=='__main__':
    main()
