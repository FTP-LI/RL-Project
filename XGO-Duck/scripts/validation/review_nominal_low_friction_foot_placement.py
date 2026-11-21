"""Measure each foot's translation in the frozen initial heading of the archived branches."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from review_nominal_low_friction_right_support_contact_timing import ARMS, PLAN, PROJECT, REPORT, ROOT, SOURCE, read, rel, sha, trace_rows, write


FREEZE = ROOT/'configs/experiments/nominal_low_friction_foot_placement.json'
OUTPUT = REPORT/'foot-placement.json'


def freeze():
    spec = read(PLAN)
    goal = read(PROJECT)['goal_execution']
    timing = goal['nominal_low_friction_right_support_contact_timing']
    assert timing['status']=='reviewed' and goal[SOURCE]['full_route_admitted'] is False
    target = ROOT/timing['result']
    assert sha(target)==timing['result_sha256']
    assert spec['shared_control']==136 and spec['grouping']['gained']==[218]
    assert len(spec['grouping']['persistently_missed'])==8
    assert not FREEZE.exists() and not OUTPUT.exists()
    write(FREEZE,dict(timing_protocol=rel(PLAN),timing_protocol_sha256=sha(PLAN),timing_result=rel(target),
        timing_result_sha256=sha(target),collection=spec['collection'],collection_sha256=sha(ROOT/spec['collection']),
        analysis_script_sha256=sha(Path(__file__)),checkpoints=[136,139,144,150,156],
        coordinate='Foot sites and base translation from the matched G136 state in its horizontal heading.',
        contact='Observed endpoint contact samples in (G136, checkpoint]; 20 ms between reads.',
        independent_seed=2246,additional_physical_steps=0,additional_learning_updates=0))
    print('FOOT_PLACEMENT_FROZEN',sha(FREEZE),flush=True)


def sample(rows, i, k, start, forward):
    base, after = rows[start],rows[k]
    original_xy, actual_xy = (x['foot_positions'][i,:,:2].astype(np.float64) for x in (base,after))
    foot_delta = (actual_xy-original_xy)@forward*1000.
    center_delta = float((after['qpos'][i,:2].astype(np.float64)-base['qpos'][i,:2])@forward*1000.)
    support = np.stack([rows[step]['foot_contact'][i] for step in range(start,k+1)])
    contact_samples = support[1:].sum(axis=0)
    last_air = [int(start+positions[-1]) if len(positions) else None for positions in
        (np.flatnonzero(~support[:,leg]) for leg in (0,1))]
    signed_gap = float((actual_xy[1]-actual_xy[0])@forward*1000.)
    return dict(left_site_delta_mm=float(foot_delta[0]),right_site_delta_mm=float(foot_delta[1]),
        body_delta_mm=center_delta,left_relative_body_delta_mm=float(foot_delta[0]-center_delta),
        right_relative_body_delta_mm=float(foot_delta[1]-center_delta),signed_right_minus_left_mm=signed_gap,
        absolute_gap_mm=abs(signed_gap),left_contact_endpoint_samples=int(contact_samples[0]),
        right_contact_endpoint_samples=int(contact_samples[1]),last_noncontact_control_by_foot=last_air)


def score(check):
    freeze = read(FREEZE)
    assert sha(Path(__file__))==freeze['analysis_script_sha256'] and sha(PLAN)==freeze['timing_protocol_sha256']
    timing = read(ROOT/freeze['timing_result'],freeze['timing_result_sha256'])
    plan = read(PLAN)
    collection = ROOT/freeze['collection']
    raw = read(collection,freeze['collection_sha256'])
    assert timing['worlds']==len(plan['same_state_worlds']) and raw['actor_weights_unchanged']
    assert not any(raw[k] for k in ('hardware_io','learning_updates','reserved_seed_used'))
    start,*later = freeze['checkpoints']
    assert raw['prefix']['controls']==start and tuple(raw['branches'])==ARMS
    ids = np.asarray(plan['same_state_worlds'])
    original = trace_rows(collection.parent/'shared',raw['prefix']['chunks'],ids,start,start)
    rows = {role:original|trace_rows(collection.parent/role,raw['branches'][role]['chunks'],ids,start+1,later[-1])
        for role in ARMS}
    results = {}
    for i, world in enumerate(ids):
        w,x,y,z = original[start]['qpos'][i,3:7].astype(np.float64)
        yaw = np.arctan2(2.*(w*z+x*y),1.-2.*(y*y+z*z))
        forward = np.array([np.cos(yaw),np.sin(yaw)])
        results[str(world)] = {role:{str(k):sample(rows[role],i,k,start,forward) for k in freeze['checkpoints']}
            for role in ARMS}
    fields = ('left_site_delta_mm','right_site_delta_mm','body_delta_mm','absolute_gap_mm')
    aggregate = {group:{role:{str(k):{field:float(np.median([results[str(world)][role][str(k)][field] for world in worlds]))
        for field in fields} for k in later} for role in ARMS} for group,worlds in plan['grouping'].items()}
    payload = dict(status='archived-foot-translation-measurement',protocol_sha256=sha(FREEZE),
        worlds=len(ids),seed=plan['seed'],additional_physical_steps=0,additional_learning_updates=0,
        group_medians=aggregate,worlds_by_control=results)
    if check:
        assert read(OUTPUT)==payload
    else:
        assert not OUTPUT.exists()
        write(OUTPUT,payload)
    print('FOOT_PLACEMENT_VERIFIED',sha(OUTPUT),{role:aggregate['gained'][role]['150'] for role in ARMS},flush=True)


if __name__=='__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','score','check'))
    option = parser.parse_args().action
    freeze() if option=='freeze' else score(option=='check')
