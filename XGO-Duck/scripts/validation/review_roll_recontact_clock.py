"""Read aligned physical substeps to identify when endpoint-excluded roll angles had load."""

import argparse
import json

import numpy as np

from locate_roll_endpoint_support_loss import ROOT, read, sha
from roll_metrics import progress
from roll_stand_walk_sequence import ROLL


def summarize_role(raw, role, plan, expected, sources):
    folder = raw/'clips'/role
    paths = {name: folder/name for name in ('trace.npz','contact-clock.npz','contact-clock.json')}
    for path in paths.values():
        key = path.relative_to(ROOT).as_posix()
        if sha(path) != sources[key]:
            raise ValueError(f'Physical evidence changed for {key}')
    meta = read(paths['contact-clock.json'])
    motion_start = plan['initial_stand_controls']
    shared = plan['checkpoint_selection']['plant']['shared_snapshot_control']
    first,last = (plan['force_capture'][name] for name in ('first_control','last_control'))
    start,stop = first-shared,last-shared+1
    with np.load(paths['trace.npz'],allow_pickle=False) as trace, np.load(paths['contact-clock.npz'],allow_pickle=False) as clock:
        prefix = trace['qpos'][motion_start-shared:start+1,:,3:7]
        physical = np.concatenate((prefix,clock['after']))
        g = np.rad2deg(progress(dict(quaternion=physical,ground_contact=np.ones(physical.shape[:2],bool)))[0])[len(prefix)-1:]
        phase_delta = np.maximum(0.,np.minimum(g[1:],270.)-np.maximum(g[:-1],180.))
        frames = (~trace['ground_contact'][start+1:stop]) & (trace['phase_next'][start:stop-1]==ROLL)
        angle = phase_delta.reshape(last-first,5,plan['worlds'])*frames[:,None,:]
        force = clock['normal_force_n'].reshape(last-first,5,plan['worlds'],4)
        loaded = force[...,0]>meta['force_threshold_n']
        head = force[...,1]>meta['force_threshold_n']
    outcomes = {}
    for name,cohort in expected.items():
        ids = cohort['worlds']
        n = len(ids)
        x,y,h = angle[:,:,ids],loaded[:,:,ids],head[:,:,ids]
        no_credit = x.sum(axis=1)>0.
        head_deg = float((x*h).sum())
        load_deg = float((x*y).sum())
        total = float(x.sum())
        previous = cohort['measurements'][role]
        for value,key in ((x,'endpoint_excluded_angle_deg'),(x*y,'endpoint_excluded_angle_with_load_deg'),
                          (x*h,'endpoint_excluded_angle_with_head_load_deg')):
            if n:
                np.testing.assert_allclose(np.median(value.sum(axis=(0,1))),previous[key]['median'],atol=1e-8,rtol=0)
        outcomes[name] = dict(worlds=n, endpoint_excluded_intervals=int(no_credit.sum()),
            endpoint_excluded_positive_angle_sum_deg=total, head_loaded_angle_sum_deg=head_deg,
            whole_body_loaded_angle_sum_deg=load_deg, no_load_angle_sum_deg=total-load_deg,
            fraction_angle_under_head_load=head_deg/total if total else None,
            fraction_angle_under_any_load=load_deg/total if total else None,
            angle_deg_by_substep=x.sum(axis=(0,2)).tolist(), head_loaded_angle_deg_by_substep=(x*h).sum(axis=(0,2)).tolist(),
            endpoint_excluded_intervals_with_last_solve_loaded=int((no_credit & y[:,4]).sum()),
            endpoint_excluded_intervals_with_observed_next_solve=int(no_credit[:-1].sum()),
            endpoint_excluded_intervals_with_next_solve_loaded=int((no_credit[:-1] & y[1:,0]).sum()))
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    project = read(ROOT/'project.json')
    state = project['goal_execution']['roll_head_recontact_load_recovery']
    source, master = (ROOT/state[key] for key in ('load_result','master'))
    if sha(source)!=state['load_result_sha256'] or sha(master)!=state['master_sha256']:
        raise ValueError('Physical evidence identity differs')
    original, plan = read(source), read(master)
    cases = {}
    for case,result in original['cases'].items():
        path = master.with_name(case+'.json')
        if sha(path)!=plan['protocols'][case]:
            raise ValueError('Case plan differs')
        spec = read(path)
        raw = ROOT/'records/evaluations'/result['run_id']
        arms = {role:summarize_role(raw,role,spec,result['cohorts'],original['sources_sha256'])
                for role in result['original_gate_passes']}
        cases[case] = dict(scores=result['original_gate_passes'],repeated_focus_worlds=result['archived_regressions_repeated'],arms=arms)
    out = source.with_name('physical-contact-clock-audit.json')
    result = dict(source_sha256=sha(source),scope='Actual 4 ms solver load during positive 180 to 270 degree rotation excluded by the original control-endpoint contact score',
        cases=cases,new_physical_steps=0,new_learning_updates=0,reserved_seed_used=False)
    data = (json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    if args.check:
        if out.read_bytes()!=data:
            raise ValueError('Physical clock statistics did not reproduce')
    else:
        with out.open('xb') as stream:
            stream.write(data)
    for name,case in cases.items():
        current = case['arms']['endpoint-reward-candidate']['new_matched_regressions']
        print('ROLL_RECONTACT_CLOCK',name,case['scores'],'worlds',current['worlds'],
              'head_loaded_angle_fraction',current['fraction_angle_under_head_load'],
              'observed_next_solve_loaded',current['endpoint_excluded_intervals_with_next_solve_loaded'],
              'of',current['endpoint_excluded_intervals_with_observed_next_solve'],flush=True)


if __name__ == '__main__':
    main()
