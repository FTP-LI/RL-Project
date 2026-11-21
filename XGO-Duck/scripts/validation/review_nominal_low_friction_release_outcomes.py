"""Read back stage-specific command losses and the common release routing."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, read, rel, sha, write


KEY = 'nominal_low_friction_release_integrator'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-release-integrator'


def roll_entry(folder, branch, end):
    chunk, = (row for row in branch['chunks'] if row['first']<=end<=row['last'])
    path = folder/chunk['file']
    assert sha(path)==chunk['sha256']
    with np.load(path,allow_pickle=False) as data:
        phase = data['roll_phase'][end-chunk['first']]
    # The frozen roll phase enum records phase 1 at G600 for each admitted world.
    return phase==1,dict(path=rel(path),sha256=chunk['sha256'])


def analyze():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    assert node['status']=='reviewed'
    plan,outcome,raw = (read(ROOT/node[key],node[key+'_sha256']) for key in ('protocol','result','collection'))
    assert (plan['environment']['seed'],plan['composition']['worlds'],plan['composition']['foot_friction'])==(1257,896,.8)
    assert raw['actor_weights_unchanged'] and not any(raw[k] for k in ('learning_updates','reserved_seed_used','hardware_io'))
    run = ROOT/plan['record']
    release,routing = run/'release-poses.npz',run/'release-routing.npz'
    assert sha(release)==raw['release_sha256'] and sha(routing)==raw['release_routing_sha256']
    with np.load(release,allow_pickle=False) as p, np.load(routing,allow_pickle=False) as route:
        pose = p['poses']
        right, supine = route['right'],route['latched_supine']&~route['right']
    latched = {}
    for name,total in raw['actual_release_counts'].items():
        mask = pose==name
        assert int(mask.sum())==total
        latched[name] = dict(scenes=total,supine_branch=int(supine[mask].sum()),right_branch=int(right[mask].sum()))
    assert np.array_equal(supine,pose=='supine') and np.array_equal(right,pose=='right-side')
    stage = plan['composition']['roll_start_control']
    prior_recurring = plan['release_integrator']['archived_recurring_worlds']
    repeated = set(map(int,prior_recurring))
    roles = {}
    for name,branch in raw['branches'].items():
        observed = outcome['results'][name]
        entered,trace = roll_entry(run/name,branch,stage)
        assert int(entered.sum())==observed['roll_entry']
        admitted = np.isin(np.arange(len(pose)),branch['command_entry_worlds'])
        assert not (admitted&~entered).any() and int(admitted.sum())==observed['roll_complete']
        lost = {}
        for command,group in plan['seven_command']['groups'].items():
            ids = np.asarray(group)
            assert admitted[ids].sum()==observed['command_admission'][command]
            lost[command] = dict(assigned=len(ids),before_roll=ids[~entered[ids]].tolist(),
                after_roll_entry_before_command=ids[entered[ids]&~admitted[ids]].tolist())
        repeated.intersection_update(np.flatnonzero(supine&~entered).tolist())
        roles[name] = dict(trace=trace,requested_supine_getup=observed['getup']['requested']['supine'],
            by_command=lost,command_entered=int(admitted.sum()))
    return dict(status='verified-release-routing-and-command-stage-losses',source_sha256={
        node[k]:node[k+'_sha256'] for k in ('protocol','result','collection')},
        source_script_sha256=sha(Path(__file__)),release_sha256=raw['release_sha256'],
        routing_sha256=raw['release_routing_sha256'],release_routes=latched,roles=roles,
        prior_recurring_supine_all_three_still_no_roll=sorted(repeated),
        new_physical_steps=0,learning_updates=0,hardware_io=False,reserved_seed_used=False)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('--check',action='store_true')
    args = cli.parse_args()
    target = REPORT/'paired-failure-readback.json'
    data = analyze()
    if args.check:
        assert read(target)==data
    else:
        assert not target.exists()
        write(target,data)
    print('RELEASE_ROUTE_AND_FAILURES_VERIFIED',sha(target),data['prior_recurring_supine_all_three_still_no_roll'],flush=True)
