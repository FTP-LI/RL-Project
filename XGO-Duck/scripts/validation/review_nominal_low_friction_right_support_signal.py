"""Validate one early sagittal trigger on archived physical and actor inputs."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from review_nominal_low_friction_failure_recurrence import PROJECT, ROOT, read, rel, sha, write
from standup_metrics import projected_gravity


NAME = 'nominal_low_friction_right_support_signal'
PLAN = ROOT/'configs/experiments'/f'{NAME}.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-right-support-signal'
HIP = 'nominal_low_friction_hip_knee_authority'
RELEASES = (
    ('nominal_low_friction_release_integrator',('reference-4ms-first','reference-4ms-repeat','release-2ms-candidate')),
    ('nominal_low_friction_support_ppo_action',('parent-policy',)),
    ('nominal_supine_early_load_1256',(None,)),
)


def freeze():
    project = read(PROJECT)
    goal,task = project['goal_execution'],project['staged_training_request']
    assert goal['status']=='active' and not goal['pause_requested_by_user'] and not task['pause_requested']
    assert not task['deployment_authorized_now'] and not task['learning_running']
    assert not PLAN.exists() and not REPORT.exists()
    info = goal[HIP]
    prior = read(ROOT/info['result'],info['result_sha256'])
    assert info['status']=='reviewed' and not prior['eligible_for_position_action_preflight']
    order = [prior['joints'].index(name) for name in ('right_hip_pitch','right_knee')]
    signs = [np.sign(case['samples'][1]['missed']['proposed_change_nm'])[order].tolist() for case in prior['cases']]
    assert len(signs)==13 and all(row==[-1.,1.] for row in signs)
    sources = []
    for key,branches in RELEASES:
        node = goal[key]
        protocol,collection = (dict(path=node[field],sha256=node[field+'_sha256']) for field in ('protocol','collection'))
        captured = read(ROOT/collection['path'],collection['sha256'])
        spec = read(ROOT/protocol['path'],protocol['sha256'])
        assert captured['status']=='collected' and not any(captured[k] for k in ('learning_updates','hardware_io','reserved_seed_used'))
        assert spec['composition']['release_control']==100 and spec['composition']['roll_start_control']==600
        sources.append(dict(key=key,branches=branches,protocol=protocol,collection=collection))
    spec = dict(status='frozen-fixed-g136-archive-replay',control=136,release_control=100,entry_control=600,
        selection='Apply one rule to the latched supine actor at G136. Actor input 3:6 is the measured projected gravity.',
        earlier_contacts_used_as_inputs=False,tilt_min_deg=20.,tilt_max_deg=60.,stance_basis='body sagittal plane',
        minimum_failed_coverage_fraction=.75,maximum_success_exposure_fraction=.05,
        prior_torque_response=dict(path=info['result'],sha256=info['result_sha256'],right_hip_pitch_sign=-1,right_knee_sign=1),
        sources=sources,script_sha256=sha(Path(__file__)),new_physical_steps=0,learning_updates=0,reserved_seed_used=False)
    REPORT.mkdir()
    write(PLAN,spec)
    print('RIGHT_SUPPORT_SIGNAL_FROZEN',sha(PLAN),flush=True)


def checkpoint(folder,chunks,want,fields):
    chunk, = (item for item in chunks if item['first']<=want<=item['last'])
    path = folder/chunk['file']
    assert sha(path)==chunk['sha256']
    ind = want-chunk['first']
    with np.load(path,allow_pickle=False) as values:
        assert values['control'][ind]==want
        return {key:values[key][ind] for key in fields}


def mask(gravity, eligible, spec):
    assert np.isfinite(gravity).all()
    pitch = np.rad2deg(np.arctan2(gravity[:,0],-gravity[:,2]))
    return eligible&(pitch>=spec['tilt_min_deg'])&(pitch<=spec['tilt_max_deg'])


def summary(tripped,supine,passed,spec):
    fail,success = supine&~passed,supine&passed
    assert fail.any() and success.any()
    f,s = int(fail.sum()),int(success.sum())
    seen_f,seen_s = int((tripped&fail).sum()),int((tripped&success).sum())
    outside = int((tripped&~supine).sum())
    return dict(actual_supine_missed=f,actual_supine_passed=s,missed_triggered=seen_f,passed_triggered=seen_s,
        other_release_triggered=outside,passed=seen_f/f>=spec['minimum_failed_coverage_fraction']
        and seen_s/s<=spec['maximum_success_exposure_fraction'] and outside==0,
        missed_worlds_marked=np.flatnonzero(tripped&fail).tolist(),
        missed_worlds_unmarked=np.flatnonzero(~tripped&fail).tolist())


def score(check=False):
    spec = read(PLAN)
    assert sha(Path(__file__))==spec['script_sha256']
    evidence = spec['prior_torque_response']
    assert sha(ROOT/evidence['path'])==evidence['sha256']
    rows = {}
    for item in spec['sources']:
        p,c = item['protocol'],item['collection']
        plan,raw = (read(ROOT/row['path'],row['sha256']) for row in (p,c))
        folder = (ROOT/c['path']).parent
        k = spec['control']
        seed,friction = plan['environment']['seed'],plan['composition']['foot_friction']
        assert (seed,friction) in ((1257,.8),(1256,1.))
        poses = folder/'release-poses.npz'
        assert sha(poses)==raw['release_sha256']
        with np.load(poses,allow_pickle=False) as values:
            supine = values['poses']=='supine'
        entries = rows[item['key']] = {}
        for branch in item['branches']:
            if branch is None:
                record,storage = raw,folder
            else:
                assert raw['prefix']['controls']==spec['release_control']
                record,storage = raw['branches'][branch],folder/branch
            at = checkpoint(storage,record['chunks'],k,('qpos','active'))
            terminal = checkpoint(storage,record['chunks'],spec['entry_control'],('roll_phase',))
            physical = projected_gravity(at['qpos'][:,3:7])
            # Phase 1 is enter-roll at the original G600 boundary.
            passed = terminal['roll_phase']==1
            proxy = mask(physical,supine&at['active'],spec)
            result = dict(seed=seed,foot_slide=friction,physical_proxy=summary(proxy,supine,passed,spec))
            if branch is not None:
                routing = folder/'release-routing.npz'
                assert sha(routing)==raw['release_routing_sha256']
                with np.load(routing,allow_pickle=False) as routed:
                    latch = routed['latched_supine'] & ~routed['right']
                    first_gravity = routed['gravity']
                actual = storage/'pre-roll-standing.npz'
                assert sha(actual)==record['standing_sha256']
                with np.load(actual,allow_pickle=False) as obs:
                    block = obs['actor_input']
                    np.testing.assert_array_equal(block[0,:,3:6],first_gravity)
                    cursor = k-spec['release_control']
                    gravity = block[cursor,:,3:6]
                    assert (obs['mode'][cursor,latch]==0).all()
                live = mask(gravity,latch&at['active'],spec)
                result.update(actor_observation=summary(live,supine,passed,spec),
                    maximum_actor_to_physical_gravity_abs_difference=float(abs(gravity-physical).max()),
                    latch_matches_actual_supine=int((latch==supine).sum()),actor_observation_sha256=record['standing_sha256'])
            entries[branch or 'nominal-independent-geometry'] = result
    candidates = [row['actor_observation']['passed'] for rolls in rows.values() for row in rolls.values()
                  if 'actor_observation' in row]
    independent = rows[RELEASES[-1][0]]['nominal-independent-geometry']
    out = REPORT/'analysis.json'
    result = dict(status='verified-fixed-time-sagittal-observation',protocol_sha256=sha(PLAN),results=rows,
        all_archived_actor_observations_passed=bool(all(candidates)),independent_nominal_geometry_passed=independent['physical_proxy']['passed'],
        new_physical_steps=0,learning_updates=0,reserved_seed_used=False)
    if check:
        assert read(out)==result
    else:
        assert not out.exists()
        write(out,result)
    print('RIGHT_SUPPORT_SIGNAL_VERIFIED',sha(out),len(candidates),result['all_archived_actor_observations_passed'],
        result['independent_nominal_geometry_passed'],flush=True)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','score','check'))
    args = cli.parse_args()
    freeze() if args.action=='freeze' else score(args.action=='check')
