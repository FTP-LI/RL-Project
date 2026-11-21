"""Screen one bounded weight-transfer signal on frozen get-up trajectories."""

from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT/'project.json'
HOME = ROOT/'records/reports/nominal-skill-composition'
KEY, STAGE = 'nominal_low_friction_support_credit', 'nominal-low-friction-support-credit'
PREVIOUS_KEY = 'nominal_low_friction_support_effort_limit_readback'
PREVIOUS_STAGE = 'nominal-low-friction-support-effort-limit-readback'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
RESULT = HOME/'low-friction-support-effort'/'verified-entry'/'support-credit-preflight.json'


def digest(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def load(path, expected=None):
    if expected is not None:
        assert digest(path)==expected, path
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, payload):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    tmp.replace(path)


def permitted(project, stage):
    task, goal = project['staged_training_request'], project['goal_execution']
    assert goal['status']=='active' and not goal['pause_requested_by_user'] and task['current_stage']==stage
    assert not any(task[name] for name in ('pause_requested','simulation_running','learning_running','deployment_authorized_now'))


def freeze():
    project = load(PROJECT)
    permitted(project, PREVIOUS_STAGE)
    goal, task = project['goal_execution'], project['staged_training_request']
    prior, nominal = (goal[key] for key in (PREVIOUS_KEY,'nominal_supine_early_load'))
    assert prior['status']=='reviewed' and KEY not in goal and not (PLAN.exists() or RESULT.exists())
    low_ref, nominal_ref = ROOT/prior['result'], ROOT/nominal['analysis']
    low, archived = load(low_ref,prior['result_sha256']), load(nominal_ref,nominal['analysis_sha256'])
    alignment = HOME/'early-getup-load'/'rotation-aligned-to-load.json'
    matching = load(alignment)
    assert matching['source_analysis_sha256']==nominal['analysis_sha256']
    assert low['seed']==1257 and low['foot_sliding_friction']==.8 and not low['reserved_seed_used']
    effort = low['settled_support']['roll_entry_missed']
    assert effort['pwm_saturated_actuator_intervals']==effort['torque_saturated_actuator_intervals']==0
    assert len(effort['records'])==13 and all(row['all_three_loaded_share']==1. for row in effort['records'])
    low_source = (ROOT/prior['collection']).parent
    early, = (row for row in low['instrument_summary']['windows'] if row['window']=='release')
    raw_sources, cohorts = [], []
    for pair_rows, archive in zip(matching['seed_results'],archived['seed_results'],strict=True):
        seed = archive['seed']
        assert seed==pair_rows['seed'] and archive['clock_sha256']==pair_rows['clock_sha256']
        raw = ROOT/archive['output_clock']
        assert digest(raw)==archive['clock_sha256']
        raw_sources.append(raw)
        cohorts.append(dict(seed=seed, foot_sliding_friction=1., nominal_weight_n=archive['nominal_weight_n'],
            path=raw.relative_to(ROOT).as_posix(), sha256=archive['clock_sha256'],
            pairs=[[row['missed_world'],row['admitted_world']] for row in pair_rows['worlds']]))
    raw = low_source/early['file']
    assert digest(raw)==early['sha256']
    cohorts.append(dict(seed=1257,foot_sliding_friction=.8,nominal_weight_n=low['instrument_summary']['nominal_weight_n'],
        path=raw.relative_to(ROOT).as_posix(),sha256=early['sha256'],
        pairs=[[row['missed']['world'],row['admitted']['world']] for row in low['release_support']['matches']]))
    assert [len(cohort['pairs']) for cohort in cohorts]==[6,14,13]
    proof = (Path(__file__).resolve(),low_ref,nominal_ref,alignment,*raw_sources,raw)
    record = dict(stage=STAGE,status='frozen-before-credit-readback',cohorts=cohorts, controls=[124,150],
        control_period_seconds=.02, solver_period_seconds=.004, release_control=100, discount_per_control=.99,
        credit_formula='clip(min(left,right)/(0.02*weight),0,1)*clip((left+right)/(0.5*weight),0,1)*(left+right)/(head+left+right)',
        per_foot_load_fraction=.02, total_foot_load_fraction=.5, minimum_positive_pair_fraction=.75,
        minimum_paired_median_credit_difference=.05, credit_reduce='compute per solve, average five solves per control, then normalized discounted mean',
        method='Full original recovery gate for subsequent evaluation. Contact forces are training supervision only.',
        confirmation_seeds_reserved=[2256,2246], new_physical_steps=0, learning_updates=0, hardware_io=False,
        source_sha256={path.relative_to(ROOT).as_posix():digest(path) for path in proof})
    write(PLAN,record)
    path = PLAN.relative_to(ROOT).as_posix()
    goal[KEY] = dict(status='archive-screen-ready',protocol=path,protocol_sha256=digest(PLAN),hardware_io=False)
    info = dict(current_evidence=path,current_protocol_execution=STAGE+'-frozen')
    task.update(info,current_stage=STAGE,current_protocol=path,current_protocol_frozen=True,
        next_gate='Screen the frozen force transfer credit on three saved paired cohorts before any new learning.')
    project.update(info)
    write(PROJECT,project)
    print('SUPPORT_CREDIT_FROZEN',digest(PLAN),flush=True)


def credit(forces, nominal, plan):
    head,left,right = np.moveaxis(forces,-1,0)
    feet = left+right
    two_feet = np.clip(np.minimum(left,right)/(nominal*plan['per_foot_load_fraction']),0.,1.)
    total = np.clip(feet/(nominal*plan['total_foot_load_fraction']),0.,1.)
    share = np.divide(feet,head+feet,out=np.zeros_like(feet),where=(head+feet)>0.)
    return two_feet*total*share


def score(check):
    project = load(PROJECT)
    permitted(project,STAGE)
    node,task = project['goal_execution'][KEY],project['staged_training_request']
    assert node['status']==('reviewed' if check else 'archive-screen-ready')
    plan = load(PLAN,node['protocol_sha256'])
    for key,value in plan['source_sha256'].items():
        assert digest(ROOT/key)==value,key
    fixture = np.array([[0.,0.,0.],[0.,.25,.25],[.2,.4,.4],[0.,.8,0.],[1.,0.,0.]])
    np.testing.assert_allclose(credit(fixture,1.,plan),[0.,1.,.8,0.,0.],atol=1e-12,rtol=0.)
    ratio = round(plan['control_period_seconds']/plan['solver_period_seconds'])
    start,stop = plan['controls']
    horizon = stop-start
    steps = np.arange(start*ratio+1,stop*ratio+1)
    weights = plan['discount_per_control']**np.arange(horizon)
    results = []
    for cohort in plan['cohorts']:
        pairs = np.asarray(cohort['pairs'])
        assert len(np.unique(pairs))==2*len(pairs) and len(pairs)>0
        with np.load(ROOT/cohort['path'],allow_pickle=False) as saved:
            substeps = saved['substep']
            indices = np.searchsorted(substeps,steps)
            np.testing.assert_array_equal(substeps[indices],steps)
            floors = saved['normal_force_n'][indices][:,pairs,1:].astype(np.float64)
        assert floors.shape==(len(steps),len(pairs),2,3) and np.isfinite(floors).all() and (floors>=0.).all()
        corrected = credit(floors,cohort['nominal_weight_n'],plan)
        control_credit = corrected.reshape(horizon,ratio,len(pairs),2).mean(axis=1)
        discounted = np.average(control_credit,axis=0,weights=weights)
        gains = np.diff(discounted,axis=-1)[:,0]
        lo,mid,hi = np.quantile(gains,[.25,.5,.75])
        positive = int(np.count_nonzero(gains>0.))
        pass_gate = positive/len(pairs)>=plan['minimum_positive_pair_fraction'] and mid>=plan['minimum_paired_median_credit_difference']
        results.append(dict(seed=cohort['seed'],foot_sliding_friction=cohort['foot_sliding_friction'],
            pairs=len(pairs),admitted_higher=positive,paired_difference=[float(lo),float(mid),float(hi)],
            passed=bool(pass_gate),worlds=[dict(missed=int(world[0]),admitted=int(world[1]),
                missed_credit=float(values[0]),admitted_credit=float(values[1])) for world,values in zip(pairs,discounted,strict=True)]))
    ready = all(row['passed'] for row in results)
    evidence = dict(protocol_sha256=digest(PLAN),cohorts=results,gpu_readback_preflight_admitted=ready,
        candidate_training_started=False,physical_steps=0,learning_updates=0,reserved_seeds_used=False,hardware_io=False)
    if check:
        assert load(RESULT,node['result_sha256'])==evidence
    else:
        assert not RESULT.exists()
        write(RESULT,evidence)
        result = RESULT.relative_to(ROOT).as_posix()
        node.update(status='reviewed',result=result,result_sha256=digest(RESULT),admitted=ready)
        info = dict(current_evidence=result,current_protocol_execution=STAGE+'-reviewed')
        task.update(info,next_gate='Validate GPU force readback against the saved 4 ms forces and measure its overhead before freezing an equal-sample training pair.'
                    if ready else 'Review phase alignment and observable support dynamics before selecting another training signal.')
        project.update(info,current_objective='Continue the approved XGO-Duck simulation roadmap. Compare independent GPU force readback for the early foot support signal, then select an equal-sample low-friction candidate only when the measurement and throughput prerequisites hold. Keep the original recovery, roll, seven command and perturbation gates.')
        write(PROJECT,project)
    print('SUPPORT_CREDIT_SCREEN',[(r['foot_sliding_friction'],r['seed'],r['admitted_higher'],r['pairs'],
                                  round(r['paired_difference'][1],6),r['passed']) for r in results],ready,digest(RESULT))


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('freeze','score','check'))
    action = cli.parse_args().action
    if action=='freeze':
        freeze()
    else:
        score(check=action=='check')


if __name__=='__main__':
    main()
