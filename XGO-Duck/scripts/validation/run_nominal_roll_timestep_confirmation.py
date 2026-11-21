"""Run the frozen 2 ms route on the held-out nominal scene bank."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_nominal_roll_timestep_repeat as repeat


comparison, driver, verifier, parent = repeat.comparison, repeat.driver, repeat.verifier, repeat.parent
ROOT, PROJECT = repeat.ROOT, repeat.PROJECT
read, rel, sha, write = repeat.read, repeat.rel, repeat.sha, repeat.write
KEY, STAGE = 'nominal_roll_timestep_confirmation','nominal-roll-timestep-independent-confirmation'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
DESIGN = PLAN.with_name(f'{KEY}_design.json')
REPORT = repeat.REPORT.with_name('closed-loop-timestep-confirmation')
ROLE = 'heldout-nominal'
LABELS = {'shared':'保留种子的跌倒与释放', ROLE:'标称连续路线'}
ENTRY = Path(__file__).resolve()


def freeze():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    spec = read(DESIGN)
    assert not PLAN.exists() and not REPORT.exists() and not torch.cuda.is_initialized()
    assert spec['stage']==STAGE and spec['required_development_stage']==repeat.STAGE
    assert KEY not in goal and parent.allowed(project,repeat.STAGE) and not task['simulation_running']
    calibration = goal[repeat.KEY]
    reference, comparison_proof = (ROOT/calibration[k] for k in ('protocol','result'))
    plan = read(reference,calibration['protocol_sha256'])
    prior = read(comparison_proof,calibration['result_sha256'])
    assert calibration['status']=='reviewed' and calibration['qualified']
    assert prior['advance_to_independent_confirmation'] and not prior['reserved_seed_used']
    assert all(row['passed_development_gate'] for row in prior['results'].values())
    assert prior['protocol_sha256']==sha(reference) and prior['same_policy_weights']
    d, plant, seven = (plan[k] for k in ('composition','environment','seven_command'))
    begin,stop = plan['solver_comparison']['fine_solver_interval']
    seed, n = spec['nominal_heldout_seed'], spec['worlds']
    assert seed==d['reserved_seed']==2256 and plant['seed']!=seed
    assert spec['remaining_low_friction_seed']==d['existing_low_friction_reserved_seed']==2246
    assert plan['solver_comparison']['reserved_seeds']==[seed,spec['remaining_low_friction_seed']]
    for location in ('records/evaluations','records/experiments'):
        assert not any(f'-s{seed}-' in folder.name or f'_s{seed}_' in folder.name for folder in (ROOT/location).iterdir())
    assert n==d['worlds']==plant['num_envs'] and d['foot_friction']==spec['foot_friction']==1.0
    assert spec['shared_release_control']==plan['standing_pair']['common_snapshot_control']==d['release_control']
    assert spec['fine_solver_interval']==[begin,stop]
    assert spec['command_entry_control']==d['command_entry_control']
    assert spec['minimum_roll_completion']==plan['roll_route']['minimum_roll_completion']
    assert spec['minimum_total_commands']==seven['minimum_overall']
    assert spec['minimum_getup_success_fraction_by_actual_pose']==plant['success_rate_min']
    end = spec['maximum_control']
    assert end==d['maximum_controls'] and end*n==spec['maximum_control_transitions']
    motor, solver = 5*end, 5*(end+stop-begin)
    assert motor==spec['maximum_motor_intervals'] and solver==spec['maximum_solver_launches']
    assert not (spec['learning_updates'] or spec['hardware_io'] or spec['automatic_retry'])

    counts = [d['requested_pose_counts'][pose] for pose in parent.POSES]
    codes = np.random.default_rng(seed).permutation(np.repeat(range(4),counts))
    groups = parent.balanced_groups(codes,seven['commands'],seed+70000)
    assert set(groups)==set(seven['groups']) and groups!=seven['groups']
    assert all(len(ids)==128 and seven['minimum_by_command'][name]==spec['minimum_each_command']
               for name,ids in groups.items())
    assert sorted(i for ids in groups.values() for i in ids)==list(range(n))
    seven['groups'] = groups
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-roll-fixed-policy-2ms-heldout-s{seed}-{stamp}'
    d['seed'] = seed
    plant.update(seed=seed,run_id=run,split=dict(pose_codes=codes.tolist(),
        groups={pose:np.flatnonzero(codes==i).tolist() for i,pose in enumerate(parent.POSES)}))
    for model in plan['models'].values():
        model['seed'] = seed
    plan.pop('repeat_design')
    plan['standing_pair']['archived_initial'] = rel(REPORT/'zero/proof.json')
    plan['solver_comparison'].update(roles=[ROLE],maximum_solver_launches=solver,
        solver_seconds_by_role={ROLE:repeat.original.MOTOR_PERIOD/2})
    plan.update(stage=STAGE,record=f'records/evaluations/{run}',maximum_unique_controls=end,
                maximum_physics_substeps=motor,maximum_control_transitions=end*n,independent_confirmation=spec)
    plan['references'].update({rel(path):sha(path) for path in (ENTRY,DESIGN,reference,comparison_proof)})
    parent.base.shared.assert_sources(plan)
    observer_preflight = parent.preflight()
    REPORT.mkdir()
    write(PLAN,plan)
    proof = REPORT/'preflight.json'
    write(proof,dict(status='frozen-seed-and-sources',seed=seed,protocol_sha256=sha(PLAN),
        development_repeat_sha256=calibration['result_sha256'],observer=observer_preflight,
        assigned_per_command={key:len(ids) for key,ids in groups.items()},physical_substeps=0,
        learning_updates=0,hardware_io=False))
    goal[KEY] = dict(status='zero-ready',run_id=run,protocol=rel(PLAN),protocol_sha256=sha(PLAN),
                     preflight=rel(proof),seed=seed,hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(proof))
    project.update(info)
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
                next_gate='Run actual CUDA reset and one independent nominal rollout at the original gates.')
    write(PROJECT,project)
    print('ROLL_TIMESTEP_CONFIRMATION_FROZEN',sha(PLAN),run,flush=True)


def collect(env, plan, actors, folder, observer, watch, ledger):
    ledger['reserved_seed_used'] = True
    with patch.multiple(repeat,ROLES=(ROLE,),LABELS=LABELS):
        repeat.collect(env,plan,actors,folder,observer,watch,ledger)


def sample():
    project, plan = driver.checked('sample-ready')
    stage = project['goal_execution'][KEY]
    zero = read(ROOT/stage['zero'],stage['zero_sha256'])
    assert zero['protocol_sha256']==sha(PLAN) and zero['physical_substeps']==0
    assert plan['environment']['seed']==stage['seed']
    driver.sample()


def score():
    project, plan = driver.checked('score-pending')
    goal, task = project['goal_execution'], project['staged_training_request']
    node, folder = goal[KEY], ROOT/plan['record']
    collection = folder/'collection.json'
    raw = read(collection,node['collection_sha256'])
    assert raw['status']=='collected' and raw['resets']==1 and raw['actor_weights_unchanged']
    assert not (raw['learning_updates'] or raw['hardware_io']) and raw['reserved_seed_used']
    assert sha(folder/'protocol.json')==sha(PLAN)
    assert sha(folder/'zero/proof.json')==raw['process_local_zero_sha256']
    start = plan['standing_pair']['common_snapshot_control']
    begin,stop = plan['solver_comparison']['fine_solver_interval']
    assert raw['prefix']['controls']==start and tuple(raw['branches'])==(ROLE,)
    branch = raw['branches'][ROLE]
    assert raw['controls']==branch['controls'] and raw['controls']<=plan['maximum_unique_controls']
    assert raw['physics_substeps']==5*raw['controls']
    assert raw['transitions']==raw['controls']*plan['environment']['num_envs']
    assert raw['solver_launches']==raw['physics_substeps']+5*(min(stop,raw['controls'])-begin)
    shared, clock = (raw['solver_roles'][key] for key in ('shared',ROLE))
    assert clock['start_time_s']==shared['end_time_s'] and shared['motor_ticks']==shared['solver_launches']==5*start
    assert clock['motor_ticks']==raw['physics_substeps']-5*start
    assert clock['solver_launches']==raw['solver_launches']-5*start
    marker = read(folder/ROLE/'closed-loop-clock.json',clock['audit_sha256'])
    assert marker['role']==ROLE and marker['worlds']==plan['environment']['num_envs'] and marker['motor_delay_ticks']==1
    assert marker['checked_every_world_clock'] and marker['unchanged_motor_input_between_solver_launches']
    assert marker['controls']==clock['controls']
    assert [row['control'] for row in marker['controls']]==list(range(begin,begin+repeat.original.AUDIT_CONTROLS))
    for row in marker['controls']:
        repeat.original.check_control(row,2)
    fork = read(folder/'fork.json',raw['fork_sha256'])
    assert fork['physical']['counters']==[start,5*start] and len(fork['fields'])==fork['physical']['fields']
    initial_file, poses_file = folder/'initial.npz',folder/'release-poses.npz'
    assert sha(initial_file)==raw['initial_sha256'] and sha(poses_file)==raw['release_sha256']
    with np.load(initial_file,allow_pickle=False) as saved, np.load(poses_file,allow_pickle=False) as release:
        initial, poses = dict(saved), release['poses']
    common = dict(raw['prefix'],qpos_joint_addresses=raw['qpos_joint_addresses'],camera_command=raw['camera_command'])
    prefix = comparison.read_trace(folder/'shared',common['chunks'],1,start,comparison.EVIDENCE_FIELDS,sha)
    with patch.object(verifier,'standing_checks',comparison.branch_first):
        action, population, _ = verifier.verify_branch(plan,folder/ROLE,branch,prefix,common,initial,fork)
    first_roll = action.pop('standing_policy')
    first_roll['control'] = first_roll.pop('matched_snapshot_control')
    action['first_roll_action'] = first_roll
    passed = action['passed_development_gate']
    view = read(folder/'state.json')
    assert view['run_id']==raw['run_id'] and view['controls']==raw['controls']
    result = REPORT/'analysis.json'
    assert not result.exists()
    write(result,dict(status='verified-independent-nominal-confirmation',protocol_sha256=sha(PLAN),
        collection_sha256=sha(collection),seed=plan['environment']['seed'],worlds=plan['environment']['num_envs'],
        controls=raw['controls'],transitions=raw['transitions'],solver_launches=raw['solver_launches'],
        gpu_peak_mib=max(x['used_mib'] for x in raw['gpu_samples']),
        rolling_throughput=repeat.throughput(view['updates'],plan['environment']['num_envs'],begin,stop),
        verified_release_snapshot=fork['physical'],result=action,nominal_confirmation_passed=passed,
        roll_by_actual_release_pose={pose:dict(worlds=int((poses==pose).sum()),
            rolled=int(population['roll'][poses==pose].sum())) for pose in parent.POSES},
        nominal_reserved_seed_used=True,low_friction_reserved_seed_used=False,learning_updates=0,hardware_io=False))
    node.update(status='reviewed',result=rel(result),result_sha256=sha(result),
                passed=passed,nominal_heldout_consumed=True)
    info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(result))
    project.update(info)
    task.update(info,next_gate='Prepare the frozen 2 ms route for independent disturbance and lower friction evaluation.'
                if passed else 'Diagnose the specific failed entry or action gate. Preserve this held-out run and select changes on development seeds.')
    write(PROJECT,project)
    print('ROLL_TIMESTEP_CONFIRMATION_SCORED',sha(result),action['roll_complete'],
          action['command_admission'],action['complete'],passed,flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','zero','sample','score'))
    action = parser.parse_args().action
    with (
        (Path.home()/'XGO-Duck/training.lock').open('a') as lock,
        torch.inference_mode(),
        patch.multiple(parent,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT),
        patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=(ROLE,),collect=collect),
        patch.object(comparison.feedback,'components',driver.isolated_components),
    ):
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'freeze':freeze,'zero':parent.zero,'sample':sample,'score':score}[action]()


if __name__=='__main__':
    main()
