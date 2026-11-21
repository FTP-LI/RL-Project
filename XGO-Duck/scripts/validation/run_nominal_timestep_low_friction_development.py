"""Evaluate the frozen 2 ms route with just the foot sliding coefficients changed."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import recover_nominal_timestep_low_friction_preflight as preflight
from review_nominal_roll_closed_loop_timestep import forward_admission
from review_nominal_skill_composition_handoff import replay


base, repeat = preflight.base, preflight.base.repeat
parent, driver, verifier, comparison = repeat.parent, repeat.driver, repeat.verifier, repeat.comparison
ROOT, PROJECT = repeat.ROOT, repeat.PROJECT
read, rel, sha, write = repeat.read, repeat.rel, repeat.sha, repeat.write
KEY, STAGE = 'nominal_timestep_low_friction_development','nominal-timestep-low-friction-development'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = repeat.REPORT.with_name('closed-loop-timestep-low-friction-development')
ROLE = 'lower-friction'
LABELS = {'shared':'低摩擦跌倒与释放',ROLE:'低摩擦连续路线'}
ENTRY = Path(__file__).resolve()


def freeze():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    assert not PLAN.exists() and not REPORT.exists() and not torch.cuda.is_initialized()
    assert KEY not in goal and parent.allowed(project,preflight.STAGE) and not task['simulation_running']
    source, nominal = (goal[k] for k in (preflight.KEY,repeat.KEY))
    assert source['status']==nominal['status']=='reviewed' and nominal['qualified']
    zero_review = ROOT/source['evidence']
    evidence = read(zero_review,source['evidence_sha256'])
    assert evidence['status']=='verified-single-factor-cuda-zero'
    assert evidence['seed']==1257 and evidence['worlds']==896 and not evidence['reserved_seed_used']
    assert evidence['actual_sliding_coefficients']==base.MODE_VALUES
    arm = evidence['arms'][ROLE]
    init = ROOT/arm['zero']
    fields, reset = (init.with_name(name) for name in ('physical-fields.npz','reset.npz'))
    assert sha(fields)==arm['fields_sha256']
    assert sha(reset)==read(init,arm['zero_sha256'])['reset_sha256']
    prior = ROOT/nominal['protocol']
    plan = read(prior,nominal['protocol_sha256'])
    baseline = ROOT/nominal['result']
    assert read(baseline,nominal['result_sha256'])['advance_to_independent_confirmation']
    d, plant = plan['composition'],plan['environment']
    seed,n = plant['seed'],d['worlds']
    assert (seed,n,d['foot_friction'])==(1257,896,1.0)
    assert d['existing_low_friction_reserved_seed']==2246
    end = d['maximum_controls']
    start, stop = plan['solver_comparison']['fine_solver_interval']
    assert (end,start,stop)==(2125,600,1050)
    run = 'duck-roll-fixed-policy-2ms-friction-low-s1257-' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    d['foot_friction'] = base.MODE_VALUES[ROLE]
    plant['run_id'] = run
    plan.pop('repeat_design')
    plan['standing_pair']['archived_initial'] = rel(init)
    plan['solver_comparison'].update(roles=[ROLE],maximum_solver_launches=5*(end+stop-start),
        solver_seconds_by_role={ROLE:.002})
    plan.update(stage=STAGE,record=f'records/evaluations/{run}',maximum_unique_controls=end,
        maximum_physics_substeps=end*5,maximum_control_transitions=end*n,
        friction_development=dict(zero_verification=rel(zero_review),startup_fields=rel(fields),
            nominal_development_result=rel(baseline),reserved_seed=2246,simulation_only=True))
    plan['references'].update({rel(path):sha(path) for path in (
        ENTRY,prior,baseline,zero_review,init,fields,reset,Path(base.__file__).resolve(),Path(preflight.__file__).resolve(),
        Path(replay.__code__.co_filename).resolve(),Path(forward_admission.__code__.co_filename).resolve())})
    parent.base.shared.assert_sources(plan)
    observer = parent.preflight()
    REPORT.mkdir()
    write(PLAN,plan)
    receipt = REPORT/'preflight.json'
    write(receipt,dict(protocol_sha256=sha(PLAN),zero_evidence_sha256=sha(zero_review),observer=observer,
        friction=d['foot_friction'],voltage=d['voltage_v'],maximum_transitions=end*n,physical_substeps=0))
    goal[KEY] = dict(status='sample-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),preflight=rel(receipt),run_id=run)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(receipt))
    project.update(info)
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
                next_gate='Use scene seed 1257 once at actual slide friction 0.8 and the unchanged action gates.')
    write(PROJECT,project)
    print('NOMINAL_TIMESTEP_LOW_FRICTION_DEVELOPMENT_FROZEN',sha(PLAN),run,flush=True)


def collect(env, plan, actors, folder, observer, watch, ledger):
    fields = ROOT/plan['friction_development']['startup_fields']
    assert env.common_step_counter==env._sim_step_counter==0
    with np.load(fields,allow_pickle=False) as expected:
        for name in sorted(base.RESET_FIELDS):
            actual = parent.cpu(getattr(env.sim.model,name))
            np.testing.assert_array_equal(actual,expected['expanded_'+name],err_msg=name)
    ledger['startup_friction_at_actual_cuda_reset'] = plan['composition']['foot_friction']
    with patch.multiple(repeat,ROLES=(ROLE,),LABELS=LABELS):
        repeat.collect(env,plan,actors,folder,observer,watch,ledger)


def score():
    project, plan = driver.checked('score-pending')
    node = project['goal_execution'][KEY]
    folder = ROOT/plan['record']
    raw = read(folder/'collection.json',node['collection_sha256'])
    assert raw['status']=='collected' and raw['resets']==1 and raw['actor_weights_unchanged']
    assert not (raw['learning_updates'] or raw['hardware_io'] or raw['reserved_seed_used'])
    assert raw['startup_friction_at_actual_cuda_reset']==plan['composition']['foot_friction']==.8
    start = plan['standing_pair']['common_snapshot_control']
    roll, deadline = plan['solver_comparison']['fine_solver_interval']
    branch = raw['branches'][ROLE]
    assert raw['prefix']['controls']==start and raw['controls']==branch['controls']
    assert raw['transitions']==raw['controls']*plan['composition']['worlds']<=plan['maximum_control_transitions']
    assert raw['physics_substeps']==5*raw['controls']
    assert raw['solver_launches']==raw['physics_substeps']+5*(min(deadline,raw['controls'])-roll)
    assert sha(folder/'zero/proof.json')==raw['process_local_zero_sha256']
    clock = raw['solver_roles'][ROLE]
    audit = read(folder/ROLE/'closed-loop-clock.json',clock['audit_sha256'])
    assert audit['role']==ROLE and audit['unchanged_motor_input_between_solver_launches']
    assert [r['control'] for r in audit['controls']]==list(range(roll,roll+repeat.original.AUDIT_CONTROLS))
    for r in audit['controls']:
        repeat.original.check_control(r,2)
    fork = read(folder/'fork.json',raw['fork_sha256'])
    initial_file, pose_file = folder/'initial.npz', folder/'release-poses.npz'
    assert sha(initial_file)==raw['initial_sha256'] and sha(pose_file)==raw['release_sha256']
    with np.load(initial_file,allow_pickle=False) as saved, np.load(pose_file,allow_pickle=False) as released:
        initial, poses = dict(saved), released['poses']
    common = dict(raw['prefix'],qpos_joint_addresses=raw['qpos_joint_addresses'],camera_command=raw['camera_command'])
    prefix = comparison.read_trace(folder/'shared',common['chunks'],1,start,comparison.EVIDENCE_FIELDS,sha)
    with patch.object(verifier,'standing_checks',comparison.branch_first):
        outcome, population, _ = verifier.verify_branch(plan,folder/ROLE,branch,prefix,common,initial,fork)
    first = outcome.pop('standing_policy')
    first['control'] = first.pop('matched_snapshot_control')
    outcome['first_roll_action'] = first
    stitched = [dict(x,file=f'{role}/{x["file"]}') for role,data in (('shared',raw['prefix']),(ROLE,branch))
                for x in data['chunks']]
    _, entered, passed, reason, peak, _ = replay(plan,folder,dict(chunks=stitched))
    np.testing.assert_array_equal(population['roll'],passed)
    forward = forward_admission(plan,folder/ROLE,branch,passed,reason,poses,peak[1])
    status = outcome['passed_development_gate']
    result = REPORT/'analysis.json'
    assert not result.exists()
    write(result,dict(status='verified-lower-friction-development',seed=1257,worlds=int(len(poses)),
        protocol_sha256=sha(PLAN),collection_sha256=node['collection_sha256'],foot_sliding_friction=.8,
        roll_entry=int(entered.sum()),result=outcome,forward_entry_reasons=forward,
        roll_by_actual_release_pose={pose:dict(worlds=int((poses==pose).sum()),
            passed=int(passed[poses==pose].sum())) for pose in parent.POSES},
        transitions=raw['transitions'],solver_launches=raw['solver_launches'],
        peak_gpu_mib=max(x['used_mib'] for x in raw['gpu_samples']),
        qualified_for_independent_confirmation=status,reserved_seed_used=False,learning_updates=0,hardware_io=False))
    node.update(status='reviewed',result=rel(result),result_sha256=sha(result),qualified=status)
    info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(result))
    project.update(info)
    project['staged_training_request'].update(info,
        next_gate='Use reserved seed 2246 for the original full action criteria.' if status else
        'Use the saved low-friction failure stages to select the next development comparison.')
    write(PROJECT,project)
    print('NOMINAL_TIMESTEP_LOW_FRICTION_DEVELOPMENT_SCORED',sha(result),outcome['roll_complete'],
          outcome['command_admission'],outcome['complete'],status,flush=True)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','sample','score'))
    action=cli.parse_args().action
    with (
        (Path.home()/'XGO-Duck/training.lock').open('a') as lock,
        torch.inference_mode(),
        patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=(ROLE,),collect=collect),
        patch.multiple(parent,config=base.configured,PLAN=PLAN),
        patch.object(comparison.feedback,'components',driver.isolated_components),
    ):
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'freeze':freeze,'sample':driver.sample,'score':score}[action]()
