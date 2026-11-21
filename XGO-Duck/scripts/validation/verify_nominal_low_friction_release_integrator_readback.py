"""Verify the original experiment with a binary32 solver-clock readback."""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import fcntl

import numpy as np
import torch

import run_nominal_low_friction_release_integrator as run


def reconstruct_clock(origin, start, end, changes):
    # Warp stores time and timestep in binary32. A double precision sum of
    # the nominal time grid accumulates a different rounding error.
    time = np.float32(origin)
    factor = 1
    events = []
    for control in range(start,end):
        if control in changes:
            next_factor = changes[control]
            events.append(dict(control=control,prior_factor=factor,new_factor=next_factor,engine_clock_s=float(time)))
            factor = next_factor
        tick = np.float32(run.original.MOTOR_PERIOD/factor)
        for _ in range(run.MOTOR_TICKS*factor):
            time = np.float32(time+tick)
    return events,float(time)


def clock_readback(folder, plan, role, reached, raw):
    clock = raw['solver_roles'][role]
    start = plan['composition']['release_control']
    worlds = plan['composition']['worlds']
    shared = raw['solver_roles']['shared']
    assert shared['start_time_s']==0
    _,common_time = reconstruct_clock(0,0,start,{})
    origin = clock['start_time_s']
    assert origin==shared['end_time_s']==common_time
    assert clock['motor_ticks']==run.MOTOR_TICKS*(reached-start)
    assert clock['solver_launches']==run.expected_launches(reached,plan,role)
    path = folder/role/'closed-loop-clock.json'
    record = run.read(path,clock['audit_sha256'])
    assert record['role']==role and record['worlds']==worlds and record['motor_delay_ticks']==1
    assert record['host_simulation_step_s']==.004 and record['actor_observation_step_s']==.02
    assert record['checked_every_world_clock'] and record['unchanged_motor_input_between_solver_launches']
    assert record['controls']==clock['controls']
    assert [row['control'] for row in clock['controls']]==list(range(start,start+run.original.AUDIT_CONTROLS))
    changes = run.schedule(plan,role)
    for row in clock['controls']:
        run.original.check_control(row,changes.get(start,1))
    events,final_time = reconstruct_clock(origin,start,reached,changes)
    assert events==clock['transitions'] and final_time==clock['end_time_s']
    window_time = clock['early_window_wall_s']
    assert window_time>0
    return dict(solver_launches=clock['solver_launches'],early_window_wall_s=window_time,
        early_scene_transitions_per_second=60*worlds/window_time,actual_switches=events,
        actual_end_time_s=final_time,float32_accumulation_exact=True)


def archived_fixture(project):
    source = project['goal_execution'][run.window.KEY]
    raw = run.read(run.ROOT/source['collection'],source['collection_sha256'])
    limits = raw['branches']
    _,common_time = reconstruct_clock(0,0,600,{})
    assert common_time==raw['solver_roles']['shared']['end_time_s']
    for name,row in limits.items():
        changes = ({600:4,660:2,1050:1} if name==run.window.ROLES[-1] else {600:2,1050:1})
        events,end = reconstruct_clock(common_time,600,row['controls'],changes)
        actual = raw['solver_roles'][name]
        assert events==actual['transitions'] and end==actual['end_time_s']
    return dict(path=source['collection'],sha256=source['collection_sha256'],exact=True)


def main():
    lock_path = Path.home()/'XGO-Duck/training.lock'
    with ExitStack() as stack:
        lock = stack.enter_context(lock_path.open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contexts = (
            torch.inference_mode(),
            patch.multiple(run.driver,KEY=run.KEY,STAGE=run.STAGE,PLAN=run.PLAN,REPORT=run.REPORT,
                           ROLES=run.ROLES,collect=run.collect),
            patch.multiple(run.parent,config=run.base.configured,PLAN=run.PLAN),
            patch.object(run.comparison.feedback,'components',run.driver.isolated_components),
            patch.object(run,'check_timing',clock_readback),
        )
        for context in contexts:
            stack.enter_context(context)
        project,plan = run.driver.checked('score-pending')
        state = project['goal_execution'][run.KEY]
        raw = run.read(run.ROOT/plan['record']/'collection.json',state['collection_sha256'])
        receipt = run.REPORT/'time-readback-protocol.json'
        assert not receipt.exists()
        fixture = archived_fixture(project)
        run.write(receipt,dict(status='frozen-scoring-readback',protocol_sha256=run.sha(run.PLAN),
            collection_sha256=state['collection_sha256'],scorer_sha256=run.sha(Path(__file__)),archived_fixture=fixture,
            change='Reconstruct every binary32 time increment and match measured switches and final clocks exactly.',
            original_gates_preserved=True,additional_simulation_steps=0,learning_updates=0))
        print('CLOCK_READBACK_PROTOCOL',run.sha(receipt),flush=True)
        run.score()


if __name__=='__main__':
    main()
