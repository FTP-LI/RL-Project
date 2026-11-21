"""Recover the fixed-policy load probe after the recorded branch-counter error."""

from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

import run_roll_head_recontact_load_probe as probe
from record_io import write


KEY = 'roll_head_recontact_load_recovery'
STAGE = probe.STAGE+'-recovery'
PLANS = probe.PLANS.with_name('head-recontact-load-recovery')
REPORT = probe.REPORT.with_name('head-recontact-load-recovery')
RECEIPT = REPORT.with_name('head-recontact-load-recovery-admission.json')
ENTRY = Path(__file__).resolve()


def instrument(factory, base, may_continue):
    from roll_contact_clock_capture import ContactClock

    original = factory(base,may_continue)

    def collect(env, observation, plan, folder, *args):
        import mujoco

        clock = ContactClock(env,plan['force_capture'])
        model = env.sim.mj_model
        names = plan['walking']['joint_names']
        probe.require(np.all(model.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT),'Actuator transmission is not a joint motor')
        joint_ids = np.asarray(model.actuator_trnid)[:,0]
        labels = [model.joint(int(joint)).name.rsplit('/',1)[-1] for joint in joint_ids]
        probe.require(len(labels)==len(names) and set(labels)==set(names),'Compiled joint motor order differs')
        motors = [labels.index(name) for name in names]
        motor_index = base.torch.as_tensor(motors,device=env.device)
        dof_index = base.torch.as_tensor(model.jnt_dofadr[joint_ids[motors]].tolist(),device=env.device)
        physics_step = env.sim.step
        shared = plan['checkpoint_selection']['plant']['shared_snapshot_control']*probe.SUBSTEPS
        position = -1

        def flush():
            if not clock.rows:
                return
            branch = folder/'clips'/probe.ROLE_NAMES[position]
            clock.save(branch)
            stamp = branch/'contact-clock.json'
            write(stamp,dict(probe.read(stamp),joint_names=names,
                motor_control_clock='prior to each physics step',joint_actuator_torque_clock='after the same solver step'))

        def step():
            nonlocal position
            substep = env._sim_step_counter
            if substep == shared+1:
                expected = shared if position<0 else clock.maximum
                probe.require(clock.completed==expected and position+1<len(probe.ROLE_NAMES),'Unexpected roll snapshot rewind')
                flush()
                clock.rows.clear()
                position += 1
                clock.completed = shared
            capture = clock.first < substep <= clock.last
            if capture:
                probe.require(position>=0,'Force window overlaps the shared prefix')
                motor = base.cpu(env.sim.data.ctrl[:,motor_index])
            clock.record(physics_step)
            if capture:
                clock.rows[-1].update(motor_control=motor,joint_actuator_torque_nm=base.cpu(env.sim.data.qfrc_actuator[:,dof_index]))

        try:
            with patch.object(env.sim,'step',step):
                result = original(env,observation,plan,folder,*args)
            probe.require(position==len(probe.ROLE_NAMES)-1 and clock.completed==clock.maximum,
                          'A branch or its physical record is missing')
            return result
        finally:
            flush()

    return collect


@contextmanager
def bindings():
    with patch.multiple(probe,KEY=KEY,STAGE=STAGE,PLANS=PLANS,REPORT=REPORT,MASTER=PLANS/'master.json',ENTRY=ENTRY,
                        instrument=instrument):
        yield


def freeze_recovery():
    project = probe.read(probe.P)
    goals, task = project['goal_execution'], project['staged_training_request']
    previous = goals[probe.KEY]
    stopped = probe.REPORT/'seed-1247/collection-result.json'
    outcome = probe.read(stopped)
    raw = probe.ROOT/'records/evaluations'/outcome['run_id']
    runtime = raw/'runtime-contract.json'
    state = probe.read(runtime)
    failed_master = probe.ROOT/previous['master']
    proof = probe.read(failed_master)
    stop_log = stopped.with_name('collect-command.log')
    old_plan = failed_master.with_name('seed-1247.json')
    probe.require(probe.endpoint.previous.direct.pair.trial.allowed(project,probe.STAGE)
        and previous['status']=='review-required' and task['learning_running'] is False and task['simulation_running'] is False
        and not (REPORT.exists() or PLANS.exists() or RECEIPT.exists())
        and outcome['reason']=="UnboundLocalError: cannot access local variable 'position' where it is not associated with a value"
        and outcome['status']=='review-required' and state['status']=='closed'
        and (state['control_steps'],state['physics_substeps'],state['control_transitions'])==(49,245,6272)
        and not list(raw.rglob('contact-clock.npz')) and probe.sha(failed_master)==previous['master_sha256']
        and probe.sha(old_plan)==proof['protocols']['seed-1247'], 'The stopped first-branch record differs')
    repair = next(item for item in instrument.__code__.co_consts if getattr(item,'co_name',None)=='collect')
    callback = next(item for item in repair.co_consts if getattr(item,'co_name',None)=='step')
    probe.require('position' in callback.co_freevars and 'position' not in callback.co_varnames,
                  'Branch callback still has the recorded variable scoping error')
    files = (failed_master,old_plan,stopped,runtime,stop_log,Path(probe.__file__))
    hashes = {probe.rel(path):probe.sha(path) for path in files}
    probe.put(RECEIPT,dict(scope='restore the shared branch index in the callback',preserved_source_sha256=hashes,
         failed_run_id=outcome['run_id'],physical_controls_before_stop=49,physical_substeps_before_stop=245,
         physical_transitions_before_stop=6272,substep_measurements_before_stop=0,learning_updates=0,
         reserved_seed_used=False,hardware_io=False))
    previous.update(status='closed-partial-prefix-branch-counter-error',partial_transitions=6272,
                    recovery_receipt=probe.rel(RECEIPT),recovery_receipt_sha256=probe.sha(RECEIPT))
    task.update(current_stage='roll-endpoint-support-localization',current_protocol_execution='roll-load-recovery-admission')
    write(probe.P,project)
    original_verify = probe.verify_sources

    def verify_sources(plan):
        plan['frozen_sha256'].update(hashes | {probe.rel(RECEIPT):probe.sha(RECEIPT)})
        return original_verify(plan)

    with bindings(), patch.object(probe,'verify_sources',verify_sources):
        probe.freeze()


if __name__ == '__main__':
    if sys.argv[1:]==['freeze']:
        freeze_recovery()
    else:
        with bindings():
            probe.main()
