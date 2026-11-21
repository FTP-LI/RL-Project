"""Start the frozen support measurement after its entry-context transcription error."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_nominal_roll_support_transfer as original


ROOT, PROJECT, parent = original.ROOT, original.PROJECT, original.parent
read, sha, rel, write = original.read, original.sha, original.rel, original.write
REPORT, KEY, STAGE = original.REPORT, original.KEY, original.STAGE
PLAN = original.PLAN.with_stem(original.PLAN.stem + '_entry_recovery')
LOG = REPORT.parent / 'nominal-support-transfer-run.log'
ENTRY = Path(__file__).resolve()


def freeze():
    project = read(PROJECT)
    task, state = project['staged_training_request'], project['goal_execution'][KEY]
    assert parent.allowed(project, STAGE) and not task['simulation_running'] and state['status'] == 'zero-ready'
    assert task['current_protocol'] == rel(original.PLAN) and sha(original.PLAN) == state['protocol_sha256']
    assert not PLAN.exists() and not (REPORT / 'zero').exists()
    escaped = LOG.read_text(encoding='utf-8')
    assert "TypeError: bad operand type for unary +: '_patch'" in escaped and 'ZERO_EXIT_CODE=1' in escaped
    plan = read(original.PLAN)
    assert not (ROOT / plan['record']).exists() and not original.torch.cuda.is_initialized()
    parent.base.shared.assert_sources(plan)
    prior = REPORT / 'preflight.json'
    run = 'duck_nominal_roll_support_transfer_recovery_s1257_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plan['environment']['run_id'] = run
    plan['record'] = f'records/evaluations/{run}'
    plan['references'].update({rel(file): sha(file) for file in (prior, original.PLAN, LOG)})
    plan['source_sha256'][rel(ENTRY)] = sha(ENTRY)
    plan['entry_recovery'] = dict(original_plan_sha256=sha(original.PLAN), failed_before_actual_cuda=True,
        code_change='Use explicit context-manager tuples in the entry and collector', physical_substeps=0)
    parent.base.shared.assert_sources(plan)
    write(PLAN, plan)
    proof = REPORT / 'entry-recovery-preflight.json'
    write(proof, dict(protocol_sha256=sha(PLAN), original_protocol=rel(original.PLAN),
        original_protocol_sha256=sha(original.PLAN), original_log=rel(LOG), original_log_sha256=sha(LOG),
        observer=parent.preflight(), physical_substeps=0, hardware_io=False))
    state.update(protocol=rel(PLAN), protocol_sha256=sha(PLAN), run_id=run,
        preflight=rel(proof), original_zero_failed_before_dynamics=True)
    task.update(current_protocol=rel(PLAN), current_protocol_execution='nominal-roll-support-transfer-entry-recovery-frozen',
        current_evidence=rel(proof))
    write(PROJECT, project)
    print('NOMINAL_SUPPORT_TRANSFER_ENTRY_RECOVERY_FROZEN', sha(PLAN), run, flush=True)


def collect(env, plan, actors, folder, observer, watch, ledger):
    clock = original.SupportActuatorClock(env, plan)
    physical = env.sim.step

    def step():
        clock.record(physical)

    try:
        with (
            patch.object(clock.actuator, 'apply_delay', clock.capture_delay),
            patch.object(env.sim, 'step', step),
            patch.object(env, '_reset_idx', side_effect=AssertionError('Unexpected episode reset')),
        ):
            original.COLLECT(env, plan, actors, folder, observer, watch, ledger)
    finally:
        clock.save(folder)
        meta = folder / 'contact-clock.json'
        if meta.exists():
            write(meta, dict(read(meta), actuator_layout=clock.channels,
                motor_sample='desired and firmware fields before the same integration',
                joint_actuator_torque_sample='after the same integration solve'))
            ledger['contact_load'] = dict(meta=rel(meta), meta_sha256=sha(meta),
                substeps=clock.completed, measured_substeps=len(clock.rows))
    assert ledger['physics_substeps'] == clock.completed == env._sim_step_counter
    if ledger['controls'] >= plan['contact_load']['capture_last_control']:
        assert len(clock.rows) == clock.last - clock.first and read(meta)['complete']


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('freeze', 'zero', 'sample', 'score', 'review'))
    action = cli.parse_args().action
    if action == 'freeze':
        return freeze()
    with (
        patch.multiple(parent, PLAN=PLAN, REPORT=REPORT, KEY=KEY, STAGE=STAGE, qualify=original.qualify, collect=collect),
        patch.object(original.feedback, 'components', original.isolated_components),
    ):
        if action == 'score':
            from verify_nominal_skill_composition import score
            return score()
        if action == 'review':
            import review_nominal_skill_composition_contact_load as review
            with patch.object(review, 'KEY', KEY):
                return review.main()
        with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock, original.torch.inference_mode():
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            {'zero': parent.zero, 'sample': parent.sample}[action]()


if __name__ == '__main__':
    main()
