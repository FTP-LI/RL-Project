"""Capture unchanged seed-1257 support and actuator state every four milliseconds."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch

from recover_nominal_skill_composition_zero import isolated_components
from review_nominal_roll_support_transfer import KEY, REPORT
from roll_contact_clock_capture import ContactClock
import routed_getup_sequence as feedback
import run_nominal_skill_composition as parent
import run_nominal_skill_composition_contact_load as load_probe


ROOT, PROJECT = parent.ROOT, parent.PROJECT
read, sha, rel, write = parent.read, parent.sha, parent.rel, parent.write
PLAN = parent.PLAN.with_name('nominal_roll_support_transfer.json')
STAGE = 'nominal-skill-composition-support-transfer'
ENTRY = Path(__file__).resolve()
COLLECT = parent.collect


def freeze():
    project = read(PROJECT)
    task, goal = project['staged_training_request'], project['goal_execution']
    assert parent.allowed(project, STAGE + '-preparation') and not task['simulation_running']
    assert KEY not in goal and not PLAN.exists() and not (REPORT / 'preflight.json').exists()
    previous, load = (goal[name] for name in ('nominal_skill_composition_independent', 'nominal_skill_composition_contact_load'))
    assert previous['status'] == load['status'] == 'reviewed'
    evidence = {}
    for row, key in ((previous, 'protocol'), (previous, 'result'), (previous, 'zero'),
                     (previous, 'support_window_review'), (load, 'contact_load_review')):
        path = ROOT / row[key]
        assert sha(path) == row[key + '_sha256']
        evidence[key] = path
    archived = REPORT / 'archived-support-transfer.json'
    replayed = read(archived)
    assert replayed['groups']['1256']['contact_load_review_sha256'] == sha(evidence['contact_load_review'])
    assert replayed['additional_physical_steps'] == 0
    assert read(evidence['result'])['roll_complete'] == 848
    zero = read(evidence['zero'])
    assert zero['protocol_sha256'] == sha(evidence['protocol'])
    reset = evidence['zero'].with_name('reset.npz')
    assert sha(reset) == zero['reset_sha256']
    plan = read(evidence['protocol'])
    d, plant = plan['composition'], plan['environment']
    assert plant['seed'] == 1257 and plant['num_envs'] == d['worlds'] == 896
    assert d['maximum_controls'] == 2125 and d['control_period_s'] == .02 and d['roll_start_control'] == 600
    assert not torch.cuda.is_initialized()
    origin = d['roll_start_control']
    first, last = origin + 20, origin + 70
    run = 'duck_nominal_roll_support_transfer_s1257_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plant['run_id'] = run
    plan.update(stage=STAGE, record=f'records/evaluations/{run}', contact_load=dict(
        capture_first_control=first, capture_last_control=last,
        capture_seconds=[round(i * .02, 2) for i in (first, last)],
        score_gap_first_endpoint_control=origin + 27, score_gap_last_endpoint_control=origin + 46,
        physics_period_s=.004, load_threshold_nominal_weight_fraction=.02,
        maximum_batched_substeps=5 * d['maximum_controls'], initial_cuda_proof=rel(evidence['zero']),
        original_action_gates_unchanged=True, contact_scope='Normal load and original 14-firmware channels',
        reserved_confirmation_seeds=plan['independent_development']['reserved_confirmation_seeds']))
    plan['references'].update({rel(path): sha(path) for path in (*evidence.values(), archived, reset)})
    files = (ENTRY, Path(load_probe.__file__).resolve(), ENTRY.with_name('roll_contact_clock_capture.py'),
             ENTRY.with_name('review_nominal_roll_support_transfer.py'),
             ENTRY.with_name('review_nominal_skill_composition_contact_load.py'))
    plan['source_sha256'].update({rel(file): sha(file) for file in files})
    parent.base.shared.assert_sources(plan)
    proof = parent.preflight()
    write(PLAN, plan)
    write(REPORT / 'preflight.json', dict(observer=proof, protocol_sha256=sha(PLAN), physical_substeps=0,
        maximum_transitions=d['maximum_controls'] * d['worlds'], archived_transfer_sha256=sha(archived)))
    goal[KEY] = dict(status='zero-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN),
        preflight=rel(REPORT / 'preflight.json'), run_id=run, hardware_io=False)
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
        current_protocol_execution='nominal-roll-support-transfer-frozen', current_evidence=rel(REPORT / 'preflight.json'))
    write(PROJECT, project)
    print('NOMINAL_ROLL_SUPPORT_TRANSFER_FROZEN', sha(PLAN), run, flush=True)


def layout(env, names):
    model = env.sim.mj_model
    (actuator,) = env.scene['robot'].actuators
    joints = np.array([model.joint('robot/' + name).id for name in names])
    lookup = [actuator._target_names.index(name) for name in names]
    ids = [int(actuator._global_ctrl_ids[i]) for i in lookup]
    assert len(set(ids)) == len(joints) == model.nu == 14
    np.testing.assert_array_equal(model.actuator_trnid[ids, 0], joints)
    assert (model.actuator_trntype[ids] == mujoco.mjtTrn.mjTRN_JOINT).all()
    assert list(env.action_manager.get_term('joint_pos')._target_names) == names
    return actuator, dict(joint_names=names, actuator_columns=lookup, motor_ids=ids,
        dof_ids=model.jnt_dofadr[joints].tolist(), qpos_ids=model.jnt_qposadr[joints].tolist())


def qualify(env, obs, plan, actors, folder):
    proof = load_probe.qualify(env, obs, plan, actors, folder)
    _, channels = layout(env, plan['walking']['joint_names'])
    write(folder / 'actuation-layout.json', dict(channels, physical_substeps=0))
    return proof


class SupportActuatorClock(ContactClock):
    def __init__(self, env, plan):
        super().__init__(env, plan['contact_load'])
        self.actuator, self.channels = layout(env, plan['walking']['joint_names'])
        self.delay = self.actuator.apply_delay
        self.delayed = None

    def capture_delay(self, command):
        delayed = self.delay(command)
        if self.first < self.env._sim_step_counter <= self.last:
            self.delayed = delayed.position_target[:, self.channels['actuator_columns']].clone()
        return delayed

    def record(self, original_step):
        capture = self.first < self.env._sim_step_counter <= self.last
        if capture:
            c, env = self.channels, self.env
            d = env.sim.data
            source = env.action_manager.get_term('joint_pos')._processed_actions
            inner = self.actuator._bam_model.actuator
            assert self.delayed is not None and source.shape == self.delayed.shape == (env.num_envs, 14)
            fields = dict(qpos_before=parent.cpu(d.qpos), qvel_before=parent.cpu(d.qvel),
                position_target=parent.cpu(source), delayed_target=parent.cpu(self.delayed),
                filtered_target=parent.cpu(inner.q_target_smooth[:, c['actuator_columns']]),
                firmware_vin=parent.cpu(inner.vin), motor_ctrl=parent.cpu(d.ctrl[:, c['motor_ids']]))
            self.delayed = None
        super().record(original_step)
        if capture:
            self.rows[-1].update(fields, joint_actuator_torque_nm=parent.cpu(d.qfrc_actuator[:, c['dof_ids']]))


def collect(env, plan, actors, folder, observer, watch, ledger):
    clock = SupportActuatorClock(env, plan)
    physical = env.sim.step

    def step():
        clock.record(physical)

    try:
        with patch.object(clock.actuator, 'apply_delay', clock.capture_delay), +             patch.object(env.sim, 'step', step), +             patch.object(env, '_reset_idx', side_effect=AssertionError('Unexpected episode reset')):
            COLLECT(env, plan, actors, folder, observer, watch, ledger)
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
    with patch.multiple(parent, PLAN=PLAN, REPORT=REPORT, KEY=KEY, STAGE=STAGE, qualify=qualify, collect=collect), +         patch.object(feedback, 'components', isolated_components):
        if action == 'score':
            from verify_nominal_skill_composition import score
            return score()
        if action == 'review':
            import review_nominal_skill_composition_contact_load as review
            with patch.object(review, 'KEY', KEY):
                return review.main()
        with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode():
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            {'zero': parent.zero, 'sample': parent.sample}[action]()


if __name__ == '__main__':
    main()
