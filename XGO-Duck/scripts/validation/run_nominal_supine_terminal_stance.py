"""Compare neck clearance and a coordinated stance target from one CUDA snapshot."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from mujoco_warp._src.types import ContactType
import numpy as np
import torch

from audit_nominal_supine_terminal_stance import KEY, REPORT, attitude
import evaluate_roll_post_getup_training as comparison
from getup_stand_handoff import projected_slew
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence
from routed_getup_sequence import GETUP, STAND
from run_nominal_roll_closed_loop_timestep import CollectorLedger
from verify_nominal_skill_composition import check_video, read_trace


driver, recorder = comparison.driver, comparison.recorder
ROOT, PROJECT = comparison.ROOT, comparison.PROJECT
read, rel, sha, write = comparison.read, comparison.rel, comparison.sha, comparison.write
PREPARATION = 'nominal-pre-roll-standing-admission-preparation'
STAGE = 'nominal-supine-terminal-stance'
PLAN = ROOT / 'configs/experiments/nominal_supine_terminal_stance_pair.json'
ROLES = ('original-getup', 'neck-clearance', 'neck-clearance-and-stance')
LABELS = dict(zip(('shared', *ROLES), ('共同跌倒与起身', '原起身参照', '颈部归位', '颈部归位与腿部姿态'), strict=True))
LEGS = ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle')


def freeze():
    project = read(PROJECT)
    task = project['staged_training_request']
    assert driver.parent.allowed(project, PREPARATION) and not task['simulation_running']
    state = project['goal_execution'][KEY]
    assert state['status'] == 'archive-audit-reviewed' and not PLAN.exists()
    audit = read(ROOT / state['archive_audit'], state['archive_audit_sha256'])
    rules = read(ROOT / state['design'], state['design_sha256'])
    prior = project['goal_execution']['nominal_roll_timestep_closed_loop']
    plan = read(ROOT / prior['protocol'], prior['protocol_sha256'])
    archive = read(ROOT / prior['collection'], prior['collection_sha256'])
    support = read(ROOT / prior['forward_pre_roll_admission'], prior['forward_pre_roll_admission_sha256'])
    start, stop = 400, plan['composition']['roll_start_control']
    assert stop == 600 and plan['composition']['getup_to_stand_control'] == 550
    preview = audit['trigger_window_counts'][str(start)]
    assert preview['target_count'] == 5 and preview['original_roll_entries'] == 0
    for world in audit['target_worlds']:
        row, = (x for x in audit['samples'] if x['world'] == world and x['control'] == start)
        geometry = row['reconstructed_self_contacts_distance_m']
        assert all(geometry[f'head_collision + {side}_foot_collision'] < 0. for side in ('left', 'right'))
    assert audit['physical_steps'] == 0 and plan['environment']['seed'] == rules['seed'] == 1257
    assert rules['original_pre_roll_dwell_controls'] == plan['roll_route']['standing_dwell_controls'] == 25
    assert rules['original_pre_roll_tilt_deg'] == plan['roll_route']['roll']['criteria']['tilt_max_deg'] == 15.
    ref = plan['solver_comparison']['roles'][0]
    assert ref == 'reference-4ms-first'
    mark, = (x for x in archive['branches'][ref]['boundaries'] if x['control'] == stop)
    parent = ROOT / prior['collection']
    source = parent.parent / ref / mark['file']
    assert sha(source) == mark['sha256']
    with np.load(source, allow_pickle=False) as data:
        ids = audit['matched_success_reference_worlds']
        template = np.median(data['previous_target'][ids], axis=0)
    names = plan['walking']['joint_names']
    neck = names.index('neck_pitch')
    assert abs(template[neck] - plan['policy_home']['stand-and-base'][neck]) < .12
    plan.pop('solver_comparison')
    plan['environment']['run_id'] = f'duck-getup-terminal-stance-s1257-{datetime.now(ZoneInfo("Asia/Shanghai")):%Y%m%dT%H%M%S}'
    length = start + len(ROLES) * (stop - start)
    plan.update(stage=STAGE, record=f'records/evaluations/{plan["environment"]["run_id"]}', maximum_unique_controls=length,
        maximum_physics_substeps=length * 5, maximum_control_transitions=length * plan['environment']['num_envs'])
    plan['standing_pair']['common_snapshot_control'] = start
    plan['terminal_stance'] = dict(fork_control=start, end_control=stop, roles=list(ROLES), eligibility=rules['candidate_trigger'],
        eligible_release_pose='supine', stance_template_rad=template.tolist(), template_worlds=ids,
        neck_target_rad=plan['policy_home']['stand-and-base'][neck], stance_allowed_after_clear_reads=5,
        added_hip_pitch_gain=.35, added_hip_pitch_limit_rad=.12, target_rate_rad_s=2.,
        reserved_seeds=rules['reserved_seeds'], required_forward_roll_entry_gain=support['minimum_next_forward_gain'])
    for node, fields in ((state, ('design', 'archive_audit')), (prior, ('protocol','collection','forward_pre_roll_admission'))):
        for field in fields:
            plan['references'][node[field]] = node[field + '_sha256']
    plan['references'].update({rel(source): mark['sha256'], rel(Path(__file__).resolve()):sha(Path(__file__).resolve())})
    driver.parent.base.shared.assert_sources(plan)
    observer = driver.parent.preflight()
    write(PLAN, plan)
    preflight = REPORT / 'pair-preflight.json'
    write(preflight, dict(protocol_sha256=sha(PLAN), new_physical_steps=0, observer=observer,
        maximum_control_transitions=plan['maximum_control_transitions'], eligible_archived_worlds=preview['total_count']))
    state.update(status='sample-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN), preflight=rel(preflight),
        run_id=plan['environment']['run_id'])
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
        current_protocol_execution='supine-terminal-stance-frozen', current_evidence=rel(preflight))
    project.update(current_protocol_execution=task['current_protocol_execution'], current_evidence=task['current_evidence'])
    write(PROJECT, project)
    print('SUPINE_TERMINAL_STANCE_FROZEN', sha(PLAN), plan['environment']['run_id'], flush=True)


def head_foot_contact(env, head, feet):
    raw = env.sim.wp_data
    n = int(raw.nacon.numpy()[0])
    assert 0 <= n < env.guard_slots
    a, b = raw.contact.geom.numpy()[:n].T
    physical = (raw.contact.type.numpy()[:n] & int(ContactType.CONSTRAINT)) != 0
    found = physical & (((a == head) & np.isin(b, feet)) | ((b == head) & np.isin(a, feet)))
    mask = np.zeros(env.num_envs, dtype=bool)
    mask[raw.contact.worldid.numpy()[:n][found]] = True
    return mask


def collect(env, plan, actors, folder, observer, watch, ledger):
    spec, design = plan['terminal_stance'], plan['composition']
    start, end = spec['fork_control'], spec['end_control']
    # Only the captured observation horizon is shortened. The real environment
    # and its time-out, policy routing, sample clocks and gates are the parent.
    short = dict(plan, composition={**design, 'maximum_controls': end})
    names = plan['walking']['joint_names']
    neck = names.index('neck_pitch')
    leg = [names.index(f'{side}_{name}') for side in ('left', 'right') for name in LEGS]
    hips = [names.index(f'{side}_hip_pitch') for side in ('left', 'right')]
    model = env.sim.mj_model
    head = model.geom('robot/head_collision').id
    feet = [model.geom(f'robot/{side}_foot_collision').id for side in ('left', 'right')]
    role, assigned = 'shared', iter(ROLES)
    base_fork, base_ctl = recorder.StandingFork, recorder.Composition
    counters = CollectorLedger(ledger)
    flags = spec['eligibility']
    bounds = torch.as_tensor(plan['environment']['joint_limits'],device=env.device,dtype=torch.float32)
    template = torch.as_tensor(spec['stance_template_rad'],device=env.device,dtype=torch.float32)
    signs = torch.tensor([1., -1.],device=env.device)

    class TerminalStance(base_ctl):
        def __init__(self, *args):
            super().__init__(*args)
            self.dwell = np.zeros(env.num_envs, dtype=np.int32)
            self.eligible = np.zeros(env.num_envs, dtype=bool)
            self.release_supine = self.eligible.copy()
            self.records = {}

        def begin(self, arm):
            rows = np.flatnonzero(self.eligible)
            self.worlds_tracked = rows
            self.clearance = np.zeros(len(rows), dtype=np.int32)
            self.stance = self.clearance.astype(bool)
            self.records[arm] = []
            self.requested_mask = np.zeros((len(rows),len(names)), dtype=bool)

        def observe_roll(self, k, frame):
            super().observe_roll(k, frame)
            if k == design['release_control']:
                poses = recorder.classify_release(recorder.projected_gravity(frame['quaternion']),
                    plan['environment']['criteria']['release_alignment_min'])
                self.release_supine[:] = np.asarray(poses) == spec['eligible_release_pose']
            if start-flags['minimum_dual_foot_dwell_controls'] < k <= start:
                pitch, roll, angle = attitude(frame['quaternion']).T
                lo, hi = flags['root_tilt_deg']
                quiet = (self.release_supine & frame['active'] & frame['foot_contact'].all(axis=-1) &
                    frame['joint_envelope_ok'] & (pitch > 0.) & (angle >= lo) & (angle <= hi) &
                    (abs(roll) <= flags['maximum_absolute_roll_deg']) &
                    (np.linalg.norm(frame['angular_velocity'],axis=1) <= flags['maximum_angular_speed_rad_s']))
                self.dwell = np.where(quiet,self.dwell+1,0)
                if k == start:
                    self.eligible[:] = (self.dwell >= flags['minimum_dual_foot_dwell_controls']) & head_foot_contact(env,head,feet)
                    self.pitch = pitch.copy()
                    count = int(self.eligible[plan['seven_command']['groups']['forward']].sum())
                    assert count >= spec['required_forward_roll_entry_gain'], 'Insufficient live coverage of the target failure'
                    print('SUPINE_TERMINAL_ELIGIBLE',int(self.eligible.sum()),'forward',count,flush=True)
            if role != 'shared' and k > start:
                ids = self.worlds_tracked
                contact = head_foot_contact(env,head,feet)[ids]
                dual = frame['foot_contact'][ids].all(axis=-1)
                self.clearance = np.where(~contact & dual,self.clearance+1,0)
                self.stance |= self.clearance >= spec['stance_allowed_after_clear_reads']
                self.pitch = attitude(frame['quaternion'])[:,0]
                self.records[role].append(dict(control=k, pitch_deg=self.pitch[ids], head_foot_contact=contact,
                    dual_foot_contact=dual, target_overridden=self.requested_mask.copy(),
                    standing_actor=recorder.cpu(self.commands.mode)[ids] == STAND,
                    motor_nm=recorder.cpu(env.sim.data.ctrl)[ids]))

        def apply(self, term, previous, control, action):
            super().apply(term,previous,control,action)
            if role == 'shared':
                return
            self.requested_mask[:] = False
            if role == ROLES[0]:
                return
            ids = self.worlds_tracked
            getup = action.mode[ids] == GETUP
            self.requested_mask[:,neck] = getup
            if role == ROLES[-1]:
                self.requested_mask[:,leg] = (getup & self.stance)[:,None]
            selected = torch.as_tensor(ids,device=env.device)
            mask = torch.as_tensor(self.requested_mask,device=env.device)
            desired = template.expand(len(ids),-1).clone()
            desired[:,neck] = spec['neck_target_rad']
            error = torch.as_tensor(np.deg2rad(self.pitch[ids]),device=env.device,dtype=desired.dtype)
            adjustment = (error * spec['added_hip_pitch_gain']).clamp(-spec['added_hip_pitch_limit_rad'],spec['added_hip_pitch_limit_rad'])
            desired[:,hips] += adjustment[:,None] * signs
            target,clamped = projected_slew(previous[selected],desired,bounds,env.step_dt,spec['target_rate_rad_s'])
            assert (torch.abs(target-previous[selected]) <= spec['target_rate_rad_s']*env.step_dt+3e-7).all()
            for field,values in (('requested_target',desired),('projected_destination',clamped),
                                 ('_processed_actions',target),('previous_target',target)):
                current = getattr(term,field)
                current[selected] = torch.where(mask,values,current[selected])

    class SameStart(base_fork):
        def restore(self):
            nonlocal role
            state = super().restore()
            assert state['physical']['counters'] == [start,start*5]
            role = next(assigned)
            # Restore the detector's initial pitch from the frozen actual plant.
            self.control.pitch = attitude(recorder.cpu(env.sim.data.qpos)[:,3:7])[:,0]
            self.control.begin(role)
            return state

    def publish(path, values):
        previous_publish(path,dict(values,purpose=f'仰卧起身末段对照｜{LABELS[values["phase"]]}'))

    def build(*args):
        nonlocal controller
        controller = TerminalStance(*args)
        return controller

    previous_publish, controller = observer.publish, None
    try:
        with (patch.multiple(recorder, ROLES=ROLES, LABELS=LABELS, CANDIDATE=None, StandingFork=SameStart, Composition=build),
              patch.object(observer,'publish',publish)):
            recorder.collect(env,short,actors,folder,observer,watch,counters)
    finally:
        ledger.update(counters)
        if controller is not None and controller.records:
            ledger['eligible_worlds'] = np.flatnonzero(controller.eligible).tolist()
            for arm, rows in controller.records.items():
                if rows:
                    evidence = folder / arm / 'terminal-stance.npz'
                    np.savez_compressed(evidence,**{key:np.stack([row[key] for row in rows]) for key in rows[0]})
                    ledger['branches'][arm]['terminal_stance_sha256'] = sha(evidence)


def score():
    project,plan = driver.checked('score-pending')
    state = project['goal_execution'][KEY]
    folder = ROOT / plan['record']
    raw = read(folder/'collection.json',state['collection_sha256'])
    initial_path = folder/'initial.npz'
    assert sha(initial_path) == raw['initial_sha256'] and raw['actor_weights_unchanged']
    assert raw['transitions'] == plan['maximum_control_transitions'] and raw['resets'] == 1
    assert not (raw['hardware_io'] or raw['learning_updates'] or raw['reserved_seed_used'])
    with np.load(initial_path,allow_pickle=False) as data:
        initial = dict(data)
    spec, d = plan['terminal_stance'], plan['composition']
    start,end = spec['fork_control'],spec['end_control']
    fields = ('mode','roll_phase','qpos',*STATE_FIELDS)
    common = read_trace(folder/'shared',raw['prefix']['chunks'],1,start,fields,sha)
    fork = read(folder/'fork.json',raw['fork_sha256'])
    result, masks, original = {}, {}, None
    tracked = np.asarray(raw['eligible_worlds'],dtype=int)
    assert tracked.size > 0
    assigned = np.asarray(plan['seven_command']['groups']['forward'],dtype=int)
    for arm in ROLES:
        record = raw['branches'][arm]
        path = folder/arm
        assert read(path/'restored-fork.json',record['restore_sha256']) == {k:fork[k] for k in ('physical','controller_sha256')}
        readback = path/'terminal-stance.npz'
        assert sha(readback) == record['terminal_stance_sha256']
        with np.load(readback,allow_pickle=False) as evidence:
            measured = dict(evidence)
        np.testing.assert_array_equal(measured['control'],np.arange(start+1,end+1))
        branch = read_trace(path,record['chunks'],start+1,end,fields,sha)
        trace = {key: np.concatenate((common[key],branch[key])) for key in fields}
        sequence = Sequence(plan['roll_route'])
        for k in range(d['getup_to_stand_control'],end+1):
            frame = {key:trace[key][k-1] for key in STATE_FIELDS}
            if k == end:
                frame['active'] = frame['active'] & (trace['mode'][k-1] == STAND)
            sequence.update(k-d['getup_to_stand_control'],frame)
            np.testing.assert_array_equal(sequence.phase,trace['roll_phase'][k-1])
        mark, = (x for x in record['boundaries'] if x['control'] == start)
        assert sha(path/mark['file']) == mark['sha256']
        with np.load(path/mark['file'],allow_pickle=False) as data:
            first = {k:data[k] for k in ('raw','mode','previous_target','observed')}
            first['motor_first_substep'] = data['motor'][0]
        if original is None:
            original = first
        for key,value in first.items():
            np.testing.assert_array_equal(value,original[key],err_msg=f'{arm}/{key}')
        moved = measured['target_overridden']
        if arm == ROLES[0]:
            assert not moved.any()
        else:
            assert moved.any()
        masks[arm] = sequence.phase == ROLL
        result[arm] = dict(roll_entry=int(masks[arm].sum()), forward_roll_entry=int(masks[arm][assigned].sum()),
            triggered_roll_entry=int(masks[arm][tracked].sum()), triggered_standing_actor=int(measured['standing_actor'][-1].sum()),
            tracked_final_pitch_deg=measured['pitch_deg'][-1].tolist(),
            tracked_clear_of_head_foot_contact_at_end=int((~measured['head_foot_contact'][-1]).sum()),
            maximum_measured_motor_magnitude_nm=float(abs(measured['motor_nm']).max()),
            controls_with_override=int(moved.any(axis=(1,2)).sum()),
            video=check_video(path,dict(record,camera_command=raw['camera_command']),trace,initial))
    candidate, reference = masks[ROLES[-1]],masks[ROLES[0]]
    differences = dict(added_roll_entry=np.flatnonzero(candidate & ~reference).tolist(),
        lost_roll_entry=np.flatnonzero(reference & ~candidate).tolist(),
        net_forward_roll_entry=result[ROLES[-1]]['forward_roll_entry']-result[ROLES[0]]['forward_roll_entry'])
    advances = differences['net_forward_roll_entry'] >= spec['required_forward_roll_entry_gain'] and not differences['lost_roll_entry']
    out = REPORT/'pair-analysis.json'
    assert not out.exists()
    write(out,dict(protocol_sha256=sha(PLAN),collection_sha256=state['collection_sha256'],seed=plan['environment']['seed'],
        roles=result,eligible_worlds=tracked.tolist(),comparison=differences,advance_to_integrated_trial=advances,
        original_pre_roll_gate_verified=True, first_policy_input_and_first_delayed_motor_torque_exact=True,
        transitions=raw['transitions'], learning_updates=0, hardware_io=False, reserved_seed_used=False))
    state.update(status='reviewed',pair_analysis=rel(out),pair_analysis_sha256=sha(out),advance_to_integrated_trial=advances)
    task = project['staged_training_request']
    task.update(current_protocol_execution='supine-terminal-stance-reviewed',current_evidence=rel(out))
    project.update(current_protocol_execution=task['current_protocol_execution'],current_evidence=rel(out))
    write(PROJECT,project)
    print('SUPINE_TERMINAL_STANCE_SCORED',sha(out),{k:v['roll_entry'] for k,v in result.items()},differences,flush=True)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','sample','score'))
    action = cli.parse_args().action
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode(),
          patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,collect=collect),
          patch.object(driver.parent,'PLAN',PLAN),patch.object(comparison.feedback,'components',driver.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'sample':driver.sample,'score':score}[action]()


if __name__ == '__main__':
    main()
