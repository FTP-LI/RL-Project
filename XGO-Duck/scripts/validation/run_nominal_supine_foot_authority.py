"""Measure actual pre-touch foot travel from matched CUDA state and policy history."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from audit_nominal_supine_terminal_stance import attitude
from preflight_nominal_supine_unloaded_foot import FootPlacement, SIDES
from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence
from routed_getup_sequence import GETUP, STAND
from run_nominal_roll_closed_loop_timestep import CollectorLedger
from run_nominal_supine_terminal_stance import comparison, driver, recorder
from verify_nominal_skill_composition import check_video, read_trace


ROOT, PROJECT = comparison.ROOT, comparison.PROJECT
read, rel, sha, write = comparison.read, comparison.rel, comparison.sha, comparison.write
KEY = 'nominal_supine_foot_authority'
STAGE = 'nominal-supine-foot-authority'
PLAN = ROOT/'configs/experiments/nominal_supine_foot_authority.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/supine-pre-bilateral-support/foot-authority'
ROLES = ('reference-a', 'reference-b', 'unloaded-foot-placement')
LABELS = dict(zip(('shared', *ROLES), ('共同跌倒和起身', '原策略一', '原策略二', '未承重脚前移'), strict=True))
ENTRY = Path(__file__).resolve()


def freeze():
    project = read(PROJECT)
    task = project['staged_training_request']
    node = project['goal_execution'][KEY]
    assert driver.parent.allowed(project, STAGE+'-preparation') and not task['simulation_running']
    assert node['status'] == 'design-registered' and not PLAN.exists() and not REPORT.exists()
    draft = ROOT/node['design']
    design = read(draft)
    archive = project['goal_execution']['nominal_supine_pre_bilateral_support']
    timing_path = ROOT/archive['timing_analysis']
    timing = read(timing_path, design['timing_analysis_sha256'])
    assert not timing['qualifies_fixed_time_trial']
    timelines = [cohort['control_measurements'] for cohort in tuple(timing['cohorts'].values())[1:]]
    overlap = next(a['control'] for a, b in zip(*timelines, strict=True)
                   if a['useful_forward_misses'] >= 2 and b['useful_forward_misses'] >= 2)
    assert overlap == design['decision_control'] and tuple(design['roles']) == ROLES
    source = project['goal_execution']['nominal_supine_early_support']
    plan = read(ROOT/source['protocol'], source['protocol_sha256'])
    plan.pop('early_takeover')
    d = plan['composition']
    assert d['roll_start_control'] == design['end_control'] and d['worlds'] == design['worlds']
    assert plan['environment']['seed'] == design['seed'] not in design['reserved_seeds']
    assert plan['roll_route']['standing_dwell_controls'] == 25
    assert plan['roll_route']['roll']['criteria']['tilt_max_deg'] == 15
    controls = overlap+len(ROLES)*(design['end_control']-overlap)
    assert controls*5 == design['maximum_physics_substeps']
    assert controls*d['worlds'] == design['maximum_control_transitions']
    old_proof = read(ROOT/archive['unloaded_foot_preflight'], archive['unloaded_foot_preflight_sha256'])
    geometry_path = ROOT/old_proof['protocol']
    geometry = read(geometry_path, old_proof['protocol_sha256'])
    alignment = read(ROOT/source['design'], source['design_sha256'])['alignment']
    run = f'duck-getup-unloaded-foot-authority-s{design["seed"]}-'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plan['environment']['run_id'] = run
    plan['standing_pair']['common_snapshot_control'] = overlap
    plan.update(stage=STAGE, record=f'records/evaluations/{run}', maximum_unique_controls=controls,
                maximum_control_transitions=design['maximum_control_transitions'], maximum_physics_substeps=controls*5,
                foot_authority={**design, 'geometry':geometry, 'alignment':alignment})
    files = (draft, timing_path, geometry_path, ROOT/archive['unloaded_foot_preflight'], ROOT/source['protocol'], ENTRY,
             ENTRY.with_name('preflight_nominal_supine_unloaded_foot.py'))
    plan['references'].update({rel(path):sha(path) for path in files})
    driver.parent.base.shared.assert_sources(plan)
    observation = driver.parent.preflight()
    REPORT.mkdir()
    write(PLAN, plan)
    certificate = REPORT/'preflight.json'
    write(certificate, dict(protocol_sha256=sha(PLAN), observation=observation, comparison_controls=controls,
                            archived_timing_sha256=sha(timing_path), physical_substeps=0))
    node.update(status='sample-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN), preflight=rel(certificate), run_id=run)
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
                current_protocol_execution='nominal-supine-foot-authority-frozen', current_evidence=rel(certificate))
    project.update(current_protocol_execution=task['current_protocol_execution'], current_evidence=rel(certificate))
    write(PROJECT, project)
    print('SUPINE_FOOT_AUTHORITY_FROZEN', sha(PLAN), run, controls, flush=True)


def collect(env, plan, actors, folder, observer, watch, ledger):
    spec, design = plan['foot_authority'], plan['composition']
    fork_at, end = spec['decision_control'], spec['end_control']
    short = dict(plan, composition={**design, 'maximum_controls':end})
    role, roles = 'shared', iter(ROLES)
    counters = CollectorLedger(ledger)
    planner = FootPlacement(env.sim.mj_model, spec['geometry'])
    names = plan['walking']['joint_names']
    joint_limits = torch.as_tensor(plan['environment']['joint_limits'], device=env.device, dtype=torch.float32)
    pulse = spec['geometry']['target_pulse']
    rate = spec['maximum_residual_rate_rad_s']*env.step_dt
    base_fork, base_control = recorder.StandingFork, recorder.Composition

    class FootPlacementController(base_control):
        def __init__(self, *args):
            super().__init__(*args)
            self.supine = np.zeros(env.num_envs, dtype=bool)
            self.dual_reads = np.zeros(env.num_envs, dtype=np.int16)
            self.confirmed = self.supine.copy()
            self.plan_rows = []
            self.target_log = []
            self.worlds = None

        def observe_roll(self, k, frame):
            super().observe_roll(k, frame)
            observed = frame['foot_contact']
            if k == design['release_control']:
                gravity = recorder.projected_gravity(frame['quaternion'])
                alignment = plan['environment']['criteria']['release_alignment_min']
                self.supine = np.asarray(recorder.classify_release(gravity, alignment)) == 'supine'
            if design['release_control'] <= k <= fork_at:
                condition = spec['alignment']
                _, roll, tilt = attitude(frame['quaternion']).T
                dual = (frame['active'] & self.supine & observed.all(axis=1)
                        & (tilt <= condition['maximum_root_tilt_deg']) & (abs(roll) <= condition['maximum_absolute_roll_deg']))
                self.dual_reads = np.where(dual, self.dual_reads+1, 0)
                self.confirmed |= self.dual_reads >= condition['minimum_dual_foot_reads']
            if k == fork_at:
                for world in np.flatnonzero(frame['active'] & self.supine & ~self.confirmed):
                    placement, _ = planner.estimate(frame['qpos'][world], observed[world])
                    if placement is not None:
                        self.plan_rows.append(dict(world=int(world), **placement))
                self.worlds = np.array([row['world'] for row in self.plan_rows], dtype=int)
                self.feet = np.array([SIDES.index(row['foot']) for row in self.plan_rows], dtype=int)
                self.selected = torch.as_tensor(self.worlds, device=env.device)
                joints = [[names.index(name) for name in row['joint_offsets_rad']] for row in self.plan_rows]
                self.joints = torch.as_tensor(joints, device=env.device, dtype=torch.long).reshape(-1, 3)
                values = [list(row['joint_offsets_rad'].values()) for row in self.plan_rows]
                self.offset = torch.as_tensor(values, device=env.device, dtype=torch.float32).reshape(-1, 3)
            if role == ROLES[-1] and fork_at < k < fork_at+spec['observation_window_controls']:
                self.touched |= observed[self.worlds, self.feet] | ~frame['active'][self.worlds]

        def begin(self):
            self.touched = np.zeros(len(self.worlds), dtype=bool)
            self.residual = torch.zeros_like(self.offset)

        def apply(self, term, previous, control, action):
            super().apply(term, previous, control, action)
            index = control-fork_at
            if role != ROLES[-1] or not 0 <= index < spec['observation_window_controls']:
                return
            gain = pulse[index] if index < len(pulse) else 0.
            # Freeze the geometric choice at the shared state. A new touch
            # cancels the pulse and ramps the residual back to zero.
            free = (action.mode[self.worlds] == GETUP) & ~self.touched
            requested = self.offset*gain*torch.as_tensor(free, device=env.device)[:, None]
            self.residual += (requested-self.residual).clamp(-rate, rate)
            rows = self.selected[:, None]
            base = term._processed_actions[rows, self.joints]
            lower, upper = joint_limits[self.joints].unbind(-1)
            proposed = (base+self.residual).clamp(lower, upper)
            target = torch.where((base >= lower) & (base <= upper), proposed, base)
            for field in ('requested_target', 'projected_destination', '_processed_actions', 'previous_target'):
                getattr(term, field)[rows, self.joints] = target
            self.target_log.append(dict(completed_control=control+1, baseline=recorder.cpu(base), actual=recorder.cpu(target),
                                        planned_residual=recorder.cpu(self.residual), unloaded=free))

    class SharedFootState(base_fork):
        def restore(self):
            nonlocal role
            result = super().restore()
            role = next(roles)
            self.control.begin()
            return result

    publish = observer.publish
    controller = None

    def build(*args):
        nonlocal controller
        controller = FootPlacementController(*args)
        return controller

    def labeled(path, state):
        publish(path, dict(state, purpose=f'仰卧起身足部响应｜{LABELS[state["phase"]]}'))

    try:
        with (patch.multiple(recorder, ROLES=ROLES, LABELS=LABELS, CANDIDATE=None, StandingFork=SharedFootState, Composition=build),
              patch.object(observer, 'publish', labeled)):
            recorder.collect(env, short, actors, folder, observer, watch, counters)
    finally:
        ledger.update(counters)
        if controller is not None and controller.worlds is not None:
            placement = folder/'foot-placement.json'
            write(placement, dict(completed_control=fork_at, candidates=controller.plan_rows))
            ledger['foot_placement_sha256'] = sha(placement)
            if controller.target_log:
                target = folder/ROLES[-1]/'foot-targets.npz'
                np.savez_compressed(target, **{key:np.stack([row[key] for row in controller.target_log])
                                              for key in controller.target_log[0]})
                ledger['target_readback_sha256'] = sha(target)


def verify_arm(plan, folder, arm, entry, fork, common, initial):
    start, end = (plan['foot_authority'][name] for name in ('decision_control', 'end_control'))
    path = folder/arm
    assert read(path/'restored-fork.json', entry['restore_sha256']) == {k:fork[k] for k in ('physical', 'controller_sha256')}
    leaf = read_trace(path, entry['chunks'], start+1, end, tuple(common), sha)
    trace = {k:np.concatenate((common[k], leaf[k])) for k in common}
    boundary = plan['composition']['getup_to_stand_control']
    sequence = Sequence(plan['roll_route'])
    for k in range(boundary, end+1):
        frame = {key:trace[key][k-1] for key in STATE_FIELDS}
        if k == end:
            frame['active'] = frame['active'] & (trace['mode'][k-1] == STAND)
        sequence.update(k-boundary, frame)
        np.testing.assert_array_equal(sequence.phase, trace['roll_phase'][k-1])
    mark, = (item for item in entry['boundaries'] if item['control'] == start)
    assert sha(path/mark['file']) == mark['sha256']
    with np.load(path/mark['file'], allow_pickle=False) as values:
        first = dict(values)
    admission = sequence.phase == ROLL
    clip = check_video(path, entry, trace, initial)
    return leaf, first, admission, clip


def physical_response(spec, leaves, common, candidates, applied):
    worlds = np.array([row['world'] for row in candidates], dtype=int)
    feet = np.array([SIDES.index(row['foot']) for row in candidates], dtype=int)
    n = spec['observation_window_controls']
    directions = rotation(common['quaternion'][-1, worlds])[:, :2, 0]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    def measure(leaf):
        foot = leaf['foot_positions'][:n, worlds, feet, :2].astype(float)
        body = leaf['qpos'][:n, worlds, :2].astype(float)
        along = 1000.*np.einsum('twi,wi->tw', foot-body, directions)
        contact = leaf['foot_contact'][:n, worlds, feet]
        return along, contact
    (a, ca), (b, cb), (c, cc) = [measure(leaf) for leaf in leaves]
    before_touch = ~np.logical_or.accumulate(ca | cb | cc, axis=0)
    after_request = np.logical_or.accumulate(applied, axis=0)
    observed = before_touch & after_request
    gain = c-a
    jitter = np.abs(b-a)
    surplus = gain-jitter
    achieved = observed & (surplus >= spec['mechanical_margin_mm'])
    response = []
    for i, candidate in enumerate(candidates):
        rows = np.flatnonzero(observed[:, i])
        row = rows[np.argmax(surplus[rows, i])] if len(rows) else None
        response.append(dict(world=candidate['world'], foot=candidate['foot'], target_changed=bool(applied[:, i].any()),
                             before_touch_reads=len(rows), reaches_margin=bool(achieved[:, i].any()),
                             at_control=None if row is None else int(spec['decision_control']+row+1),
                             relative_gain_mm=None if row is None else float(gain[row, i]),
                             reference_gap_mm=None if row is None else float(jitter[row, i])))
    return response


def score():
    project, plan = driver.checked('score-pending')
    node = project['goal_execution'][KEY]
    folder = ROOT/plan['record']
    raw = read(folder/'collection.json', node['collection_sha256'])
    spec = plan['foot_authority']
    assert raw['actor_weights_unchanged'] and raw['resets'] == 1 and not raw['reserved_seed_used']
    assert raw['transitions'] == plan['maximum_control_transitions']
    assert raw['physics_substeps'] == plan['maximum_physics_substeps']
    assert sha(folder/'initial.npz') == raw['initial_sha256']
    with np.load(folder/'initial.npz', allow_pickle=False) as data:
        initial = dict(data)
    fork = read(folder/'fork.json', raw['fork_sha256'])
    start = spec['decision_control']
    fields = tuple(dict.fromkeys(('mode', 'roll_phase', 'qpos', 'foot_positions', *STATE_FIELDS)))
    common = read_trace(folder/'shared', raw['prefix']['chunks'], 1, start, fields, sha)
    placement = read(folder/'foot-placement.json', raw['foot_placement_sha256'])
    assert placement['completed_control'] == start
    candidates = placement['candidates']
    worlds = np.array([item['world'] for item in candidates], dtype=int)
    names = plan['walking']['joint_names']
    joints = np.array([[names.index(name) for name in item['joint_offsets_rad']]
                       for item in candidates], dtype=int).reshape(-1, 3)
    targets = folder/ROLES[-1]/'foot-targets.npz'
    assert sha(targets) == raw['target_readback_sha256']
    with np.load(targets, allow_pickle=False) as data:
        applied = (data['actual'] != data['baseline']).any(axis=-1)
        first_base, first_target = (data[key][0] for key in ('baseline', 'actual'))
        np.testing.assert_array_equal(data['completed_control'], start+np.arange(1, spec['observation_window_controls']+1))
        max_slew = np.max(np.abs(np.diff(data['planned_residual'], axis=0, prepend=0.)), initial=0.)
        assert max_slew <= spec['maximum_residual_rate_rad_s']*.02+3e-7
    arms = []
    for arm in ROLES:
        entry = dict(raw['branches'][arm], camera_command=raw['camera_command'])
        arms.append(verify_arm(plan, folder, arm, entry, fork, common, initial))
    reference = arms[0][1]
    for _, mark, _, _ in arms:
        for key in ('observed', 'actor_input', 'raw', 'previous_target'):
            np.testing.assert_array_equal(mark[key], reference[key])
        np.testing.assert_array_equal(mark['motor'][0], reference['motor'][0])
        np.testing.assert_array_equal(mark['actual_target'], mark['expected_articulation_target'])
    first_ref, first_repeat, first_applied = [arm[1]['actual_target'] for arm in arms]
    np.testing.assert_array_equal(first_ref, first_repeat)
    np.testing.assert_array_equal(first_ref[worlds[:, None], joints], first_base)
    reconstructed = first_ref.copy()
    reconstructed[worlds[:, None], joints] = first_target
    np.testing.assert_array_equal(reconstructed, first_applied)
    delta = np.any(arms[-1][1]['motor'][1:] != reference['motor'][1:], axis=(0, 2))
    response = physical_response(spec, [x[0] for x in arms], common, candidates, applied)
    forward = np.array(plan['seven_command']['groups']['forward'])
    results = {name:dict(roll_admitted=int(passes.sum()), forward_group_roll_admitted=int(passes[forward].sum()),
                         selected_roll_admitted=int(passes[worlds].sum()), video=clip)
               for name, (_, _, passes, clip) in zip(ROLES, arms, strict=True)}
    output = REPORT/'analysis.json'
    assert not output.exists()
    write(output, dict(protocol_sha256=sha(PLAN), collection_sha256=node['collection_sha256'], hardware_io=False, learning_updates=0,
                       transitions=raw['transitions'], fork_control=start, verified_snapshot_fields=len(fork['fields']),
                       roles=results, planned_placements=len(candidates), actual_target_changed=sum(x['target_changed'] for x in response),
                       first_later_motor_change_on_treated=int((delta[worlds]&applied[0]).sum()),
                       common_unloaded_response_count=sum(x['before_touch_reads'] > 0 for x in response),
                       exceeding_reference_gap_by_one_mm=sum(x['reaches_margin'] for x in response), response=response))
    node.update(status='reviewed', analysis=rel(output), analysis_sha256=sha(output))
    task = project['staged_training_request']
    task.update(current_protocol_execution='nominal-supine-foot-authority-reviewed', current_evidence=rel(output))
    project.update(current_protocol_execution=task['current_protocol_execution'], current_evidence=rel(output))
    write(PROJECT, project)
    print('SUPINE_FOOT_AUTHORITY_SCORED', sha(output), results,
          sum(x['reaches_margin'] for x in response), len(candidates), flush=True)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('freeze', 'sample', 'score'))
    action = cli.parse_args().action
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode(),
          patch.multiple(driver, KEY=KEY, STAGE=STAGE, PLAN=PLAN, REPORT=REPORT, ROLES=ROLES, collect=collect),
          patch.object(driver.parent, 'PLAN', PLAN), patch.object(comparison.feedback, 'components', driver.isolated_components)):
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze, 'sample':driver.sample, 'score':score}[action]()


if __name__ == '__main__':
    main()
