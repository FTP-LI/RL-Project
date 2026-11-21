"""Freeze and run the nominal roll's bounded neck target comparison."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from check_nominal_pre_roll_standing_snapshot import StandingFork
from collect_nominal_roll_neck_descent_pair import CANDIDATE, ROLES, collect
from local_getup_scale import gpu_sample
from nominal_roll_neck_descent_rule import limit_target
from recover_nominal_skill_composition_zero import isolated_components
from review_nominal_roll_neck_descent_limit import correction
import routed_getup_sequence as feedback
import run_nominal_pre_roll_stand_pair as driver
import run_nominal_skill_composition as base


ROOT, PROJECT = base.ROOT, base.PROJECT
read, rel, sha, write = base.read, base.rel, base.sha, base.write
KEY = 'nominal_roll_neck_descent_pair'
STAGE = 'nominal-roll-neck-descent-limit'
PLAN = ROOT / 'configs/experiments/nominal_roll_neck_descent_pair.json'
DRAFT = PLAN.with_name('nominal_roll_neck_descent_limit_draft.json')
REPORT = base.REPORT / 'neck-descent-pair'
ENTRY = Path(__file__).resolve()
QUALIFY = base.qualify


def freeze():
    project = read(PROJECT)
    task, goal = project['staged_training_request'], project['goal_execution']
    assert base.allowed(project, STAGE + '-preparation') and not task['simulation_running']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists()
    source = goal['nominal_roll_support_transfer']
    assert source['status'] == 'reviewed'
    files = {}
    for label in ('protocol', 'zero', 'result', 'actuator_geometry_review', 'neck_descent_limit_preflight'):
        path = ROOT / source[label]
        assert sha(path) == source[label + '_sha256']
        files[label] = path
    assert task['current_evidence'] == rel(files['neck_descent_limit_preflight'])
    preflight = read(files['neck_descent_limit_preflight'])
    assert preflight['approved_for_actual_cuda_zero'] and preflight['protocol_sha256'] == sha(DRAFT)
    assert preflight['maximum_addition_rad'] <= .02 + 1e-7
    spec = read(DRAFT)
    future = spec['future_dynamics']
    plan = read(files['protocol'])
    plant, d, seven = (plan[k] for k in ('environment', 'composition', 'seven_command'))
    assert future['development_seed'] == plant['seed'] == 1257 and plant['seed'] not in spec['reserved_final_seeds']
    assert future['roles'] == list(ROLES) and d['worlds'] == 896
    shared = future['matched_snapshot_control']
    maximum = shared + len(ROLES) * (d['maximum_controls'] - shared)
    assert shared == d['roll_start_control'] == 600 and maximum == future['maximum_batched_controls']
    assert maximum*d['worlds'] == future['maximum_control_transitions']
    orig = dict(roll=plan['roll_route']['minimum_roll_completion'], complete_total=seven['minimum_overall'],
        each_command=next(iter(seven['minimum_by_command'].values())))
    assert orig == future['unchanged_gates'] and set(seven['minimum_by_command'].values()) == {orig['each_command']}
    initial = files['zero']
    assert sha(initial.with_name('reset.npz')) == read(initial)['reset_sha256']
    run = 'duck_nominal_roll_neck_descent_s1257_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plant['run_id'] = run
    plan.update(stage=STAGE, record=f'records/evaluations/{run}',
        standing_pair=dict(spec, common_snapshot_control=shared, archived_initial=rel(initial), candidate=CANDIDATE,
            original_gates=orig, minimum_roll_gain_rule='net above both repeats and above the absolute reference difference',
            final_complete_retention='at least as many complete routes as each completed reference'),
        maximum_unique_controls=maximum, maximum_physics_substeps=maximum*5,
        maximum_control_transitions=maximum*d['worlds'])
    plan['contact_load'].update(capture_first_control=620, capture_last_control=655, capture_seconds=[12.4, 13.1],
        maximum_batched_substeps=5*d['maximum_controls'], load_threshold_nominal_weight_fraction=.02)
    sources = ('collect_nominal_roll_neck_descent_pair.py', 'nominal_roll_neck_descent_rule.py',
        'verify_nominal_roll_neck_descent_pair.py', 'verify_nominal_pre_roll_stand_pair.py',
        'run_nominal_pre_roll_stand_pair.py', 'review_nominal_roll_neck_descent_limit.py',
        'check_nominal_pre_roll_standing_snapshot.py', 'getup_release_scene_state.py', 'getup_release_fork_state.py',
        'routed_forward_entry_ramp.py', 'check_continuous_forward_fork_state.py')
    plan['source_sha256'].update({rel(path): sha(path) for path in (ENTRY, *(ENTRY.with_name(name) for name in sources))})
    plan['references'].update({rel(path): sha(path) for path in (*files.values(), DRAFT, initial.with_name('reset.npz'))})
    assert not torch.cuda.is_initialized()
    base.base.shared.assert_sources(plan)
    proof = base.preflight()
    REPORT.mkdir()
    write(PLAN, plan)
    write(REPORT / 'preflight.json', dict(protocol_sha256=sha(PLAN), observer=proof, physical_substeps=0,
        baseline_readback_sha256=sha(files['neck_descent_limit_preflight']), maximum_transitions=plan['maximum_control_transitions']))
    goal[KEY] = dict(status='zero-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN),
        preflight=rel(REPORT / 'preflight.json'), run_id=run, hardware_io=False)
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
        current_protocol_execution='nominal-neck-descent-pair-frozen', current_evidence=rel(REPORT / 'preflight.json'))
    write(PROJECT, project)
    print('NOMINAL_NECK_DESCENT_PAIR_FROZEN', sha(PLAN), run, flush=True)


def actual_zero(env, obs, plan, actors, folder):
    receipt = QUALIFY(env, obs, plan, actors, folder)
    pair = plan['standing_pair']
    source = ROOT / pair['archived_initial']
    reference = read(source)
    assert receipt['composition_boundary'] == reference['composition_boundary']
    with (
        np.load(source.with_name('reset.npz'), allow_pickle=False) as then,
        np.load(folder / 'reset.npz', allow_pickle=False) as now,
    ):
        assert then.files == now.files
        for field in then.files:
            np.testing.assert_array_equal(then[field], now[field], err_msg=field)

    neck = plan['walking']['joint_names'].index(pair['changed_joint'])
    low, high = plan['environment']['joint_limits'][neck]
    term = env.action_manager.get_term('joint_pos')
    target = term.previous_target[:8].clone()
    target[:, neck] = (low + high) / 2.
    source_obs = obs['actor'][:8].clone()
    none = torch.zeros(8, dtype=torch.bool, device=env.device)
    active = ~none
    torch.testing.assert_close(limit_target(target, target, source_obs, none, neck, low, high, pair), target, atol=0., rtol=0.)
    fixture = source_obs.clone()
    phi = torch.deg2rad(torch.tensor([150., 160., 180., 220., 245., 255., 200., 200.], device=env.device))
    fixture[:, 1] = torch.tensor([2., 2., 2., 2., 2., 2., -2., 2.], device=env.device)
    fixture[:, 3:6] = torch.stack((phi.sin(), torch.zeros_like(phi), -phi.cos()), dim=1)
    active[-1] = False
    previous = target.clone()
    previous[:, neck] += torch.tensor([.08, .08, .023, .12, -.08, .07, .07, .07], device=env.device)
    applied = limit_target(target, previous, fixture, active, neck, low, high, pair)
    cpu = base.cpu
    predicted = correction(cpu(previous[:, neck]), cpu(target[:, neck]), cpu(fixture[:, 3:6]), cpu(fixture[:, 1]), cpu(active), pair)
    np.testing.assert_allclose(cpu(applied[:, neck] - target[:, neck]), predicted, atol=1e-7, rtol=0.)
    np.testing.assert_array_equal(cpu(applied)[:, np.arange(14) != neck], cpu(target)[:, np.arange(14) != neck])
    assert int(np.count_nonzero(predicted > 1e-5)) == 3 and env.common_step_counter == env._sim_step_counter == 0

    # Initialize only the controller's release history on cloned real observations.
    active_actors = base.load_actors(plan, env.device, lambda: plan['composition']['release_control'])
    getup = active_actors['getup']
    getup(base.base.shared.td('actor', obs['actor'].clone()))
    original = base.Composition(plan, active_actors, env.device)
    resource = gpu_sample()
    fraction = plan['composition']['observer']['gpu_memory_fraction_limit']
    room = int((resource['total_mib'] * fraction - resource['used_mib']) * 1024**2)
    fork = StandingFork(env, original, getup, room)
    baseline = fork.plant.verify()
    assert baseline['counters'] == [0, 0]
    env.sim.data.qpos[0, 0] += .03125
    original.roll.history[0, 0] += .03125
    restored = fork.restore()
    assert restored['physical'] == baseline and restored['controller_sha256'] == fork.expected
    assert env.common_step_counter == env._sim_step_counter == 0
    write(folder / 'neck-rule-qualification.json', dict(physical_substeps=0, real_61_field_observation_checked=True,
        fixture_from_saved_phase_contract=True, positive_cases=int(np.count_nonzero(predicted > 1e-5)),
        noop_after_reset_actuator_target_exact=True, restored=restored))
    return receipt


def run_sample():
    project, _ = driver.checked('sample-ready')
    state = project['goal_execution'][KEY]
    qual = (ROOT / state['zero']).with_name('neck-rule-qualification.json')
    proof = read(qual)
    assert proof['physical_substeps'] == 0 and proof['positive_cases'] == 3 and proof['restored']['physical']['counters'] == [0, 0]
    state.update(neck_rule_qualification=rel(qual), neck_rule_qualification_sha256=sha(qual))
    write(PROJECT, project)
    driver.sample()


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('freeze', 'zero', 'sample', 'score'))
    action = cli.parse_args().action
    if action == 'freeze':
        return freeze()
    with (
        patch.multiple(base, PLAN=PLAN, REPORT=REPORT, KEY=KEY, STAGE=STAGE),
        patch.multiple(driver, PLAN=PLAN, REPORT=REPORT, KEY=KEY, STAGE=STAGE, ROLES=ROLES, collect=collect),
        patch.object(feedback, 'components', isolated_components),
    ):
        if action == 'score':
            from verify_nominal_roll_neck_descent_pair import score
            return score()
        with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode():
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if action == 'zero':
                with patch.object(base, 'qualify', actual_zero):
                    return base.zero()
            return run_sample()


if __name__ == '__main__':
    main()
