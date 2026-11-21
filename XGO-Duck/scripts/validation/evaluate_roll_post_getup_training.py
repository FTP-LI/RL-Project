"""Compare the official actor and the two learned actors at one real roll entry."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import collect_nominal_pre_roll_stand_pair as recorder
from nominal_skill_composition_controller import ROLLING
from record_io import write
import routed_getup_sequence as feedback
import run_nominal_pre_roll_stand_pair as driver
from run_roll_post_getup_training import KEY as TRAIN_KEY, PROJECT, ROOT, checked, read, rel, sha
import verify_nominal_pre_roll_stand_pair as verifier
from verify_nominal_skill_composition import PHYSICAL, read_trace
from roll_stand_walk_sequence import STATE_FIELDS


KEY = 'roll_post_getup_action_comparison'
STAGE = 'nominal-roll-post-getup-action-comparison'
PLAN = ROOT / 'configs/experiments/roll_post_getup_action_comparison.json'
REPORT = ROOT / 'records/reports/nominal-skill-composition/post-getup-roll-curriculum/action-comparison'
ROLES = ('official-reference', 'official-standing-reset', 'post-getup-start-curriculum')
LABELS = dict(zip(('shared', *ROLES), ('共同跌倒、起身及站立', '官方滚翻模型', '原始站立重置训练组', '起身后启动状态课程训练组'), strict=True))
ROLL_MODEL = 'official-forward-roll'
EVIDENCE_FIELDS = tuple(dict.fromkeys((*PHYSICAL, *STATE_FIELDS, 'roll_phase', 'roll_ready', 'base_stand_reads', 'original_feet')))


def freeze():
    project, task, learning, _ = checked()
    assert learning['status'] == 'action-evaluation-ready' and not (task['learning_running'] or task['simulation_running'])
    assert not PLAN.exists() and not REPORT.exists()
    goal = project['goal_execution']
    prior = goal['nominal_roll_neck_descent_pair']
    old_plan = ROOT / prior['protocol']
    old_evaluation = ROOT / prior['result']
    old = read(old_evaluation, prior['result_sha256'])
    plan = read(old_plan, prior['protocol_sha256'])
    assert old['matched_reference_repeat']['roll']['net'] == 4
    plan.pop('contact_load', None)
    assert plan['environment']['seed'] == old['seed'] == 1257 and plan['composition']['roll_start_control'] == 600
    results = ROOT / learning['training_results']
    learned = read(results, learning['training_results_sha256'])['results']
    models = {ROLES[0]: plan['models'][ROLL_MODEL]}
    for role in ROLES[1:]:
        proof = learned[role]
        model = ROOT / proof['model']
        assert sha(model) == proof['model_sha256'] and proof['export']['metadata_preserved']
        models[role] = dict(model=proof['model'], model_sha256=proof['model_sha256'], seed=plan['environment']['seed'])
    current = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-roll-post-getup-actor-comparison-s1257-{current}'
    start = plan['composition']['roll_start_control']
    maximum = start + len(ROLES) * (plan['composition']['maximum_controls'] - start)
    prior_zero = ROOT / prior['zero']
    assert sha(prior_zero) == prior['zero_sha256']
    initial = prior_zero.with_name('reset.npz')
    assert sha(initial) == read(prior_zero)['reset_sha256']
    plan['environment']['run_id'] = run
    plan.update(stage=STAGE, record=f'records/evaluations/{run}', maximum_unique_controls=maximum,
        maximum_control_transitions=maximum * old['worlds'], maximum_physics_substeps=5 * maximum,
        standing_pair=dict(common_snapshot_control=start, archived_initial=rel(prior_zero)),
        roll_policy_comparison=dict(models=models, minimum_net_gain_over_official=abs(old['matched_reference_repeat']['roll']['net'])+1,
            minimum_net_gain_over_trained_control=1, confirmation_seeds=[2256, 2246]))
    files = (old_plan, old_evaluation, results, prior_zero, initial, Path(__file__).resolve(),
        Path(recorder.__file__).resolve(), Path(verifier.__file__).resolve(), Path(driver.__file__).resolve(),
        *(ROOT / item['model'] for item in models.values()))
    plan['references'].update({rel(path): sha(path) for path in files})
    driver.parent.base.shared.assert_sources(plan)
    REPORT.mkdir()
    write(PLAN, plan)
    write(REPORT / 'preflight.json', dict(plan_sha256=sha(PLAN), observer=driver.parent.preflight(), new_physical_steps=0))
    goal[KEY] = dict(status='sample-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN), run_id=run)
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
        current_protocol_execution='post-getup-action-comparison-frozen', current_evidence=rel(REPORT / 'preflight.json'))
    write(PROJECT, project)
    print('ROLL_POST_GETUP_ACTION_COMPARISON_FROZEN', sha(PLAN), flush=True)


def collect(env, plan, actors, folder, observer, watch, ledger):
    actors = dict(actors)
    models = plan['roll_policy_comparison']['models']
    session = driver.parent.onnx_session(ROOT / models[ROLES[0]]['model'])
    metadata = session.get_modelmeta().custom_metadata_map
    x = env.obs_buf['actor'][:32].clone()
    x[:, 6:20] += torch.as_tensor(plan['environment']['model_home'], device=env.device) - torch.as_tensor(
        plan['policy_home'][ROLL_MODEL], device=env.device)
    x[:, 34:] = 0.
    admitted = {}
    for role in ROLES:
        model = models[role]
        loaded = driver.parent.onnx_session(ROOT / model['model'])
        assert loaded.get_modelmeta().custom_metadata_map == metadata
        actors[role] = actors[ROLL_MODEL] if role == ROLES[0] else driver.parent.route.guarded_actor(model, env.device)
        prediction = driver.parent.cpu(actors[role](driver.parent.base.shared.td('actor', x)))
        expected = np.concatenate([loaded.run(None, {'obs': row[None]})[0] for row in driver.parent.cpu(x)])
        np.testing.assert_allclose(prediction, expected, atol=2e-5, rtol=2e-5)
        admitted[role] = dict(model_sha256=model['model_sha256'], max_abs_error=float(np.abs(prediction - expected).max()))
    assert env.common_step_counter == env._sim_step_counter == 0
    write(folder / 'roll-actor-admission.json', dict(models=admitted, controls=0, physical_substeps=0, observations=len(x)))

    parent_fork = recorder.StandingFork
    assigned = iter(ROLES)
    activations = []

    class RollPolicyFork(parent_fork):
        def restore(self):
            state = super().restore()
            role = next(assigned)
            assert state['physical']['counters'] == [600, 3000]
            self.control.roll.networks['roll'] = actors[role]
            activations.append(role)
            write(folder / role / 'roll-policy.json', dict(role=role, model_sha256=models[role]['model_sha256'],
                control=600, controller_sha256=state['controller_sha256']))
            return state

    # The common snapshot is at roll entry. No standing policy is substituted.
    with patch.multiple(recorder, ROLES=ROLES, LABELS=LABELS, CANDIDATE=None, StandingFork=RollPolicyFork):
        recorder.collect(env, plan, actors, folder, observer, watch, ledger)
    if ledger['branches']:
        assert tuple(activations) == ROLES


def branch_first(plan, folder, record, trajectory):
    start = plan['composition']['roll_start_control']
    entry, = (row for row in record['boundaries'] if row['control'] == start)
    path = folder / entry['file']
    assert sha(path) == entry['sha256']
    with np.load(path, allow_pickle=False) as data:
        first = dict(data)
    rolling = first['mode'] == ROLLING
    np.testing.assert_array_equal(first['mode'], trajectory['mode'][start])
    assert not first['actor_input'][rolling, 34:].any() and rolling.any()
    return dict(matched_snapshot_control=start, worlds_entered=int(rolling.sum()),
        roll_model_sha256=plan['models'][ROLL_MODEL]['model_sha256']), first


def score():
    project, plan = driver.checked('score-pending')
    state = project['goal_execution'][KEY]
    folder = ROOT / plan['record']
    path = folder / 'collection.json'
    ledger = read(path, state['collection_sha256'])
    assert sha(folder / 'protocol.json') == sha(PLAN)
    assert ledger['status'] == 'collected' and ledger['resets'] == 1 and ledger['actor_weights_unchanged']
    start = plan['composition']['roll_start_control']
    assert ledger['prefix']['controls'] == start and tuple(ledger['branches']) == ROLES
    assert ledger['controls'] == start + sum(b['controls'] - start for b in ledger['branches'].values())
    assert ledger['transitions'] == ledger['controls'] * plan['composition']['worlds'] <= plan['maximum_control_transitions']
    assert ledger['physics_substeps'] == ledger['controls'] * 5
    fork = read(folder / 'fork.json', ledger['fork_sha256'])
    admission = read(folder / 'roll-actor-admission.json')
    assert admission['controls'] == admission['physical_substeps'] == 0
    initial = folder / 'initial.npz'
    assert sha(initial) == ledger['initial_sha256']
    with np.load(initial, allow_pickle=False) as data:
        starting = dict(data)
    prefix = read_trace(folder / 'shared', ledger['prefix']['chunks'], 1, start, EVIDENCE_FIELDS, sha)
    common = dict(ledger['prefix'], qpos_joint_addresses=ledger['qpos_joint_addresses'], camera_command=ledger['camera_command'])
    outcomes, population, first = {}, {}, {}
    contract = plan['roll_policy_comparison']
    with patch.object(verifier, 'standing_checks', branch_first):
        for role, model in contract['models'].items():
            expected = dict(role=role, model_sha256=model['model_sha256'], control=start, controller_sha256=fork['controller_sha256'])
            assert read(folder / role / 'roll-policy.json') == expected
            assert admission['models'][role]['model_sha256'] == model['model_sha256']
            plan_with_actor = dict(plan, models={**plan['models'], ROLL_MODEL: model})
            outcomes[role], population[role], first[role] = verifier.verify_branch(
                plan_with_actor, folder / role, ledger['branches'][role], prefix, common, starting, fork)
            outcomes[role]['candidate'] = role == ROLES[-1]
            outcomes[role]['roll_policy_admission'] = outcomes[role].pop('standing_policy')
    reference, control, candidate = ROLES
    for role in ROLES[1:]:
        for field in ('observed', 'previous_target', 'mode', 'actor_input'):
            np.testing.assert_array_equal(first[reference][field], first[role][field], err_msg=field)
    comparisons = {role:verifier.compare(population[candidate], population[role], plan['seven_command']['groups'])
                   for role in ROLES[:-1]}
    complete = outcomes[candidate]['complete']
    passed = bool(outcomes[candidate]['passed_development_gate']
        and comparisons[reference]['roll']['net'] >= contract['minimum_net_gain_over_official']
        and comparisons[control]['roll']['net'] >= contract['minimum_net_gain_over_trained_control']
        and all(row['complete'] is None or complete >= row['complete'] for role,row in outcomes.items() if role != candidate))
    dest = REPORT / 'analysis.json'
    assert not dest.exists()
    write(dest, dict(protocol_sha256=sha(PLAN), collection_sha256=sha(path), seed=plan['environment']['seed'],
        worlds=plan['composition']['worlds'], results=outcomes, comparisons=comparisons, admissions=admission,
        verified_snapshot=fork['physical'], first_observations_and_prior_targets_exact=True,
        first_motor_input_exact=all(np.array_equal(first[reference]['motor'][0], first[r]['motor'][0]) for r in ROLES[1:]),
        candidate_eligible_for_independent_confirmation=passed, transitions=ledger['transitions'],
        hardware_io=False, reserved_seed_used=False, learning_updates=0))
    state.update(status='reviewed', candidate_advanced=passed, result=rel(dest), result_sha256=sha(dest))
    project['goal_execution'][TRAIN_KEY].update(action_comparison=rel(dest), action_comparison_sha256=sha(dest))
    project['staged_training_request'].update(current_protocol_execution='post-getup-action-comparison-reviewed', current_evidence=rel(dest))
    write(PROJECT, project)
    print('ROLL_POST_GETUP_ACTION_COMPARISON_SCORED', sha(dest), {r:x['roll_complete'] for r,x in outcomes.items()}, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'sample', 'score', 'run'))
    action = parser.parse_args().action
    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode(), \
         patch.multiple(driver, KEY=KEY, STAGE=STAGE, PLAN=PLAN, REPORT=REPORT, ROLES=ROLES, collect=collect), \
         patch.object(driver.parent, 'PLAN', PLAN), patch.object(feedback, 'components', driver.isolated_components):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == 'run':
            freeze()
            driver.sample()
            state = read(PROJECT)['goal_execution'][KEY]['status']
            if state == 'score-pending':
                score()
            else:
                print('ROLL_POST_GETUP_ACTION_COMPARISON_STOPPED', state, flush=True)
        else:
            {'freeze': freeze, 'sample': driver.sample, 'score': score}[action]()


if __name__ == '__main__':
    main()
