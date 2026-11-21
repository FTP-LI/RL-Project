"""Capture solved loads during nominal recovery without changing the control route."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import torch

from recover_nominal_skill_composition_zero import isolated_components
import routed_getup_sequence as feedback
import run_nominal_skill_composition as parent
import run_nominal_skill_composition_contact_load as instrument


ROOT, PROJECT = parent.ROOT, parent.PROJECT
read, rel, sha, write = parent.read, parent.rel, parent.sha, parent.write
KEY = 'nominal_supine_early_load'
STAGE = 'nominal-supine-early-load'
REPORT = parent.REPORT/'early-getup-load'
ENTRY = Path(__file__).resolve()
ORIGINAL_CONFIG = parent.config
RENDER_STATUS = parent.render_status


def paths(seed):
    return (f'{KEY}_{seed}', f'{STAGE}-s{seed}', ROOT/f'configs/experiments/{KEY}_verified_s{seed}.json',
            REPORT/'verified-entry'/f'seed-{seed}')


def select(project, seed):
    key, stage, plan, _ = paths(seed)
    proof = project['goal_execution'][key]
    task = project['staged_training_request']
    task.update(current_stage=stage, current_protocol=rel(plan), current_protocol_frozen=True,
                current_protocol_execution=stage+'-frozen', current_evidence=proof['preflight'])
    project.update(current_protocol_execution=task['current_protocol_execution'], current_evidence=task['current_evidence'])


def freeze():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    draft = ROOT/node['design']
    design = read(draft)
    assert parent.allowed(project, design['preparation_stage']) and not project['staged_training_request']['simulation_running']
    assert node['status'] == 'design-registered' and not torch.cuda.is_initialized()
    failed = node['first_entry_stop']
    assert failed['physical_substeps'] == 0 and sha(ROOT/failed['archived_entry']) == failed['entry_sha256']
    for seed, expected in failed['protocol_sha256s'].items():
        old = ROOT/f'configs/experiments/{KEY}_s{seed}.json'
        assert sha(old) == expected and not (ROOT/read(old)['record']).exists()
    previous = project['goal_execution']['nominal_supine_pre_first_touch']
    chronology_path = ROOT/previous['contact_chronology']
    assert sha(chronology_path) == design['release_contact_review_sha256']
    chronology = read(chronology_path)
    assert all(group['already_touched'] == entry['n'] for record in chronology['cohorts']
               for entry in record['groups'].values() for group in entry['controls'])
    seeds = [entry['seed'] for entry in design['sources']]
    assert seeds == [1256, 1257] and not set(seeds).intersection(design['reserved_seeds'])
    worlds, end = design['worlds_per_seed'], design['end_control']
    assert design['maximum_control_transitions'] == len(seeds)*end*worlds
    assert design['maximum_batched_substeps'] == len(seeds)*end*5
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    first, last = design['capture_controls']
    frozen = []
    for source in design['sources']:
        seed = source['seed']
        key, stage, destination, evidence = paths(seed)
        row = project['goal_execution'][source['key']]
        original = ROOT/row['protocol']
        zero = ROOT/row['zero']
        assert sha(original) == source['protocol_sha256'] == row['protocol_sha256']
        assert sha(zero) == source['zero_sha256'] == row['zero_sha256']
        assert sha(zero.with_name('reset.npz')) == read(zero)['reset_sha256']
        assert read(zero)['protocol_sha256'] == sha(original) and not destination.exists() and not evidence.exists()
        plan = read(original)
        env, route = plan['environment'], plan['composition']
        assert env['seed'] == seed and env['num_envs'] == route['worlds'] == worlds
        assert route['release_control'] == design['release_control'] and route['roll_start_control'] == end
        assert route['control_period_s'] == design['controller_period_s'] and route['maximum_controls'] == 2125
        assert plan['roll_route']['standing_dwell_controls'] == 25
        assert plan['roll_route']['roll']['criteria']['tilt_max_deg'] == 15
        run = f'duck-getup-early-load-s{seed}-{stamp}'
        env['run_id'] = run
        environment_horizon = route['maximum_controls']
        route['maximum_controls'] = end
        plan.update(stage=stage, record=f'records/evaluations/{run}', maximum_unique_controls=end,
            maximum_physics_substeps=end*5, maximum_control_transitions=end*worlds, early_load={
                **design, 'environment_horizon_controls':environment_horizon,
                'fit_seed':seeds[0], 'validation_seed':seeds[1]},
            contact_load=dict(capture_first_control=first, capture_last_control=last,
                capture_seconds=[k*design['controller_period_s'] for k in (first,last)],
                physics_period_s=design['physics_period_s'], load_threshold_nominal_weight_fraction=design['any_foot_load_fraction'],
                maximum_batched_substeps=end*5, initial_cuda_proof=rel(zero)))
        archived_entry = ROOT/failed['archived_entry']
        plan['references'].update({rel(path):sha(path) for path in (draft, chronology_path, original, zero,
            zero.with_name('reset.npz'), archived_entry)})
        plan['source_sha256'].update({rel(path):sha(path) for path in (
            ENTRY, Path(instrument.__file__).resolve(), ENTRY.with_name('roll_contact_clock_capture.py'))})
        parent.base.shared.assert_sources(plan)
        frozen.append((key, destination, evidence, run, plan))

    observation = parent.preflight()
    for key, destination, evidence, run, plan in frozen:
        evidence.mkdir(parents=True)
        write(destination, plan)
        preflight = evidence/'preflight.json'
        write(preflight, dict(observer=observation, protocol_sha256=sha(destination), physical_substeps=0,
              maximum_transitions=end*worlds, capture_controls=[first,last]))
        project['goal_execution'][key] = dict(status='zero-ready', protocol=rel(destination), protocol_sha256=sha(destination),
            run_id=run, preflight=rel(preflight), hardware_io=False)
    node.update(status='frozen', design_sha256=sha(draft), seeds=seeds,
                protocol_sha256s={str(seed):project['goal_execution'][paths(seed)[0]]['protocol_sha256'] for seed in seeds})
    select(project, seeds[0])
    write(PROJECT, project)
    print('SUPINE_EARLY_LOAD_FROZEN', node['protocol_sha256s'], flush=True)


def preserved_config(plan):
    # The original episode horizon remains in the environment. Only the
    # collector stops at the end of the original standing-admission window.
    env_plan = dict(plan, composition={**plan['composition'],
                        'maximum_controls':plan['early_load']['environment_horizon_controls']})
    return ORIGINAL_CONFIG(env_plan)


def render_status(observer, camera, folder, run, count, horizon, worlds, progress, mode):
    label = '足底载荷｜跌倒与起身' if count < 550 else '足底载荷｜滚翻前站稳'
    return RENDER_STATUS(observer, camera, folder, run, count, horizon, worlds, progress, label)


def collect(env, plan, actors, folder, observer, watch, ledger):
    instrument.collect(env, plan, actors, folder, observer, watch, ledger)
    if ledger.get('video'):
        display = read(folder/'state.json')
        observer.publish(folder, dict(display, latest_clip=ledger['video']['file']))


def run(seed):
    project = read(PROJECT)
    goal = project['goal_execution']
    run_key, run_stage, plan, report = paths(seed)
    stage = project['staged_training_request']['current_stage']
    assert parent.allowed(project, stage) and not project['staged_training_request']['simulation_running']
    assert goal[KEY]['status'] == 'frozen' and goal[run_key]['status'] == 'zero-ready'
    first, other = goal[KEY]['seeds']
    if seed == other:
        assert stage == paths(first)[1] and goal[paths(first)[0]]['status'] == 'score-pending'
        select(project, seed)
        write(PROJECT, project)
    else:
        assert seed == first and stage == run_stage
    with (
        patch.multiple(parent, KEY=run_key, STAGE=run_stage, PLAN=plan, REPORT=report,
                       config=preserved_config, qualify=instrument.qualify, collect=collect, render_status=render_status),
        patch.object(feedback, 'components', isolated_components),
    ):
        parent.zero()
        parent.sample()
    completed = read(PROJECT)
    summary = completed['goal_execution'][run_key]
    assert summary['status'] == 'score-pending' and read(ROOT/summary['collection'])['status'] == 'collected'
    task = completed['staged_training_request']
    task.update(current_protocol_execution=run_stage+'-collected', current_evidence=summary['collection'])
    completed.update(current_protocol_execution=task['current_protocol_execution'], current_evidence=task['current_evidence'])
    if seed == other:
        completed['goal_execution'][KEY]['status'] = 'score-pending'
    write(PROJECT, completed)
    print('SUPINE_EARLY_LOAD_COLLECTED', seed, summary['collection_sha256'], flush=True)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('freeze','1256','1257'))
    action = cli.parse_args().action
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze() if action == 'freeze' else run(int(action))


if __name__ == '__main__':
    main()
