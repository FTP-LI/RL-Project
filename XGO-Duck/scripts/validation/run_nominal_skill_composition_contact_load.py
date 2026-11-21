"""Record four-millisecond solver loads on the original nominal route."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from functools import partial
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from recover_nominal_skill_composition_zero import isolated_components
from roll_contact_clock_capture import ContactClock
import routed_getup_sequence as feedback
import run_nominal_skill_composition as parent


ROOT, PROJECT = parent.ROOT, parent.PROJECT
read, sha, rel, write = parent.read, parent.sha, parent.rel, parent.write
PLAN = parent.PLAN.with_name('nominal_skill_composition_contact_load.json')
REPORT = parent.REPORT / 'contact-load'
KEY = 'nominal_skill_composition_contact_load'
STAGE = 'nominal-skill-composition-contact-load'
ENTRY = Path(__file__).resolve()
QUALIFY, COLLECT = parent.qualify, parent.collect


def freeze():
    project = read(PROJECT)
    task, goal = project['staged_training_request'], project['goal_execution']
    assert parent.allowed(project, STAGE + '-preparation') and not task['simulation_running']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists()
    paired, original = goal['nominal_pre_roll_stand_pair'], goal['nominal_skill_composition']
    assert paired['status'] == 'reviewed' and not paired['candidate_advanced']
    source = {}
    for label, row, name in (
        ('original_protocol', original, 'protocol'), ('initial_cuda_proof', original, 'zero'),
        ('paired_score', paired, 'result'), ('paired_handoff', paired, 'handoff_review'),
        ('paired_support_window', paired, 'supported_rotation_review'),
    ):
        path = ROOT / row[name]
        assert sha(path) == row[name + '_sha256']
        source[label] = path
    plan = read(source['original_protocol'])
    support = read(source['paired_support_window'])
    comparison = read(source['paired_score'])
    assert comparison['seed'] == plan['environment']['seed'] == 1256 and not comparison['reserved_seed_used']
    assert support['additional_physical_steps'] == 0 and sum(k['case_count'] for k in support['arms'].values()) == 54
    initial = source['initial_cuda_proof']
    assert read(initial)['protocol_sha256'] == sha(source['original_protocol'])
    assert sha(initial.with_name('reset.npz')) == read(initial)['reset_sha256']
    assert not torch.cuda.is_initialized()
    d, seven = plan['composition'], plan['seven_command']
    assert d['worlds'] == 896 and d['maximum_controls'] == 2125 and d['control_period_s'] == .02
    assert d['roll_start_control'] == 600 and d['roll_deadline_control'] == 1050
    assert plan['roll_route']['minimum_roll_completion'] == 812
    assert seven['minimum_overall'] == 852 and set(seven['minimum_by_command'].values()) == {122}
    run = 'duck_nominal_roll_contact_load_s1256_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plan['environment']['run_id'] = run
    start = d['roll_start_control']
    capture_first, capture_last = start + 20, start + 55
    plan.update(stage=STAGE, record=f'records/evaluations/{run}', contact_load=dict(
        selection='Original standing and official roll. Diagnostic factors are only readbacks.',
        capture_first_control=capture_first, capture_last_control=capture_last,
        capture_seconds=[round(i * .02, 2) for i in (capture_first, capture_last)],
        score_gap_first_endpoint_control=start + 27, score_gap_last_endpoint_control=start + 46,
        physics_period_s=.004, load_threshold_nominal_weight_fraction=.02,
        maximum_batched_substeps=5 * d['maximum_controls'], minimum_lift_support_comparison='strictly greater',
        independent_development_seed_reserved=1257, confirmation_seeds_reserved=[2256, 2246],
        initial_cuda_proof=rel(initial), original_action_gates_unchanged=True))
    references = (*source.values(), initial.with_name('reset.npz'))
    plan['references'].update({rel(file): sha(file) for file in references})
    files = (ENTRY, ENTRY.with_name('roll_contact_clock_capture.py'), ENTRY.with_name('review_nominal_skill_composition_contact_load.py'))
    plan['source_sha256'].update({rel(file): sha(file) for file in files})
    parent.base.shared.assert_sources(plan)
    proof = dict(observer=parent.preflight(), diagnostic_source_sha256={str(name): sha(path) for name, path in source.items()},
                 maximum_control_transitions=d['maximum_controls'] * d['worlds'], physical_substeps=0)
    REPORT.mkdir()
    write(PLAN, plan)
    write(REPORT / 'preflight.json', dict(proof, protocol_sha256=sha(PLAN)))
    goal[KEY] = dict(status='zero-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN),
                     preflight=rel(REPORT / 'preflight.json'), run_id=run, hardware_io=False)
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
                current_protocol_execution='nominal-roll-contact-load-frozen', current_evidence=rel(REPORT / 'preflight.json'))
    write(PROJECT, project)
    print('NOMINAL_ROLL_CONTACT_LOAD_FROZEN', sha(PLAN), run, flush=True)


def qualify(env, obs, plan, actors, folder):
    receipt = QUALIFY(env, obs, plan, actors, folder)
    initial = ROOT / plan['contact_load']['initial_cuda_proof']
    archived = read(initial)
    assert receipt['composition_boundary'] == archived['composition_boundary']
    with np.load(initial.with_name('reset.npz'), allow_pickle=False) as previous, \
         np.load(folder / 'reset.npz', allow_pickle=False) as actual:
        assert previous.files == actual.files and sha(initial.with_name('reset.npz')) == archived['reset_sha256']
        for field in previous.files:
            np.testing.assert_array_equal(actual[field], previous[field], err_msg=field)
    clock = ContactClock(env, plan['contact_load'])
    d, settings = plan['composition'], plan['contact_load']
    assert clock.first == 5 * settings['capture_first_control'] and clock.last == 5 * settings['capture_last_control']
    assert env.physics_dt == settings['physics_period_s'] and clock.maximum == 5 * d['maximum_controls']
    qpos = parent.cpu(env.sim.data.qpos)
    points, loads = clock.read_contacts()
    assert points.shape == loads.shape == (env.num_envs, 4) and np.isfinite(loads).all() and (loads >= 0.).all()
    np.testing.assert_array_equal(qpos, parent.cpu(env.sim.data.qpos))
    assert env.common_step_counter == env._sim_step_counter == clock.completed == 0
    write(folder / 'load-instrument-zero.json', dict(archived_reset_matches=True, first_recorded_substep=clock.first + 1,
        last_recorded_substep=clock.last, contact_slots=clock.slots, channels=loads.shape[1],
        nominal_weight_n=clock.weight, threshold_n=clock.threshold, controls=0, physical_substeps=0))
    return receipt


def collect(env, plan, actors, folder, observer, watch, ledger):
    clock = ContactClock(env, plan['contact_load'])
    physical_step = env.sim.step
    try:
        # The original collector advances physics. Capture solved loads before
        # the separate control-boundary forward refreshes the contact buffer.
        with patch.object(env.sim, 'step', partial(clock.record, physical_step)), \
             patch.object(env, '_reset_idx', side_effect=AssertionError('Unexpected episode reset')):
            COLLECT(env, plan, actors, folder, observer, watch, ledger)
    finally:
        clock.save(folder)
        metadata = folder / 'contact-clock.json'
        if metadata.exists():
            ledger['contact_load'] = dict(meta=rel(metadata), meta_sha256=sha(metadata),
                substeps=clock.completed, measured_substeps=len(clock.rows))
    assert ledger['physics_substeps'] == clock.completed == env._sim_step_counter
    if ledger['controls'] >= plan['contact_load']['capture_last_control']:
        assert len(clock.rows) == clock.last - clock.first and read(metadata)['complete']


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('freeze', 'zero', 'sample', 'score', 'review', 'keyframes'))
    action = cli.parse_args().action
    if action == 'freeze':
        return freeze()
    with patch.multiple(parent, PLAN=PLAN, REPORT=REPORT, KEY=KEY, STAGE=STAGE, qualify=qualify, collect=collect), \
         patch.object(feedback, 'components', isolated_components):
        if action == 'score':
            from verify_nominal_skill_composition import score
            return score()
        if action == 'review':
            from review_nominal_skill_composition_contact_load import main as review
            return review()
        if action == 'keyframes':
            import render_nominal_skill_composition_keyframes as photo
            original_read = photo.read

            def load(path):
                data = original_read(path)
                if path == PROJECT:
                    data['goal_execution'] = {'nominal_skill_composition': data['goal_execution'][KEY]}
                return data

            with patch.object(photo, 'REPORT', REPORT), patch.object(photo, 'read', load), \
                 patch('sys.argv', [str(ENTRY)]):
                return photo.main()
        with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode():
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            {'zero': parent.zero, 'sample': parent.sample}[action]()


if __name__ == '__main__':
    main()
