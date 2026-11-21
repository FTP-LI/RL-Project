"""Evaluate the unchanged nominal route on its reserved independent development seed."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from recover_nominal_skill_composition_zero import isolated_components
import routed_getup_sequence as feedback
import run_nominal_skill_composition as parent


ROOT, PROJECT = parent.ROOT, parent.PROJECT
read, sha, rel, write = parent.read, parent.sha, parent.rel, parent.write
PLAN = parent.PLAN.with_name('nominal_skill_composition_independent.json')
REPORT = parent.REPORT / 'independent-development'
KEY = 'nominal_skill_composition_independent'
STAGE = 'nominal-skill-composition-independent-development'
ENTRY = Path(__file__).resolve()


def freeze():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    assert parent.allowed(project, STAGE + '-preparation') and not task['simulation_running']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists()
    original, pair, measured = (goal[name] for name in ('nominal_skill_composition',
        'nominal_pre_roll_stand_pair', 'nominal_skill_composition_contact_load'))
    archive = {}
    for label, row, key in (('original', original, 'protocol'), ('paired', pair, 'result'),
            ('load_protocol', measured, 'protocol'), ('load_review', measured, 'contact_load_review')):
        file = ROOT / row[key]
        assert sha(file) == row[key + '_sha256']
        archive[label] = file
    contact = read(archive['load_protocol'])['contact_load']
    observed = read(archive['load_review'])
    assert observed['qpos_physics_alignment_exact'] and observed['original_gates_unchanged']
    assert measured['status'] == pair['status'] == 'reviewed' and not pair['candidate_advanced']
    assert any(row['passed_development_gate'] and not row['candidate'] for row in read(archive['paired'])['results'].values())
    seed = contact['independent_development_seed_reserved']
    assert seed == 1257 and seed not in contact['confirmation_seeds_reserved']
    plan = read(archive['original'])
    d, plant, seven = (plan[k] for k in ('composition', 'environment', 'seven_command'))
    d['seed'] = seed
    requests = [d['requested_pose_counts'][pose] for pose in parent.POSES]
    codes = np.random.default_rng(seed).permutation(np.repeat(range(4), requests))
    assert len(codes) == d['worlds'] == plant['num_envs'] == 896
    groups = parent.balanced_groups(codes, seven['commands'], seed + 70000)
    assert set(groups) == set(seven['groups']) and all(len(ids) == 128 for ids in groups.values())
    ordered = [i for ids in groups.values() for i in ids]
    assert sorted(ordered) == list(range(len(codes))) and groups != seven['groups']
    seven['groups'] = groups
    run = 'duck_nominal_skill_composition_independent_s1257_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plant.update(seed=seed, run_id=run, split=dict(pose_codes=codes.tolist(),
        groups={name: np.flatnonzero(codes == i).tolist() for i, name in enumerate(parent.POSES)}))
    for model in plan['models'].values():
        model['seed'] = seed
    plan.update(stage=STAGE, record=f'records/evaluations/{run}', independent_development=dict(
        seed=seed, mechanism_seed=1256, reserved_confirmation_seeds=contact['confirmation_seeds_reserved'],
        immutable_controller='Original standing strategy and official roll',
        preregistered_actions='Original thresholds and full twenty-second command.')),
    plan['references'].update({rel(file): sha(file) for file in archive.values()})
    plan['source_sha256'][rel(ENTRY)] = sha(ENTRY)
    parent.base.shared.assert_sources(plan)
    REPORT.mkdir()
    write(PLAN, plan)
    write(REPORT / 'preflight.json', dict(protocol_sha256=sha(PLAN), observer=parent.preflight(),
        command_worlds={name: len(ids) for name, ids in groups.items()}, maximum_controls=d['maximum_controls'],
        seed=seed, physical_substeps=0, hardware_io=False))
    goal[KEY] = dict(status='zero-ready', protocol=rel(PLAN), protocol_sha256=sha(PLAN),
                     preflight=rel(REPORT / 'preflight.json'), run_id=run, hardware_io=False)
    task.update(current_stage=STAGE, current_protocol=rel(PLAN), current_protocol_frozen=True,
                current_protocol_execution='nominal-skill-composition-independent-frozen', current_evidence=rel(REPORT / 'preflight.json'))
    write(PROJECT, project)
    print('NOMINAL_SKILL_COMPOSITION_INDEPENDENT_FROZEN', sha(PLAN), run, flush=True)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('freeze', 'zero', 'sample', 'score'))
    action = cli.parse_args().action
    if action == 'freeze':
        return freeze()
    with patch.multiple(parent, PLAN=PLAN, REPORT=REPORT, KEY=KEY, STAGE=STAGE), \
         patch.object(feedback, 'components', isolated_components):
        if action == 'score':
            from verify_nominal_skill_composition import score
            return score()
        with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode():
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            {'zero': parent.zero, 'sample': parent.sample}[action]()


if __name__ == '__main__':
    main()
