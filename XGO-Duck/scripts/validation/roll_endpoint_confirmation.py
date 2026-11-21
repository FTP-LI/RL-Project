"""Confirm post-forward roll scoring on a new development and conditional heldout seed."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

import collect_roll_confirmation as official
import roll_common_confirmation as base
from common_scene_contract import compare_arrays, model_arrays
from evaluate_standup_reference import Tee
from recover_roll_import_rng import preserving_actor
from record_io import write
from roll_endpoint_readback_capture import check_geom_layout, collect
from s1_reference_inputs import ROOT, WORK, read, sha
from verify_roll_confirmation import verify as verify_legacy
from verify_roll_endpoint_confirmation import reconstruct


STAGE = 'S4-roll-endpoint-confirmation'
PLAN = ROOT/'configs/experiments/roll_endpoint_confirmation.json'
DRAFT = PLAN.with_name('roll_endpoint_confirmation_draft.json')
OUT = ROOT/'records/reports/roll-endpoint-confirmation'
EARLIER = ROOT/'records/reports/roll-endpoint-readback'
PROJECT = ROOT/'project.json'
ROLES = ('development-b', 'heldout')


def passes(receipt):
    return receipt['fresh_action_gate_passed'] and receipt['constraint_agreement_gate_passed']


def freeze():
    from mjlab.scene import Scene

    assert not PLAN.exists() and not OUT.exists(), 'Preserve earlier attempts'
    task = read(PROJECT)['staged_training_request']
    assert task['current_stage'] == STAGE+'-preparation' and not task['pause_requested']
    draft = read(DRAFT)
    assert draft['status'] == 'draft-not-executable' and draft['stage'] == STAGE+'-preparation'
    parent_path = ROOT/draft['parent_protocol']
    with patch.multiple(base, PLAN=parent_path, OUT=EARLIER):
        parent = base.checked()
    proof_path = ROOT/draft['development_a_evidence']
    previous = read(proof_path)
    coordinator = read(EARLIER/'coordinator.json')
    assert coordinator['status'] == 'completed' and coordinator['analysis_sha256'] == sha(proof_path)
    assert previous['fresh_gate_passed'] and previous['constraint_agreement_gate_passed'] and not previous['heldout_used']
    prior_record = ROOT/previous['record']
    original = read(prior_record/'verification.json')
    assert original['seed'] == draft['development_a_seed'] == 1025
    assert previous['original_verification_sha256'] == sha(prior_record/'verification.json')
    reproduced = reconstruct(parent, original)
    assert passes(reproduced) and reproduced['paired'] == previous['paired']
    assert reproduced['fresh_successes'] == previous['fresh_score']['successes'] == 116
    assert reproduced['fresh_vs_constraints'] == previous['fresh_vs_constraints']

    seeds = draft['development_b_seed'], draft['heldout_seed']
    assert seeds == (1026, 2026) and draft['episodes_per_seed'] == parent['episodes'] == 128
    assert not any(f'_s{seed}_' in folder.name for folder in (ROOT/'records/evaluations').iterdir() for seed in seeds)
    assert draft['minimum_successes_each_seed'] == parent['minimum_successes_each'] == 116
    assert draft['minimum_readback_agreement'] == parent['agreement_minimum'] == .995
    assert draft['maximum_new_control_transitions'] == 2*parent['maximum_control_transitions'] == 115200
    assert draft['maximum_new_batched_substeps'] == 2*parent['maximum_batched_substeps'] == 4500

    sources = dict(parent['source_sha256'])
    for name in ('roll_endpoint_confirmation.py', 'verify_roll_endpoint_confirmation.py'):
        path = ROOT/'scripts/validation'/name
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    references = dict(parent['reference_sha256'])
    scene_path = EARLIER/'cpu-scene-fields.npz'
    for path in (DRAFT, parent_path, proof_path, scene_path, prior_record/'verification.json'):
        references[path.relative_to(ROOT).as_posix()] = sha(path)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    evaluations, configs = {}, {}
    for role, seed in zip(ROLES, seeds, strict=True):
        startup = OUT/f'startup-{role}'
        plan = dict(parent['evaluations']['development-a'], role=role, stage=STAGE, seed=seed,
            heldout_used=role == 'heldout', run_id=f'duck_roll_endpoint-confirmation_{role}_s{seed}_{stamp}',
            startup_archive=(startup/'contact-clock-initial.npz').relative_to(ROOT).as_posix(),
            startup_endpoints=(startup/'endpoint-contact.npz').relative_to(ROOT).as_posix(), source_sha256=sources)
        evaluations[role] = plan
        cfg = official.prepare(plan)
        assert cfg.episode_length_s == draft['episode_seconds'] == 9. and cfg.scene.num_envs == 128 and not cfg.auto_reset
        configs[role] = yaml.load(yaml.dump(asdict(cfg)), Loader=yaml.BaseLoader)
    compiled = Scene(official.prepare(evaluations[ROLES[0]]).scene, device='cpu').compile()
    fields = model_arrays(compiled)
    with np.load(scene_path) as archive:
        compare_arrays(fields, {name: archive[name] for name in archive.files})
    check_geom_layout(compiled)
    assert not torch.cuda.is_initialized()
    spec = dict(parent, status='frozen-before-dynamics', stage=STAGE, evaluations=evaluations,
        maximum_new_control_transitions=draft['maximum_new_control_transitions'],
        maximum_new_batched_substeps=draft['maximum_new_batched_substeps'], development_a=proof_path.relative_to(ROOT).as_posix(),
        development_a_sha256=sha(proof_path), heldout_only_after_both_development_pass=True,
        source_sha256=sources, reference_sha256=references, scope=draft['policy_scope'])
    OUT.mkdir()
    np.savez_compressed(OUT/'cpu-scene-fields.npz', **fields)
    write(PLAN, spec)
    write(OUT/'preflight.json', dict(status='passed-cpu-only', protocol_sha256=sha(PLAN), source_count=len(sources),
        cpu_scene_fields_sha256=sha(OUT/'cpu-scene-fields.npz'), evaluation_configs=configs,
        development_a_recomputed=dict(cached=reproduced['cached_successes'], fresh=reproduced['fresh_successes'],
                                     wins=len(reproduced['paired']['wins']), losses=len(reproduced['paired']['losses'])),
        physical_substeps=0, cuda_initialized=False, heldout_used=False))
    print('ROLL_ENDPOINT_CONFIRMATION_FROZEN', sha(PLAN), seeds, flush=True)


def capture(spec, role, zero):
    plan = spec['evaluations'][role]
    folder = OUT/f'startup-{role}' if zero else ROOT/'records/evaluations'/plan['run_id']
    folder.mkdir(exist_ok=False)
    (folder/'videos').mkdir()
    write(folder/'protocol.json', plan)
    for name in spec['source_sha256']:
        target = folder/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    state = dict(status='running', run_id=plan['run_id'], role=role, seed=plan['seed'], path=str(folder), pid=os.getpid(),
        foreground_tty=os.isatty(1), started_at=datetime.now(timezone.utc).isoformat(), master_sha256=sha(PLAN),
        protocol_sha256=sha(folder/'protocol.json'), model_sha256=plan['model_sha256'], learning_updates=0,
        heldout_used=role == 'heldout' and not zero, hardware_io=False)

    def save():
        write(folder/'evaluation.json', state)

    save()
    if not zero:
        write(ROOT/'records/evaluations/current-evaluation.json', dict(path=str(folder), pid=os.getpid(), run_id=plan['run_id']))
    with (folder/'execution.log').open('x', buffering=1) as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            print('VISIBLE_ROLL_ENDPOINT_CONFIRMATION', role, 'seed', plan['seed'], 'zero' if zero else 'sampling', flush=True)
            with patch.object(official, 'actor_model', preserving_actor):
                collect(plan, folder, state, save, startup=zero)
            state['status'] = 'completed-zero-step' if zero else 'completed'
        except BaseException as error:
            state.update(status='stopped-review-required', error=repr(error))
            raise
        finally:
            state['ended_at'] = datetime.now(timezone.utc).isoformat()
            save()


def run(worker):
    assert os.isatty(1), 'Use the existing visible VS Code shared task terminal'
    project = read(PROJECT)
    task = project['staged_training_request']
    assert task['current_stage'] == STAGE and task['current_protocol_frozen'] and not task['pause_requested']
    assert ROOT/task['current_protocol'] == PLAN
    progress = OUT/'coordinator.json'
    with patch.multiple(base, PLAN=PLAN, OUT=OUT):
        spec = base.checked()
    if worker:
        state = read(progress)
        assert state['status'] == 'running' and state['phase'] == worker and state['protocol_sha256'] == sha(PLAN)
        role = state['role']
        assert role in ROLES
        if role == 'heldout':
            assert passes(read(OUT/'development-b-readback.json'))
        capture(spec, role, zero=worker == 'gpu-zero-step')
        return

    assert not progress.exists(), 'One-time protocol. Preserve ended or interrupted work.'
    state = dict(status='running', phase='preparing', pid=os.getpid(), visible_tty=True, protocol_sha256=sha(PLAN),
                 started_at=datetime.now(timezone.utc).isoformat(), completed_roles=[])
    write(progress, state)
    task.update(simulation_running=True, learning_running=False)
    write(PROJECT, project)

    def child(phase):
        state['phase'] = phase
        write(progress, state)
        subprocess.run([sys.executable, '-B', str(Path(__file__)), '--worker', phase], cwd=WORK, check=True)

    results = {}
    try:
        for role in ROLES:
            assert not read(PROJECT)['staged_training_request']['pause_requested']
            plan = spec['evaluations'][role]
            state.update(role=role, seed=plan['seed'], record=f'records/evaluations/{plan["run_id"]}')
            child('gpu-zero-step')
            start = OUT/f'startup-{role}'
            assert read(start/'contact-clock-zero.json')['physical_substeps'] == 0
            assert read(start/'endpoint-contact.json')['complete']
            assert not read(PROJECT)['staged_training_request']['pause_requested']
            child('sampling')
            state['phase'] = 'independent-verification'
            write(progress, state)
            original = verify_legacy(role, spec, PLAN, OUT)
            result = reconstruct(spec, original)
            write(OUT/f'{role}-readback.json', result)
            results[role] = result
            state['completed_roles'].append(role)
            write(progress, state)
            print('ROLL_ENDPOINT_SEED', role, 'cached', result['cached_successes'], 'fresh', result['fresh_successes'], 'passed', passes(result), flush=True)
            if not passes(result):
                break
        confirmed = len(results) == len(ROLES) and all(passes(result) for result in results.values())
        summary = dict(status='passed-independent-verification', development_a=spec['development_a'],
            development_a_sha256=spec['development_a_sha256'], roles=results,
            nominal_post_forward_gate_passed=confirmed, heldout_used='heldout' in results,
            actual_control_transitions=sum(r['actual_control_transitions'] for r in results.values()),
            actual_batched_substeps=sum(r['physical_substeps'] for r in results.values()), learning_updates=0,
            protocol_sha256=sha(PLAN), decision='nominal-roll-confirmed-transitions-pending' if confirmed else 'closed-gate-not-met')
        assert summary['actual_control_transitions'] <= spec['maximum_new_control_transitions']
        assert summary['actual_batched_substeps'] <= spec['maximum_new_batched_substeps']
        write(OUT/'analysis.json', summary)
        state.update(status='completed', phase='closed', analysis_sha256=sha(OUT/'analysis.json'), decision=summary['decision'])
    except BaseException as error:
        state.update(status='stopped-review-required', error=repr(error))
        raise
    finally:
        state['ended_at'] = datetime.now(timezone.utc).isoformat()
        write(progress, state)
        project = read(PROJECT)
        project['staged_training_request'].update(simulation_running=False, learning_running=False)
        write(PROJECT, project)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--freeze', action='store_true')
    options.add_argument('--worker', choices=('gpu-zero-step', 'sampling'))
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        run(args.worker)
