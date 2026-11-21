"""Freeze and run one same-trajectory endpoint comparison in VS Code."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
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
from diagnose_roll_contact_sensor_cache import check_sensor_cache
from recover_roll_import_rng import preserving_actor
from record_io import write
from roll_endpoint_readback_capture import check_geom_layout, collect, constraint_counts
from s1_reference_inputs import ROOT, WORK, read, sha
from verify_roll_confirmation import verify as verify_original
from verify_roll_endpoint_readback import TRACE_FIELDS, score, scoring_view, verify as verify_readback


STAGE = 'S4-roll-endpoint-readback'
PLAN = ROOT/'configs/experiments/roll_endpoint_readback.json'
DRAFT = PLAN.with_name('roll_endpoint_readback_draft.json')
OUT = ROOT/'records/reports/roll-endpoint-readback'
PARENT_OUT = ROOT/'records/reports/roll-contact-clock'
PROJECT = ROOT/'project.json'
ROLE = 'development-a'


def check_scoring(parent, reference):
    folder = ROOT/reference['record']
    archive = folder/'forward-roll-trace.npz'
    assert sha(archive) == reference['trace_sha256']
    with np.load(archive) as data:
        trace = {name: data[name] for name in TRACE_FIELDS}
    raw = read(folder/'evaluation.json')
    limits = np.asarray(raw['joint_limits'])
    tol = np.deg2rad(parent['evaluations'][ROLE]['actual_joint_limit_tolerance_deg'])
    original = score(trace, parent['criteria'], limits, tol)
    assert original['successes'] == reference['original_score']['successes'] == 109 and original['rows'][0]['success']
    contacts = np.concatenate((trace['ground_contact'][..., None], trace['head_contact'][..., None], trace['foot_contact']), axis=-1)
    contacts[:, 0, 1] = False
    changed = scoring_view(trace, contacts)
    assert trace['head_contact'][:, 0].any() and not changed['head_contact'][:, 0].any()
    assert all(changed[name] is trace[name] for name in TRACE_FIELDS if name not in ('ground_contact', 'head_contact', 'foot_contact'))
    checked = score(changed, parent['criteria'], limits, tol)
    assert not checked['rows'][0]['success'] and checked['rows'][1:] == original['rows'][1:]
    assert checked['successes'] == original['successes']-1
    return dict(original_successes=original['successes'], head_removed_fixture_successes=checked['successes'],
                unaffected_worlds=127, source_sha256=sha(archive), source=archive.relative_to(ROOT).as_posix())


def freeze():
    from mjlab.scene import Scene

    assert not PLAN.exists() and not OUT.exists(), 'Preserve any earlier attempt'
    task = read(PROJECT)['staged_training_request']
    assert task['current_stage'] == STAGE+'-preparation' and not task['pause_requested']
    draft = read(DRAFT)
    assert draft['status'] == 'draft-not-executable' and draft['stage'] == STAGE+'-preparation'
    previous = ROOT/draft['parent_protocol']
    with patch.multiple(base, PLAN=previous, OUT=PARENT_OUT):
        parent = base.checked()
    cache_path, prior_path = ROOT/draft['evidence'], PARENT_OUT/'analysis.json'
    cache, prior = read(cache_path), read(prior_path)
    assert cache['clock_analysis_sha256'] == sha(prior_path)
    assert cache['status'] == 'passed-archived-alignment-and-cpu-fixture' and cache['active_world_boundaries'] == 7552
    assert cache['reader_sha256'] == sha(ROOT/draft['reader'])
    assert read(PARENT_OUT/'coordinator.json')['analysis_sha256'] == sha(prior_path)
    for path, digest in cache['source_sha256'].items():
        assert sha(Path(path)) == digest, path
    seed, n, seconds = (draft[key] for key in ('development_seed', 'episodes', 'episode_seconds'))
    assert (seed, n, seconds) == (1025, 128, 9.)
    assert n*round(seconds/parent['control_dt']) == draft['maximum_control_transitions']
    assert round(seconds/parent['physics_dt']) == draft['maximum_batched_substeps']
    assert not any(f'_s{seed}_' in folder.name for folder in (ROOT/'records/evaluations').iterdir())

    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_plan = dict(parent['evaluations'][ROLE], stage=STAGE, seed=seed,
        run_id=f'duck_roll_endpoint-readback_development_s{seed}_{stamp}', scope=draft['matching_contract'],
        startup_archive=(OUT/'startup/contact-clock-initial.npz').relative_to(ROOT).as_posix(),
        startup_endpoints=(OUT/'startup/endpoint-contact.npz').relative_to(ROOT).as_posix())
    cfg = official.prepare(run_plan)
    assert cfg.episode_length_s == seconds and cfg.scene.num_envs == n and not cfg.auto_reset
    model = Scene(cfg.scene, device='cpu').compile()
    scene = model_arrays(model)
    scene_path = PARENT_OUT/'cpu-scene-fields.npz'
    with np.load(scene_path) as archive:
        compare_arrays(scene, {name: archive[name] for name in archive.files})
    floor, _, slots = check_geom_layout(model)
    assert floor >= 0
    groups = constraint_counts(np.array([[0, 1], [0, 2], [3, 0], [0, 4], [4, 3], [0, 5], [2, 0]]),
        np.array([0, 0, 0, 1, 1, 1, 1]), np.array([1, 2, 3, 1, 3, 1, 0]),
        0, np.array([-1, 1, 2, 0, 3, -1]), 2)
    np.testing.assert_array_equal(groups, [[2, 1, 0, 0], [1, 0, 0, 1]])
    fixture = check_sensor_cache()
    scoring = check_scoring(parent, prior)
    assert not torch.cuda.is_initialized()

    sources = dict(parent['source_sha256'])
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT/'scripts'):
                name, digest = path.relative_to(ROOT).as_posix(), sha(path)
                assert sources.get(name, digest) == digest, name
                sources[name] = digest
    libraries = dict(parent['library_source_sha256'])
    for name, digest in cache['source_sha256'].items():
        assert libraries.get(name, digest) == digest, name
        libraries[name] = digest
    references = dict(parent['reference_sha256'])
    references.update({path.relative_to(ROOT).as_posix(): sha(path) for path in (DRAFT, previous, cache_path, prior_path, scene_path)})
    run_plan.update(source_sha256=sources, library_source_sha256=libraries)
    spec = dict(parent, status='frozen-before-dynamics', stage=STAGE, seed=seed, episodes=n,
        evaluations={ROLE: run_plan}, source_sha256=sources, library_source_sha256=libraries, reference_sha256=references,
        dynamic_execution_authorized=True, agreement_minimum=.995,
        joint_limit_tolerance_deg=run_plan['actual_joint_limit_tolerance_deg'],
        same_physical_trajectory=True, draft=DRAFT.relative_to(ROOT).as_posix(), scope=draft['interpretation'])
    assert spec['minimum_successes_each'] == 116 and sha(ROOT/run_plan['model']) == run_plan['model_sha256']
    OUT.mkdir()
    np.savez_compressed(OUT/'cpu-scene-fields.npz', **scene)
    write(PLAN, spec)
    write(OUT/'preflight.json', dict(status='passed-cpu-only', protocol_sha256=sha(PLAN), source_count=len(sources),
        cpu_scene_fields_sha256=sha(OUT/'cpu-scene-fields.npz'), found_slots=slots, cache_fixture=fixture,
        geom_filter_fixture=True, scoring_fixture=scoring, physical_substeps=0, cuda_initialized=False,
        evaluation_configs={ROLE: yaml.load(yaml.dump(asdict(cfg)), Loader=yaml.BaseLoader)}))
    print('ROLL_ENDPOINT_FROZEN', sha(PLAN), len(sources), 'sources', flush=True)


def run(worker):
    assert os.isatty(1), 'Use the existing visible VS Code shared task terminal'
    project = read(PROJECT)
    task = project['staged_training_request']
    assert task['current_stage'] == STAGE and task['current_protocol_frozen'] and not task['pause_requested']
    assert ROOT/task['current_protocol'] == PLAN
    progress = OUT/'coordinator.json'
    with patch.multiple(base, PLAN=PLAN, OUT=OUT, STAGE=STAGE, collect=collect), patch.object(official, 'actor_model', preserving_actor):
        spec = base.checked()
        if worker:
            state = read(progress)
            assert state['status'] == 'running' and state['phase'] == worker and state['protocol_sha256'] == sha(PLAN)
            if worker == 'gpu-zero-step':
                base.run_one(ROLE, startup=True)
            else:
                verify_readback(spec, OUT, verify_original(ROLE, spec, PLAN, OUT))
            return
        assert not progress.exists(), 'One-time protocol. Preserve an ended or interrupted run.'
        state = dict(status='running', phase='gpu-zero-step', pid=os.getpid(), visible_tty=True,
            record=f'records/evaluations/{spec["evaluations"][ROLE]["run_id"]}', protocol_sha256=sha(PLAN),
            started_at=datetime.now(timezone.utc).isoformat())
        write(progress, state)
        task.update(simulation_running=True, learning_running=False)
        write(PROJECT, project)

        def child(phase):
            state['phase'] = phase
            write(progress, state)
            print('VISIBLE_ROLL_ENDPOINT', phase, flush=True)
            subprocess.run([sys.executable, '-B', str(Path(__file__)), '--worker', phase], cwd=WORK, check=True)

        try:
            child('gpu-zero-step')
            assert read(OUT/'startup/contact-clock-zero.json')['physical_substeps'] == 0
            assert read(OUT/'startup/endpoint-contact.json')['complete']
            assert not read(PROJECT)['staged_training_request']['pause_requested']
            state['phase'] = 'fixed-official-sampling'
            write(progress, state)
            base.run_one(ROLE)
            child('independent-verification')
            result = read(OUT/'analysis.json')
            state.update(status='completed', phase='closed', analysis_sha256=sha(OUT/'analysis.json'), decision=result['decision'])
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
    options.add_argument('--worker', choices=('gpu-zero-step', 'independent-verification'))
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        run(args.worker)
