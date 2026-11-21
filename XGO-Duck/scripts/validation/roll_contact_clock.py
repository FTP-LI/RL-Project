"""Freeze and run a single visible development contact-clock probe."""
import argparse
from copy import deepcopy
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
from recover_roll_import_rng import preserving_actor
from record_io import write
from roll_contact_clock_capture import collect, contact_layout, reduce_contacts
from s1_reference_inputs import ROOT, WORK, read, sha
from verify_roll_confirmation import verify as verify_original
from verify_roll_contact_clock import check_alignment_fixture, verify as verify_clock


STAGE = 'S4-roll-contact-clock'
PLAN = ROOT/'configs/experiments/roll_contact_clock.json'
DRAFT = PLAN.with_name('roll_contact_clock_draft.json')
OUT = ROOT/'records/reports/roll-contact-clock'
PROJECT = ROOT/'project.json'
ROLE = 'development-a'  # Reuse the original single development worker/verifier entry.


def freeze():
    import mujoco
    import mujoco_warp
    import mjlab
    from mjlab.scene import Scene

    assert not PLAN.exists() and not OUT.exists(), 'Preserve any earlier attempt'
    project = read(PROJECT)['staged_training_request']
    assert project['current_stage'] == STAGE+'-preparation' and not project['pause_requested']
    draft = read(DRAFT)
    assert draft['status'] == 'draft-not-executable' and draft['stage'] == STAGE
    previous, review = ROOT/draft['parent'], ROOT/draft['development_evidence']
    parent, evidence = read(previous), read(review)
    assert evidence['matched_pairs'] == 14 and not evidence['heldout_used'] and evidence['original_scores_unchanged']
    for name, digest in evidence['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    seed, n = draft['seed'], draft['episodes']
    assert (seed, n, draft['episode_seconds']) == (1024, 128, 9.)
    assert n*round(9./draft['control_dt']) == draft['maximum_control_transitions']
    assert round(9./draft['physics_dt']) == draft['maximum_batched_substeps']
    assert not any(f'_s{seed}_' in folder.name for folder in (ROOT/'records/evaluations').iterdir())
    for cohort in evidence['cohorts'].values():
        for row in cohort['worlds']:
            start, stop = draft['capture_seconds']
            assert start <= row['events']['departure']['time_s'] < row['events']['first_sole']['time_s'] <= stop

    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plan = dict(deepcopy(parent['evaluations'][ROLE]), stage=STAGE, seed=seed,
                run_id=f'duck_roll_contact-clock_development_s{seed}_{stamp}',
                capture_seconds=draft['capture_seconds'],
                maximum_batched_substeps=draft['maximum_batched_substeps'],
                maximum_sampling_wall_seconds=draft['maximum_sampling_wall_seconds'],
                load_threshold_nominal_weight_fraction=draft['load_threshold_nominal_weight_fraction'],
                startup_archive=(OUT/'startup/contact-clock-initial.npz').relative_to(ROOT).as_posix(),
                scope=draft['scope'])
    cfg = official.prepare(plan)
    assert cfg.episode_length_s == 9. and cfg.scene.num_envs == n and not cfg.auto_reset
    assert str(cfg.sim.mujoco.integrator).lower() == 'implicitfast'
    model = Scene(cfg.scene, device='cpu').compile()
    scene = model_arrays(model)
    reference_scene = ROOT/'records/reports/roll-common-confirmation/rng-recovery/cpu-scene-fields.npz'
    with np.load(reference_scene) as archive:
        compare_arrays(scene, {name: archive[name] for name in archive.files})
    floor, kind, weight = contact_layout(model)
    assert floor >= 0 and set(kind) == {-1, 0, 1, 2, 3}
    fixture, loads = reduce_contacts(np.array([[0, 1], [2, 0], [3, 0], [0, 4], [1, 2], [0, 5]]),
        np.array([0, 0, 0, 1, 1, 1]), np.array([2., 3., 0., 4., 100., 50.]),
        0, np.array([-1, 1, 2, 0, 3, -1]), 2)
    np.testing.assert_array_equal(fixture, [[3, 1, 1, 0], [1, 0, 0, 1]])
    np.testing.assert_array_equal(loads, [[5., 2., 3., 0.], [4., 0., 0., 4.]])
    check_alignment_fixture()
    assert not torch.cuda.is_initialized()

    sources = dict(parent['source_sha256'])
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT/'scripts'):
                sources[path.relative_to(ROOT).as_posix()] = sha(path)
    display = ROOT/'configs/environments/training_display.json'
    sources[display.relative_to(ROOT).as_posix()] = sha(display)
    libraries = dict(parent['library_source_sha256'])
    for root, names in ((Path(mujoco_warp.__file__).parent, ('_src/support.py', '_src/forward.py', '_src/types.py')),
                        (Path(mjlab.__file__).parent, ('envs/manager_based_rl_env.py', 'sim/sim.py'))):
        for name in names:
            path = root/name
            libraries[str(path)] = sha(path)
    references = dict(parent['reference_sha256'])
    references.update({path.relative_to(ROOT).as_posix(): sha(path) for path in (DRAFT, previous, review, reference_scene)})
    for root, names in ((ROOT, sources), (WORK, parent['training_source_sha256']), (Path('/'), libraries), (ROOT, references)):
        for name, digest in names.items():
            assert sha(root/name) == digest, name
    plan.update(source_sha256=sources, library_source_sha256=libraries)
    spec = dict(draft, status='frozen-before-dynamics', evaluations={ROLE: plan}, criteria=plan['criteria'],
                source_sha256=sources, library_source_sha256=libraries,
                training_source_sha256=parent['training_source_sha256'], reference_sha256=references,
                minimum_successes_each=parent['minimum_successes_each'], full_goal_preserved=True,
                source_step_order='Contacts and forces solved from the pre-integration state; later control forward recomputes contacts.')
    assert sha(ROOT/plan['model']) == plan['model_sha256']
    OUT.mkdir()
    np.savez_compressed(OUT/'cpu-scene-fields.npz', **scene)
    write(PLAN, spec)
    write(OUT/'preflight.json', dict(status='passed-cpu-only', protocol_sha256=sha(PLAN), source_count=len(sources),
        cpu_scene_fields_sha256=sha(OUT/'cpu-scene-fields.npz'), nominal_weight_n=weight,
        evaluated_contact_reduction_fixture=True, checked_clock_alignment_fixture=True,
        source_integrator='implicitfast', physical_substeps=0,
        evaluation_configs={ROLE: yaml.load(yaml.dump(asdict(cfg)), Loader=yaml.BaseLoader)},
        old_development_scope_only=True, cuda_initialized=False))
    print('ROLL_CONTACT_CLOCK_FROZEN', sha(PLAN), len(sources), 'weight', weight, flush=True)


def available():
    project = read(PROJECT)['staged_training_request']
    assert project['current_stage'] == STAGE and project['current_protocol_frozen'] and not project['pause_requested']
    assert ROOT/project['current_protocol'] == PLAN


def run(worker):
    assert os.isatty(1), 'Use the existing visible VS Code task terminal'
    available()
    progress = OUT/'coordinator.json'
    with patch.multiple(base, PLAN=PLAN, OUT=OUT, STAGE=STAGE, collect=collect), patch.object(official, 'actor_model', preserving_actor):
        spec = base.checked()
        if worker:
            state = read(progress)
            assert state['status'] == 'running' and state['phase'] == worker
            assert state['protocol_sha256'] == sha(PLAN)
            if worker == 'gpu-zero-step':
                base.run_one(ROLE, startup=True)
            else:
                score = verify_original(ROLE, spec, PLAN, OUT)
                verify_clock(spec, OUT, score)
            return
        assert not progress.exists(), 'The frozen probe is one-time'
        state = dict(status='running', phase='gpu-zero-step', pid=os.getpid(), visible_tty=True,
            record=f'records/evaluations/{spec["evaluations"][ROLE]["run_id"]}', protocol_sha256=sha(PLAN),
            started_at=datetime.now(timezone.utc).isoformat())
        write(progress, state)
        project = read(PROJECT)
        project['staged_training_request'].update(simulation_running=True, learning_running=False)
        write(PROJECT, project)

        def child(phase):
            state['phase'] = phase
            write(progress, state)
            print('VISIBLE_ROLL_CONTACT_CLOCK', phase, flush=True)
            subprocess.run([sys.executable, '-B', str(Path(__file__)), '--worker', phase], cwd=WORK, check=True)

        try:
            child('gpu-zero-step')
            assert read(OUT/'startup/contact-clock-zero.json')['physical_substeps'] == 0
            available()
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--worker', choices=('gpu-zero-step', 'independent-verification'))
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        run(args.worker)
