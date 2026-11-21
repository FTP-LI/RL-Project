"""Freeze, run and independently verify the bounded neck-support screen."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch

from s1_reference_inputs import ROOT, WORK, read, sha
import collect_roll_confirmation as original
from common_scene_contract import compare_arrays, install_guard, model_arrays
from display_preferences import prepare_display
from record_io import write
from recover_roll_import_rng import preserving_actor
from roll_metrics import summarize
from roll_neck_support import NeckSupport
from roll_neck_support_collect import collect, joint_contract, probe_motor


STAGE = 'S4-neck-support-pair'
DRAFT = ROOT / 'configs/experiments/roll_neck_support_draft.json'
PLAN = ROOT / 'configs/experiments/roll_neck_support_trial.json'
OUT = ROOT / 'records/reports/roll-neck-support'
PARENT = ROOT / 'configs/experiments/roll_common_rng_recovery.json'
ROLL_PROOF = ROOT / 'records/reports/roll-common-confirmation/rng-recovery'
STARTUP = ROOT / 'records/reports/common-plant-screen/startup'
PROJECT = ROOT / 'project.json'


def make_config(p):
    from mjlab.scripts.train import TrainConfig

    env = original.prepare(p)
    train, display, _ = prepare_display(replace(TrainConfig.from_task(p['task']), env=env), sparse_snapshots=True)
    assert train.env is env and not train.video and not display['native_live_window']
    assert env.scene.num_envs == 128 and env.episode_length_s == 9. and not env.auto_reset
    return env, display


def check_sources(spec):
    for base, mapping in ((ROOT, spec['source_sha256']), (ROOT, spec['reference_sha256']),
                          (WORK, spec['training_source_sha256']), (Path('/'), spec['library_source_sha256'])):
        for name, expected in mapping.items():
            assert sha(base / name) == expected, name
    p = spec['environment']
    assert sha(ROOT / p['model']) == p['model_sha256']


def check_saved_cue(draft, old):
    from verify_roll_reference import independent

    replay = {}
    for role in ('development-a', 'development-b'):
        proof = read(ROLL_PROOF / f'{role}-analysis.json')
        assert not proof['heldout_used']
        folder = ROOT / proof['evaluator']
        path = folder / 'forward-roll-trace.npz'
        assert sha(path) == proof['trace_sha256']
        report, plan = read(folder / 'evaluation.json'), read(folder / 'protocol.json')
        neck = report['joint_names'].index('neck_pitch')
        limits = np.asarray(report['joint_limits'])
        names = report['joint_names']
        from mjlab.scene import Scene

        scene = Scene(original.prepare(plan).scene, device='cpu').compile()
        dof = int(scene.joint('robot/' + names[neck]).dofadr.item())
        with np.load(path) as saved:
            trace = {name: saved[name] for name in saved.files}
        first = summarize(trace, plan['criteria'])
        second = independent(trace, plan['criteria'], limits, np.deg2rad(plan['actual_joint_limit_tolerance_deg']))
        assert [row['success'] for row in first['episodes']] == [row['success'] for row in second]
        assert first['successes'] == proof['successes']
        policy = NeckSupport(len(second), draft['intervention'], neck, limits[neck])
        cues = []
        for k, request in enumerate(trace['applied_targets'][1:]):
            result, cue, _ = policy.choose(trace['quaternion'][k], trace['foot_contact'][k], trace['head_contact'][k],
                trace['angular_velocity'][k, :, 1], trace['qvel'][k, :, dof], trace['active'][k], request, True)
            delta = result - request
            other = np.arange(request.shape[-1]) != neck
            assert not delta[:, other].any() and not delta[~cue].any()
            assert (delta[:, neck] >= 0).all() and delta[:, neck].max() <= .080001
            cues.append(cue)
        assert int(np.sum(cues)) == old['cohorts'][role]['affected_samples']
        assert int(np.any(cues, axis=0).sum()) == old['cohorts'][role]['affected_worlds']
        replay[role] = dict(episodes=len(second), original_successes=first['successes'], trigger_samples=int(np.sum(cues)))
    return replay


def freeze():
    from mjlab.envs import ManagerBasedRlEnv
    import verify_roll_neck_support_trial

    assert not PLAN.exists() and not OUT.exists()
    draft, parent = read(DRAFT), read(PARENT)
    offline_path = ROLL_PROOF / 'neck-support-timing-preflight.json'
    offline = read(offline_path)
    assert sha(DRAFT) == offline['draft_sha256'] and offline['offline_gate_passed'] and not offline['heldout_used']
    assert sha(ROOT / draft['development_evidence']) == offline['motor_envelope_sha256']
    assert sha(ROOT / 'scripts/validation/check_roll_neck_support_timing.py') == offline['script_sha256']
    p = deepcopy(parent['evaluations']['development-a'])
    seed, warmup = draft['first_development_seed'], 10
    roles = draft['proposed_arms']
    assert seed not in {old['seed'] for old in parent['evaluations'].values()}
    assert not any(f'_s{seed}_' in path.name for path in (ROOT / 'records/evaluations').iterdir())
    assert draft['episodes_each'] == p['episodes'] and p['criteria']['max_seconds'] == draft['episode_seconds']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    p.update(stage=STAGE, role='paired', seed=seed, run_id=f'duck_roll_neck-support_paired_s{seed}_{stamp}')
    controls = warmup + len(roles) * (450-warmup)
    sources = dict(parent['source_sha256'])
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT / 'scripts'):
                sources[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ('configs/environments/action_recording.json', 'configs/environments/training_display.json'):
        sources[name] = sha(ROOT / name)
    references = dict(parent['reference_sha256'])
    for path in (PARENT, DRAFT, offline_path, ROOT / draft['development_evidence'],
                 STARTUP / 'roll/scene-static-fields.npz', STARTUP / 'roll/nominal-primary-fields.npz',
                 ROLL_PROOF / 'development-a-analysis.json', ROLL_PROOF / 'development-b-analysis.json',
                 ROLL_PROOF / 'neck-support-cpu-first-stop.json'):
        references[path.relative_to(ROOT).as_posix()] = sha(path)
    spec = dict(status='frozen-before-physics', stage=STAGE, environment=p, roles=roles, warmup_controls=warmup,
        intervention=draft['intervention'], gate=draft['proposed_dynamic_gate'], source_sha256=sources,
        reference_sha256=references, training_source_sha256=parent['training_source_sha256'],
        library_source_sha256=parent['library_source_sha256'], maximum_batched_controls=controls,
        maximum_control_transitions=p['episodes']*controls, maximum_physics_substeps=5*controls,
        maximum_sampling_wall_seconds=600, maximum_warp_snapshot_bytes=256*1024**2, initial_resets=1,
        learning_updates=0, hardware_io=False, heldout_used=False, automatic_retry=False, automatic_extension=False,
        stop_rule='Stop on integrity, early termination, pause or budget. Preserve partial physical states and original clips.')
    assert spec['maximum_control_transitions'] <= draft['proposed_maximum_control_transitions']
    check_sources(spec)
    replay = check_saved_cue(draft, offline)
    cfg, display = make_config(p)
    torch.set_num_threads(4)
    env = ManagerBasedRlEnv(cfg=cfg, device='cpu', render_mode=None)
    try:
        env.reset(seed=seed)
        with np.load(STARTUP / 'roll/scene-static-fields.npz') as stored:
            compare_arrays(model_arrays(env.sim.mj_model), {name: stored[name] for name in stored.files})
        names = list(env.action_manager.get_term('joint_pos')._target_names)
        _, limits, _ = joint_contract(env, names)
        motor = probe_motor(env, names, limits, spec['maximum_warp_snapshot_bytes'])
        assert not torch.cuda.is_initialized()
    finally:
        env.close()
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT / 'preflight.json', dict(status='passed-cpu-zero-step', protocol_sha256=sha(PLAN), replay=replay,
        motor=motor, display=display, source_count=len(sources), static_scene_exact=True, cuda_initialized=False))
    print('ROLL_NECK_CPU_ZERO_OK', sha(PLAN), 'pinned sources', len(sources), 'triggers', replay, 'motor', motor, flush=True)


def available():
    stage = read(PROJECT)['staged_training_request']
    assert stage['current_stage'] == STAGE and stage['current_protocol_frozen'] and not stage['pause_requested']
    assert ROOT / stage['current_protocol'] == PLAN


def running(value):
    project = read(PROJECT)
    project['staged_training_request'].update(simulation_running=value, learning_running=False)
    write(PROJECT, project)


def run():
    from evaluate_standup_reference import Tee
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    assert os.isatty(1), 'Use the original shared visible VS Code terminal'
    available()
    spec, pre = read(PLAN), read(OUT / 'preflight.json')
    assert pre['status'] == 'passed-cpu-zero-step' and pre['protocol_sha256'] == sha(PLAN)
    assert not (OUT / 'coordinator.json').exists(), 'No automatic retry or extension'
    check_sources(spec)
    p = spec['environment']
    record = ROOT / 'records/evaluations' / p['run_id']
    record.mkdir()
    shutil.copy2(PLAN, record / 'protocol.json')
    state = dict(status='running', branch='gpu-zero-step', pid=os.getpid(), run_id=p['run_id'],
        record=record.relative_to(ROOT).as_posix(), protocol_sha256=sha(PLAN), conditions=[],
        started_at=datetime.now(timezone.utc).isoformat(), actual=dict(controls=0, substeps=0, resets=0),
        learning_updates=0, heldout_used=False, hardware_io=False)

    def publish():
        write(record / 'evaluation.json', state)
        write(OUT / 'coordinator.json', state)

    publish()
    write(record / 'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), pid=os.getpid(), visible_tty=True))
    write(ROOT / 'records/evaluations/current-evaluation.json', dict(path=str(record), run_id=p['run_id'], pid=os.getpid()))
    running(True)
    try:
        with (record / 'execution.log').open('x', buffering=1) as log, +                redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            configure_torch_backends(allow_tf32=False)
            torch.set_num_threads(4)
            cfg, display = make_config(p)
            write(record / 'display.json', display)
            dump_yaml(record / 'env.yaml', asdict(cfg))
            env_class = install_guard(p, record, STARTUP, 'rollout')
            env = env_class(cfg=cfg, device='cuda:0', render_mode=None)
            try:
                env.seed(p['seed'])
                obs, _ = env.reset()
                state['actual']['resets'] = 1
                old_import = original.actor_model
                original.actor_model = preserving_actor
                try:
                    actor, weights, home, sim_home, names = original.parity(env, obs, p, record)
                finally:
                    original.actor_model = old_import
                _, limits, _ = joint_contract(env, names)
                motor = probe_motor(env, names, limits, spec['maximum_warp_snapshot_bytes'])
                write(record / 'zero-motor-readback.json', motor)
                print('ROLL_NECK_GPU_ZERO_OK', motor, flush=True)
                collect(env, actor, weights, home, sim_home, names, spec, record, state, publish, available)
            finally:
                env.close()
        state.update(status='collected-review-required', branch='independent-verification')
        publish()
        subprocess.run([sys.executable, '-B', str(Path(__file__)), '--verify'], cwd=WORK, check=True)
        state = read(OUT / 'coordinator.json')
    except BaseException as error:
        state.update(status='stopped-review-required', error=repr(error))
        with (record / 'execution.log').open('a') as log:
            traceback.print_exc(file=log)
        raise
    finally:
        state['ended_at'] = datetime.now(timezone.utc).isoformat()
        publish()
        running(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.verify:
        from verify_roll_neck_support_trial import verify

        verify()
    else:
        run()
