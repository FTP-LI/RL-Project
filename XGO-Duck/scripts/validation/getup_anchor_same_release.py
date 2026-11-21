"""Freeze and visibly compare the trained getup anchor at one exact release."""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch

import getup_shared_release as shared
from collect_getup_anchor_same_release import collect
from common_scene_contract import compare_arrays, model_arrays
from evaluate_standup_reference import Tee
from getup_release_scene_state import SceneReleaseSnapshot
from getup_shared_release_terrain import cpu_scene_readback
from record_io import write
from s1_reference_inputs import ROOT, WORK, read, sha


STAGE = 'S3-success-anchor-same-release'
DRAFT = ROOT / 'configs/experiments/getup_anchor_same_release_draft.json'
PLAN = ROOT / 'configs/experiments/getup_anchor_same_release.json'
OUT = ROOT / 'records/reports/getup-anchor-same-release'
ANCHOR = ROOT / 'records/reports/getup-teacher-anchor-pair'
PARENT = ROOT / 'configs/experiments/getup_contact_shift.json'
BASELINE = ROOT / 'configs/experiments/getup_shared_release_terrain.json'
PROJECT = ROOT / 'project.json'
MODEL_META = ('joint_names', 'default_joint_pos', 'action_scale', 'observation_names', 'command_names')


@contextmanager
def visible_log(stream):
    with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
        yield


def check_sources(spec):
    shared.assert_sources(spec)
    for plan in spec['policies'].values():
        assert sha(ROOT / plan['model']) == plan['model_sha256']


def model_metadata(p, candidate):
    import onnx

    recorded = {}
    for role, path in (('official', p['model']), ('anchor', candidate['model'])):
        model = onnx.load(ROOT/path)
        metadata = {item.key: item.value for item in model.metadata_props}
        recorded[role] = {name: metadata[name] for name in MODEL_META}
    assert recorded['official'] == recorded['anchor'] and float(recorded['official']['action_scale']) == 1.
    np.testing.assert_array_equal(np.asarray(recorded['official']['default_joint_pos'].split(','), np.float32),
                                  np.asarray(p['model_home'], np.float32))
    return recorded['official']


def check_historical(addresses):
    from getup_coverage_metrics import evaluate
    from recovery_end_hold import evaluate_end_hold
    from verify_getup_anchor_same_release import check_trace, independent_holds, load_arrays

    results = {}
    for role in ('original', 'candidate'):
        proof = read(ANCHOR / f'{role}-analysis.json')
        record = ROOT / proof['record']
        path = record / 'coverage-trace.npz'
        assert sha(path) == proof['trace_sha256']
        p = read(record / 'protocol.json')
        t = load_arrays(path)
        t['actor_input'] = t['observations'].copy()
        t['actor_input'][..., 48:] = 0
        step = check_trace(t, p, addresses)
        score = evaluate(t, p)
        end = evaluate_end_hold(t, p, score['episodes'])
        timing, final = independent_holds(t, p['criteria'])
        np.testing.assert_array_equal(timing, [x['hold_start_seconds'] is not None for x in score['episodes']])
        np.testing.assert_array_equal(final, end['stable_by_environment'])
        for group in ('requested', 'actual', 'overall'):
            assert score[group] == proof[group] and end[group] == proof['end_hold'][group]
        results[role] = dict(timely_holds=score['overall']['timely_holds'],
            end_holds=end['overall']['timely_holds'], maximum_late_step_rad=step)
    return results


def freeze():
    from mjlab.scene import Scene
    import verify_getup_anchor_same_release

    assert not PLAN.exists() and not OUT.exists(), 'Preserve previous attempts'
    draft = read(DRAFT)
    old = ROOT / draft['official_reference_protocol']
    match_path = ROOT / draft['release_match_evidence']
    match = read(match_path)
    assert match['status'] == 'passed-saved-data-only' and not match['heldout_used']
    supine = match['comparisons']['original']['close_by_actual_pose']['supine']
    assert supine['worlds'] == 166 and supine['first_successes'] == 150 and supine['second_successes'] == 135
    historical, parent, baseline = read(old), read(PARENT), read(BASELINE)
    shared.assert_sources(parent)
    shared.assert_sources(baseline)
    p = deepcopy(parent['environment'])
    for key in ('criteria', 'model', 'model_sha256', 'model_home', 'simulation_home', 'joint_limits',
                'standing_model', 'standing_model_sha256', 'handoff_settings', 'neck_home_enabled', 'handoff_enabled'):
        assert p[key] == historical[key], key
    assert not p['neck_home_enabled'] and not p['handoff_enabled']
    assert p['actual_release_minimum'] == draft['minimum_actual_release_count_each'] == 32
    assert p['success_rate_min'] == draft['minimum_timely_and_end_hold_fraction_each'] == .95
    seed, n = draft['development_seed'], draft['worlds']
    for folder in ('records/evaluations', 'records/experiments'):
        assert not any(f'_s{seed}_' in path.name for path in (ROOT/folder).iterdir())
    requested = [draft['requested_groups'][pose] for pose in shared.POSES]
    requested[1] -= 1
    codes = np.concatenate(([1], np.random.default_rng(seed).permutation(np.repeat(np.arange(4), requested))))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(shared.POSES)}
    assert len(codes) == n
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    p.update(seed=seed, num_envs=n, stage=STAGE, split=dict(pose_codes=codes.tolist(), groups=groups),
             run_id=f'duck_getup_anchor-same-release_paired_s{seed}_{stamp}', maximum_control_transitions=n*550)
    candidate = dict(p, model=draft['candidate_model'], model_sha256=draft['candidate_model_sha256'])
    training_path = ANCHOR / 'candidate/training-verification.json'
    training = read(training_path)
    assert training['status'] == 'passed' and training['model_sha256'] == sha(ROOT/candidate['model'])
    assert training['model_sha256'] == candidate['model_sha256'] and training['rollout_updates'] == 41
    roles = draft['arms']
    assert roles == ['reference-a', 'reference-b', 'success-anchor']
    policies = {role: {key: q[key] for key in ('model', 'model_sha256', 'seed')}
                for role, q in zip(roles, (p, p, candidate), strict=True)}
    interface = model_metadata(p, candidate)
    cameras = {role: {'supine': groups['supine'][0]} for role in roles}
    cameras[roles[-1]] = {pose: groups[pose][0] for pose in shared.POSES}
    release = round(draft['shared_passive_seconds'] / draft['control_dt'])
    controls = release + len(roles)*(p['maximum_control_steps']-release)
    assert controls == draft['maximum_batched_controls'] and n*controls == draft['maximum_control_transitions']
    sources = dict(parent['source_sha256'])
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT/'scripts'):
                sources[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ('configs/environments/action_recording.json', 'configs/environments/training_display.json'):
        sources[name] = sha(ROOT/name)
    expected_scene = ANCHOR / 'cpu-scene-fields.npz'
    reference_files = (DRAFT, PARENT, BASELINE, match_path, old, training_path, expected_scene,
                       ANCHOR/'original-analysis.json', ANCHOR/'candidate-analysis.json')
    spec = dict(status='frozen-before-physical-steps', stage=STAGE, environment=p, branches=roles,
        policies=policies, model_interface=interface, cameras=cameras, prefix_controls=release,
        source_sha256=sources, references={file.relative_to(ROOT).as_posix(): sha(file) for file in reference_files},
        maximum_unique_controls=controls, maximum_control_transitions=n*controls, maximum_physics_substeps=5*controls,
        maximum_warp_snapshot_bytes=256*1024**2, maximum_sampling_wall_seconds=720,
        heldout_used=False, automatic_retry=False, automatic_extension=False, hardware_io=False, learning_updates=0)
    check_sources(spec)
    cfg, display, meta = shared.prepare(p)
    assert cfg.env.scene.num_envs == n and cfg.env.decimation == 5 and cfg.env.episode_length_s == 11.
    assert not cfg.env.auto_reset and not display['native_live_window']
    model = Scene(cfg.env.scene, device='cpu').compile()
    with np.load(expected_scene) as f:
        compare_arrays(model_arrays(model), {key: f[key] for key in f.files})
    addresses = model.jnt_qposadr[[model.joint('robot/'+name).id for name in meta['joint_names'].split(',')]].tolist()
    fixture = check_historical(addresses)
    parity = shared.cpu_check({'roles': policies})
    snapshot = cpu_scene_readback(p, spec)
    log = io.StringIO()
    with visible_log(log):
        print('STDOUT_OK')
        print('STDERR_OK', file=sys.stderr)
    assert log.getvalue().splitlines() == ['STDOUT_OK', 'STDERR_OK'] and not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT/'preflight.json', dict(status='passed-cpu-zero-step', protocol_sha256=sha(PLAN),
        physical_substeps=0, scene_exact=True, reset_snapshot=snapshot, actor_import=parity,
        historical_replay=fixture, cpu_joint_addresses=addresses, output_streams_verified=True,
        cuda_initialized=False, source_count=len(sources)))
    print('ANCHOR_SAME_RELEASE_CPU_ZERO_OK', sha(PLAN), 'seed', seed, 'fields', snapshot['fields'], flush=True)


def available():
    state = read(PROJECT)['staged_training_request']
    assert state['current_stage'] == STAGE and state['current_protocol_frozen'] and not state['pause_requested']
    assert ROOT/state['current_protocol'] == PLAN


def gpu_startup(env, obs, memory, spec, record):
    import onnxruntime as ort

    p = spec['environment']
    rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(env.device).clone())
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        actors = {role: shared.actor_model(spec['policies'][role], env.device)
                  for role in (spec['branches'][0], spec['branches'][-1])}
        actors[spec['branches'][1]] = actors[spec['branches'][0]]
        standing = shared.actor_model(dict(p, model=p['standing_model'], model_sha256=p['standing_model_sha256']), env.device)
        shared.startup(env, obs, dict(getup=actors[spec['branches'][0]], standing=standing), p, record)
        inputs = shared.cpu(obs['actor'][::5])
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        candidate = spec['branches'][-1]
        session = ort.InferenceSession(str(ROOT/spec['policies'][candidate]['model']),
                                       sess_options=session_options, providers=['CPUExecutionProvider'])
        expected = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in inputs])
        actual = shared.cpu(actors[candidate](shared.td('actor', torch.from_numpy(inputs).to(env.device))))
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert torch.equal(rng[0], torch.get_rng_state()) and torch.equal(rng[1], torch.cuda.get_rng_state(env.device))
    with np.load(ANCHOR/'cpu-scene-fields.npz') as scene_fields:
        compare_arrays(model_arrays(env.sim.mj_model), {key: scene_fields[key] for key in scene_fields.files})
    shared.PLAN = PLAN
    shared.ReleaseSnapshot = SceneReleaseSnapshot
    shared.zero_check(env, memory, spec, record)
    write(record/'anchor-import-check.json', dict(status='passed', actual_reset_inputs=len(inputs),
        maximum_absolute_error=float(abs(actual-expected).max()), cpu_cuda_rng_unchanged=True, physical_substeps=0))
    return actors


def run():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    assert os.isatty(1), 'Use the original shared VS Code task terminal'
    available()
    spec, pre = read(PLAN), read(OUT/'preflight.json')
    assert pre['status'] == 'passed-cpu-zero-step' and pre['protocol_sha256'] == sha(PLAN)
    assert not (OUT/'coordinator.json').exists(), 'Closed protocols cannot restart'
    check_sources(spec)
    p = spec['environment']
    record = ROOT/'records/evaluations'/p['run_id']
    record.mkdir()
    shutil.copy2(PLAN, record/'protocol.json')
    state = dict(status='running', phase='gpu-zero-step', pid=os.getpid(), run_id=p['run_id'],
        record=record.relative_to(ROOT).as_posix(), protocol_sha256=sha(PLAN), conditions=[],
        started_at=datetime.now(timezone.utc).isoformat(), learning_updates=0, heldout_used=False,
        actual=dict(controls=0, attempted_controls=0, substeps=0, attempted_substeps=0, resets=0))

    def publish():
        write(record/'evaluation.json', state)
        write(OUT/'coordinator.json', state)

    publish()
    write(record/'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), visible_tty=True, pid=os.getpid()))
    write(ROOT/'records/evaluations/current-evaluation.json', dict(path=str(record), run_id=p['run_id'], pid=os.getpid()))
    shared.project_running(True)
    try:
        with (record/'execution.log').open('x', buffering=1) as log, visible_log(log):
            configure_torch_backends(allow_tf32=False)
            torch.set_num_threads(4)
            cfg, display, meta = shared.prepare(p)
            display['recorder'] = 'Complete 11s ActionVideo arms from a single sampled shared fall'
            write(record/'display.json', display)
            dump_yaml(record/'env.yaml', asdict(cfg.env))
            env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
            try:
                obs, _ = env.reset(seed=p['seed'])
                state['actual']['resets'] = 1
                memory = shared.make_memory(p, meta, env.device)
                actors = gpu_startup(env, obs, memory, spec, record)
                collect(env, actors, memory, meta, spec, record, state, publish, available)
            finally:
                env.close()
            assert state['qpos_joint_addresses'] == pre['cpu_joint_addresses']
            state.update(status='collected-review-required', phase='independent-verification')
            publish()
            command = [sys.executable, '-B', str(Path(__file__)), '--verify']
            check = subprocess.run(command, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(check.stdout, end='', flush=True)
            check.check_returncode()
            state = read(OUT/'coordinator.json')
    except BaseException as error:
        state.update(status='stopped-review-required', error=repr(error))
        with (record/'execution.log').open('a') as log:
            traceback.print_exc(file=log)
        raise
    finally:
        state['ended_at'] = datetime.now(timezone.utc).isoformat()
        publish()
        shared.project_running(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.verify:
        from verify_getup_anchor_same_release import verify

        verify()
    else:
        run()
