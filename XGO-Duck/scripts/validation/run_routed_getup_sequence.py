"""Freeze and execute the 32-second getup, stand and forward development batch."""

from argparse import ArgumentParser
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import fcntl
from pathlib import Path
from shutil import copyfile, disk_usage
import sys
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import onnxruntime as ort
import torch

import evaluate_getup_dual_pose_route as route
from collect_routed_getup_sequence import collect
from routed_getup_sequence import FORWARD, GETUP, STAND, ForwardFeedback, SequenceController
from run_getup_latched_head_corrective_ppo import require_stage


base, ROOT, read, sha, write = route.base, route.ROOT, route.read, route.sha, route.write
DRAFT = ROOT / 'configs/experiments/continuous_routed_getup_stand_walk_draft.json'
PLAN = DRAFT.with_stem('continuous_routed_getup_stand_walk')
OUT = ROOT / 'records/reports/continuous-routed-getup-stand-walk'
PAGE = ROOT / 'records/continuous-routed-getup-stand-walk-live.html'
STAGE, SEED = 'S5-routed-getup-stand-walk', 1170


def phase(protocol):
    require_stage(protocol, STAGE + '-preparation' if protocol == DRAFT else STAGE)


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def prepare(p):
    cfg, display, meta = ORIGINAL_PREPARE(p)
    assert cfg.env.episode_length_s == 11 and cfg.env.decimation == 5 and not cfg.env.auto_reset
    cfg.env.episode_length_s = 32.
    return cfg, display, meta


ORIGINAL_PREPARE = base.shared.prepare


def switch_models(spec, device):
    return {name: route.guarded_actor(item, device) for name, item in spec['switch_policies'].items()}


def boundary_fixture(spec, actors, device, actual_term=None):
    """Check both incoming actors, skipped worlds and native target storage at zero physics."""
    p, s = spec['environment'], spec['sequence']
    homes = [p['model_home'], spec['switch_policies']['stand']['home'], spec['switch_policies']['forward']['home']]
    feedback = ForwardFeedback(spec['walking'], 3)
    controller = SequenceController(3, device, s['controller'], homes, p['joint_limits'], feedback)
    sensed = torch.zeros((3, 61), device=device)
    sensed[:, 5] = -1.
    sensed[2, 5], sensed[2, 3] = 0., 1.
    controller.history[GETUP].fill_(.123)
    fields = ('requested_target', 'projected_destination', '_processed_actions', 'previous_target')
    term = SimpleNamespace(**{key: torch.zeros((3, 14), device=device) if actual_term is None
        else getattr(actual_term, key)[:3] for key in fields})
    saved = {key: getattr(term, key).clone() for key in fields}
    incoming = {GETUP: lambda data: torch.full_like(data['actor'][:, 6:20], .123),
        STAND: actors['stand'], FORWARD: actors['forward']}
    checks = {}
    try:
        with torch.inference_mode():
            for control in range(526, 551):
                x, bridge = controller.advance(control, sensed)
            assert controller.mode.tolist() == [STAND, STAND, GETUP]
            assert bridge.tolist() == [True, True, False]
            assert not x[:2, 34:48].any() and torch.equal(x[2, 34:48], controller.history[GETUP, 2])
            for boundary, role, name in ((550, STAND, 'stand'), (600, FORWARD, 'forward')):
                if role == FORWARD:
                    sensed[1, 0] = 2.
                    for control in range(551, 601):
                        x, bridge = controller.advance(control, sensed)
                    assert controller.mode.tolist() == [FORWARD, STAND, GETUP]
                    assert bridge.tolist() == [True, False, False]
                    assert not x[0, 34:48].any()
                    assert torch.equal(x[1, 34:48], controller.history[STAND, 1])
                ids = controller.mode == role
                torch.testing.assert_close(x[ids, 6:20] + controller.homes[role],
                    sensed[ids, 6:20] + controller.homes[GETUP], atol=2e-7, rtol=0)
                raw, yaw, alternate = controller.infer(boundary, x, incoming)
                assert not yaw.any() and not alternate.any()
                onnx = session(ROOT / spec['switch_policies'][name]['model'])
                expected = np.concatenate([onnx.run(None, {'obs': row[None]})[0] for row in base.shared.cpu(x[ids])])
                actual = base.shared.cpu(raw[ids])
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
                previous = controller.homes[GETUP].expand(3, 14).clone()
                unchanged = term._processed_actions[2].clone()
                controller.apply(term, previous, raw, alternate, bridge, boundary, .02)
                step = float((term._processed_actions[ids] - previous[ids]).abs().max())
                assert step <= .0400002
                torch.testing.assert_close(term._processed_actions[2], unchanged, atol=0, rtol=0)
                checks[name] = dict(selected=int(ids.sum()), maximum_import_error_rad=float(abs(actual-expected).max()),
                    maximum_target_step_rad=step, own_history=True, previous_target_bridge=True)
    finally:
        for key, value in saved.items():
            getattr(term, key).copy_(value)
    return checks


def freeze():
    from mjlab.scene import Scene

    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE)) and not torch.cuda.is_initialized()
    d = read(DRAFT)
    assert d['proposed_development_seed'] == SEED and SEED not in d['excluded_for_tuning']
    assert not any(d[key] for key in ('physical_substeps_authorized', 'learning_updates', 'heldout_used', 'hardware_io'))
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{SEED}_' in item.name for item in (ROOT / folder).iterdir())
    links = [route.pinned(d[key]) for key in ('getup_delivery', 'walking_delivery', 'boundary_check',
        'walking_criteria', 'existing_continuous_forward_evidence')]
    delivery, walking, boundary, walk, roll = map(read, links)
    assert delivery['status'] == 'verified-four-pose-getup-simulation-components'
    assert walking['status'] == 'verified-seven-command-simulation-components'
    assert boundary['status'] == 'passed-cpu-request-and-slew-check' and not boundary['heldout_used']
    assert roll['passed_development_gate']
    source = route.pinned((delivery['heldout']['protocol']['path'], delivery['heldout']['protocol']['sha256']))
    decision = route.pinned((delivery['heldout']['decision']['path'], delivery['heldout']['decision']['sha256']))
    assert read(decision)['final_holdout_passed']
    spec = read(source)
    p = spec['environment']
    assert p['criteria']['max_seconds'] == 11. and not p['handoff_enabled'] and not p['neck_home_enabled']
    expected = {'fallback': spec['common_fallback'], 'supine': spec['policies']['reference-a'],
                'right-side': spec['policies']['candidate']}
    for label, actor in expected.items():
        item = delivery['components'][label]['files']['onnx']
        assert actor['model_sha256'] == item['sha256'] == sha(ROOT / item['path'])
    original_walk, = (item['environment'] for item in walk['cases'].values())
    feedback = {key: walk[key] for key in ('observer_receipt', 'observer_protocol', 'direction_predictor', 'feedback_screen',
        'feedback', 'phase', 'shared_prefix_controls', 'direction_pilot')}
    feedback.update({key: original_walk[key] for key in ('criteria', 'commands', 'joint_names', 'joint_limits')})
    assert feedback['joint_names'] == delivery['joint_order'] == walking['joint_names']
    assert feedback['commands']['forward'][0] == d['controller']['forward_command_m_per_s']
    np.testing.assert_allclose(feedback['joint_limits'], p['joint_limits'], atol=1e-7, rtol=0)
    assert feedback['feedback']['maximum_joint_correction_rad'] == walking['maximum_joint_correction_rad'] == .03
    assert (feedback['phase']['warmup_controls'], feedback['shared_prefix_controls']) == (50, 84)
    switch = {}
    for name in ('stand', 'forward'):
        item = walking['models'][walking['route_by_command'][name]]['files']['onnx']
        model = route.pinned((item['path'], item['sha256']))
        onnx = session(model)
        metadata = onnx.get_modelmeta().custom_metadata_map
        assert metadata['joint_names'].split(',') == walking['joint_names']
        assert float(metadata['action_scale']) == 1.
        assert onnx.get_inputs()[0].shape == [1, 61] and onnx.get_outputs()[0].shape == [1, 14]
        switch[name] = dict(model=item['path'], model_sha256=item['sha256'], seed=SEED,
            home=list(map(float, metadata['default_joint_pos'].split(','))))
        links.append(model)
    assert switch['stand']['model_sha256'] == boundary['standing_model']['sha256']
    steps = np.cumsum([item['controls'] for item in d['phases']]).tolist()
    assert steps == [100, 550, 600, 1600] and steps[-1] * 5 == d['maximum_physics_substeps']
    scale = d['scale_options'][0]
    n = scale['worlds']
    assert n == p['num_envs'] == 3072 and scale['maximum_transitions'] == n * steps[-1]
    assert disk_usage(ROOT).free >= 12 * 1024 ** 3
    codes = np.random.default_rng(SEED).permutation(np.repeat(np.arange(4), scale['requested_pose_counts']))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(base.shared.POSES)}
    run_id = f'duck_switch_routed-getup-stand-walk_development_s{SEED}_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    record = ROOT / 'records/evaluations' / run_id
    p.update(seed=SEED, stage=STAGE, run_id=run_id, split=dict(pose_codes=codes.tolist(), groups=groups))
    spec['dual_route'].update(seed=SEED, reserved_holdout_seed=None, requested_pose_counts=scale['requested_pose_counts'])
    for policy in (*spec['policies'].values(), spec['common_fallback']):
        policy['seed'] = SEED
    spec.update(stage=STAGE, record=base.rel(record), page=base.rel(PAGE), data_scope='development',
        reserved_holdout=False, heldout_used=False, maximum_unique_controls=steps[-1], maximum_physics_substeps=steps[-1]*5,
        maximum_sampling_wall_seconds=900, switch_policies=switch, walking=feedback,
        sequence=dict(d, boundary_controls=steps[:-1], record_window_controls=200, maximum_transitions=scale['maximum_transitions']))
    spec.pop('holdout_admission')
    spec['draft']['gpu_memory_fraction_cap'] = d['maximum_gpu_memory_fraction']
    spec['references'].update({base.rel(path): sha(path) for path in (DRAFT, source, decision, *links)})
    for path in (Path(__file__), Path(__file__).with_name('collect_routed_getup_sequence.py'),
                 Path(__file__).with_name('routed_getup_sequence.py'), Path(__file__).with_name('verify_routed_getup_sequence.py')):
        spec['source_sha256'][base.rel(path.resolve())] = sha(path)
    # Preserve the actual S2 implementation, observer coefficients and command map.
    for key in ('run_walk_forward_yaw_direction_pilot.py', 'walking_actuator_yaw_feedback.py',
                'command_input_mapping.py', 'roll_stand_walk_sequence.py',
                'common_walk_forward_phase_metrics.py', 'planar_block_metrics.py'):
        path = f'scripts/validation/{key}'
        spec['source_sha256'][path] = sha(ROOT / path)
        if path in walk['source_sha256']:
            assert spec['source_sha256'][path] == walk['source_sha256'][path]
    for key in ('observer_receipt', 'observer_protocol', 'direction_predictor', 'feedback_screen'):
        path = feedback[key]
        spec['references'][path] = sha(ROOT / path)
        assert spec['references'][path] == walk['reference_sha256'][path]
    fit = 'configs/experiments/common_walk_forward_yaw_prediction.json'
    spec['references'][fit] = sha(ROOT / fit)
    base.shared.assert_sources(spec)
    cfg, _, meta = prepare(p)
    assert cfg.env.scene.num_envs == n and cfg.env.episode_length_s == 32
    base.exact_scene(Scene(cfg.env.scene, device='cpu').compile())
    torch.set_num_threads(4)
    proof = boundary_fixture(spec, switch_models(spec, 'cpu'), 'cpu')
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    record.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(status='passed-cpu-freeze', protocol_sha256=sha(PLAN),
        physical_substeps=0, cpu_scene_exact=True, boundaries=proof))
    (record / 'README.md').write_text('# 起身、站立与前进连续评估\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../continuous-routed-getup-stand-walk-live.html)\n', encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='起身、站立与前进连续评估', run_id=run_id,
        controls=0, maximum_controls=steps[-1], transitions=0, updates=[], learning_updates=0))
    print('ROUTED_SEQUENCE_FROZEN', base.rel(record), sha(PLAN), flush=True)


def snapshot(env, obs, memory, spec, target):
    actors, proof, fields, inputs = route.snapshot(env, obs, memory, spec, target)
    term = env.action_manager.get_term('joint_pos')
    assert env.scene['robot'].data.joint_pos_target.shape[0] == env.num_envs
    proof['sequence_boundaries'] = boundary_fixture(spec, switch_models(spec, env.device), env.device, term)
    assert env.common_step_counter == env._sim_step_counter == 0
    return actors, proof, fields, inputs


def sample():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    phase(PLAN)
    spec = read(PLAN)
    base.shared.assert_sources(spec)
    record = ROOT / spec['record']
    assert not (record / 'evaluation.json').exists() and not (OUT / 'native-zero.json').exists()
    assert sha(PLAN) == sha(record / 'protocol.json') == read(OUT / 'freeze.json')['protocol_sha256']
    state = read(record / 'state.json')
    ready = read(record / 'windows-observer-ready.json')
    assert ready['status'] == 'starting-page-observed' and ready['protocol_sha256'] == sha(PLAN)
    assert ready['page_sha256'] == sha(PAGE) and ready['initial_state_sha256'] == sha(record / 'state.json')
    assert state['status'] == 'starting' and disk_usage(ROOT).free >= 12 * 1024 ** 3
    p = spec['environment']
    state.update(actual=dict(controls=0, attempted_controls=0, substeps=0, attempted_substeps=0, resets=0),
        protocol_sha256=sha(PLAN), phase='gpu-zero', heldout_used=False, hardware_io=False)
    watch = base.MemoryWatch(spec['draft']['gpu_memory_fraction_cap'])
    last, timestamp = 0, monotonic()

    def guard():
        phase(PLAN)
        watch.check()

    def publish():
        nonlocal last, timestamp
        k, now = state['actual']['controls'], monotonic()
        if k > last:
            state['updates'].append(dict(control=k, transitions_per_second=(k-last)*p['num_envs']/(now-timestamp)))
            last, timestamp = k, now
        state.update(controls=k, transitions=k*p['num_envs'])
        write(record / 'evaluation.json', state)
        base.publish(record, PAGE, state)

    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg, _, meta = prepare(p)
    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock, watch.active():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (record / 'execution.log').open('x', buffering=1) as log, \
                redirect_stdout(base.Tee(sys.stdout, log)), redirect_stderr(base.Tee(sys.stderr, log)):
            write(record / 'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), protocol_sha256=sha(PLAN)))
            env = None
            try:
                env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
                obs, _ = env.reset(seed=SEED)
                state['actual']['resets'] = 1
                memory = base.shared.make_memory(p, meta, env.device)
                actors, proof = route.prior.qualify(env, obs, memory, spec, record)
                independent = read(OUT / 'independent-zero/proof.json')
                assert proof['sequence_boundaries'] == independent['sequence_boundaries']
                guard()
                watch.sample()
                write(OUT / 'native-zero.json', dict(status='passed-native-zero', proof=proof, gpu=watch.summary(),
                    qualification_sha256=sha(record / 'zero-qualification.json')))
                walking = switch_models(spec, env.device)
                state['status'] = 'running'
                base.shared.project_running(True)
                publish()
                collect(env, actors, walking, meta, spec, record, state, publish, guard)
                assert state['actual'] == dict(controls=1600, attempted_controls=1600, substeps=8000,
                    attempted_substeps=8000, resets=1)
                guard()
                state.update(status='completed', phase='independent-score-pending', latest_clip=state['video']['file'])
            except BaseException as error:
                state.update(status='stopped-review-required', error=repr(error))
                raise
            finally:
                if env is not None:
                    env.close()
                state['gpu'] = watch.summary()
                write(record / 'gpu-telemetry.json', watch.rows)
                base.shared.project_running(False)
                publish()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample'))
    mode = parser.parse_args().mode
    if mode == 'freeze':
        return freeze()
    with patch.multiple(route.prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase, snapshot=snapshot), \
         patch.object(base.shared, 'prepare', prepare), patch.object(base.shared, 'actor_model', route.guarded_actor):
        (route.prior.zero if mode == 'zero' else sample)()


if __name__ == '__main__':
    main()
