"""Acquire two bounded backward-only cohorts on the frozen common plant."""
from dataclasses import asdict, replace
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

import common_walk_forward_phase_screen as prior
from common_scene_contract import compare_arrays, model_arrays, nominal_primary
from common_walk_forward_phase_snapshot import WalkingSnapshot
from common_walk_startup import make_actor, td
from getup_anchor_same_release import visible_log
from record_io import write
from run_local_getup_ppo import check_project
from s1_reference_inputs import ROOT, read, sha
import training_live_page as live
from walking_command_controller import create_controller


STAGE = 'S2-backward-onset-development'
DRAFT = ROOT/'configs/experiments/common_walk_backward_onset_draft.json'
PLAN = ROOT/'configs/experiments/common_walk_backward_onset.json'
OUT = ROOT/'records/reports/common-walk-backward-onset'
TZ = ZoneInfo('Asia/Shanghai')


def page(seed):
    return ROOT/f'records/walk-backward-onset-s{seed}-live.html'


class BackwardController:
    """Use the original mapping and its original per-world memory."""
    def __init__(self, plan, settings, desired):
        self.base = create_controller(len(desired), .02)
        self.desired = desired

    def map(self, observed, step, candidate, active):
        return self.base.map(self.desired, observed)

    def reset(self, ids):
        self.base.reset(ids)

    def snapshot(self):
        return self.base.ema.copy(), self.base.ready.copy()

    def restore(self, state):
        self.base.ema[:], self.base.ready[:] = state


def config(p):
    from mjlab.scripts.train import TrainConfig

    cfg = prior.prepare(p)
    run, display, _ = prior.prepare_display(replace(TrainConfig.from_task(p['task']), env=cfg), sparse_snapshots=True)
    assert run.env is cfg and not run.video and not display['native_live_window']
    assert cfg.scene.num_envs == p['parallel_worlds'] and cfg.episode_length_s == 20. and not cfg.auto_reset
    return cfg, display


def environment(spec, seed):
    return dict(spec['environment'], seed=seed, run_id=spec['runs'][str(seed)])


def freeze():
    from mjlab.envs import ManagerBasedRlEnv

    assert not PLAN.exists() and not OUT.exists()
    draft = read(DRAFT)
    source = ROOT/draft['fixed_plant']
    parent = read(source)
    evidence = ROOT/draft['development_trace_review']
    checked = read(evidence)
    assert not checked['heldout_used'] and checked['status'] == 'passed-saved-development-review'
    assert [row['seed'] for row in checked['cohorts']] == [1047, 1048]
    prior.check_sources(parent)
    original = parent['environment']
    n, h = draft['worlds_per_seed'], draft['controls_per_episode']
    seeds = draft['development_seeds']
    assert n*h*len(seeds) == draft['maximum_control_transitions']
    assert 5*h*len(seeds) == draft['maximum_batched_physics_substeps']
    assert h*.02 == draft['episode_seconds']
    assert original['commands']['backward'] == [draft['external_command_m_s'], 0., 0.]
    assert not any(f'_s{seed}_' in f.name for seed in seeds for f in (ROOT/'records/evaluations').iterdir())
    stamp = datetime.now(TZ).strftime('%Y%m%dT%H%M%S')
    p = dict(original, seed=seeds[0], parallel_worlds=n, episodes_per_command=n,
             commands={'backward': original['commands']['backward']}, groups={'backward': list(range(n))},
             stage=STAGE, maximum_control_steps=h, maximum_transitions=n*h)
    _, desired = prior.groups_and_commands(p)
    mapped = create_controller(n).map(desired, np.zeros((n, 61), np.float32))
    np.testing.assert_allclose(mapped, np.tile([draft['expected_policy_command_m_s'], 0., 0.], (n, 1)), atol=1e-8, rtol=0)
    cfg, _ = config(p)
    baseline, _ = config(original)
    assert baseline.scene.terrain.num_envs == baseline.scene.num_envs
    assert cfg.scene.terrain.num_envs == cfg.scene.num_envs
    baseline.seed, baseline.scene.num_envs = cfg.seed, cfg.scene.num_envs
    baseline.scene.terrain.num_envs = cfg.scene.terrain.num_envs
    assert yaml.dump(asdict(baseline)) == yaml.dump(asdict(cfg)), 'Only world count and release seed may differ'
    torch.set_num_threads(4)
    env = ManagerBasedRlEnv(cfg=cfg, device='cpu', render_mode=None)
    try:
        env.reset(seed=seeds[0])
        env.command_manager.get_command('twist').copy_(torch.from_numpy(desired))
        receipt = prior.snapshot_probe(env, parent['maximum_warp_snapshot_bytes'])
        fields = model_arrays(env.sim.mj_model)
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    spec = {key: parent[key] for key in ('source_sha256', 'reference_sha256', 'library_source_sha256',
                                         'maximum_warp_snapshot_bytes')}
    spec.update(status='frozen-before-physics', stage=STAGE, environment=p,
                runs={str(seed):f'duck_walk_backward-onset_development_s{seed}_{stamp}' for seed in seeds},
                phase=dict(warmup_controls=0), maximum_control_transitions=n*h,
                maximum_physics_substeps=5*h, maximum_sampling_wall_seconds=240, full_budget=draft['maximum_control_transitions'],
                block_seconds=1., near_static_max_absolute_block_speed_m_s=.02, heldout_used=False,
                learning_updates=0, hardware_io=False, automatic_extension=False, automatic_retry=False,
                scope='Observe gait onset before any controller change; the original 20-second criteria stay separate.')
    for path in (source, evidence, DRAFT):
        spec['reference_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (Path(__file__).resolve(), Path(__file__).resolve().with_name('verify_walk_backward_onset.py')):
        spec['source_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    prior.check_sources(spec)
    OUT.mkdir()
    np.savez_compressed(OUT/'cpu-scene-fields.npz', **fields)
    spec['reference_sha256'][(OUT/'cpu-scene-fields.npz').relative_to(ROOT).as_posix()] = sha(OUT/'cpu-scene-fields.npz')
    write(PLAN, spec)
    write(OUT/'preflight.json', dict(status='passed-cpu-zero-step', protocol_sha256=sha(PLAN),
          snapshot=receipt, physics_substeps=0, only_seed_and_parallel_world_count_changed=True))
    print('WALK_BACKWARD_ONSET_FROZEN', sha(PLAN), spec['runs'], flush=True)


def checked():
    spec, proof = read(PLAN), read(OUT/'preflight.json')
    assert proof['status'] == 'passed-cpu-zero-step' and proof['protocol_sha256'] == sha(PLAN)
    prior.check_sources(spec)
    return spec


def startup(env, obs, actor, p, desired, spec, record):
    import onnxruntime as ort

    with np.load(OUT/'cpu-scene-fields.npz') as saved:
        compare_arrays(model_arrays(env.sim.mj_model), {name:saved[name] for name in saved.files})
    nominal_primary({k:prior.cpu(getattr(env.sim.model, k)) for k in env.sim.expanded_fields}, env.num_envs)
    snapshot = WalkingSnapshot(env, spec['maximum_warp_snapshot_bytes'])
    before = snapshot.verify()
    assert before['counters'] == [0, 0]
    rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(env.device)
    robot = env.scene['robot']
    ids, names = robot.find_joints(p['joint_names'], preserve_order=True)
    assert names == p['joint_names']
    home = np.asarray(p['model_home'], np.float32)
    observed = prior.cpu(obs['actor'])
    observed[:, 6:20] += prior.cpu(robot.data.default_joint_pos[:, ids])-home
    np.testing.assert_array_equal(observed[:, 34:48], 0.)
    observed[:, 48:51] = create_controller(len(desired)).map(desired, observed)
    inputs = observed[:32]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/p['model']), sess_options=options, providers=['CPUExecutionProvider'])
    expected = np.concatenate([session.run(None, {'obs':row[None]})[0] for row in inputs])
    with torch.no_grad():
        actual = prior.cpu(actor(td(torch.from_numpy(inputs).to(env.device))))
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    term = env.action_manager.get_term('joint_pos')
    raw = torch.full_like(term._raw_actions, 4.)
    env.action_manager.process_action(raw)
    requested = prior.cpu(raw)+home
    limits = np.asarray(p['joint_limits'], np.float32)
    np.testing.assert_array_equal(prior.cpu(term._raw_actions), prior.cpu(raw))
    np.testing.assert_array_equal(prior.cpu(term.requested_target), requested)
    np.testing.assert_array_equal(prior.cpu(term._processed_actions), np.clip(requested, limits[:, 0], limits[:, 1]))
    assert snapshot.restore() == before
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state(env.device))
    np.savez_compressed(record/'startup-inputs.npz', observations=inputs, actor_outputs=actual)
    write(record/'startup.json', dict(status='passed', physics_substeps=0, snapshot=before, model_sha256=p['model_sha256'],
          input_sha256=sha(record/'startup-inputs.npz'), cpu_cuda_rng_unchanged=True,
          scene_exact=True, actual_processor_restored=True, maximum_onnx_error=float(abs(actual-expected).max())))


def sample(seed):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    def available():
        check_project()
        stage = read(ROOT/'project.json')['staged_training_request']
        assert stage['current_stage'] == STAGE and stage['current_protocol_frozen']
        assert ROOT/stage['current_protocol'] == PLAN

    available()
    assert os.isatty(1)
    spec = checked()
    p = environment(spec, seed)
    spec['environment'] = p
    record = ROOT/'records/evaluations'/p['run_id']
    record.mkdir()
    shutil.copy2(PLAN, record/'protocol.json')
    (record/'README.md').write_text('# 后退启动\n\n[冻结方案](../../../configs/experiments/common_walk_backward_onset.json) · '
        '[运行记录](evaluation.json) · [评估日志](execution.log)\n', encoding='utf-8')
    counts = dict(attempted_controls=0, completed_controls=0, attempted_substeps=0, completed_substeps=0, initial_resets=0)
    state = dict(status='running', branch='gpu-zero-step', actual=counts, conditions=[], seed=seed,
                 heldout_used=False, learning_updates=0, protocol_sha256=sha(PLAN), record=record.relative_to(ROOT).as_posix())
    clock, last, rows = perf_counter(), 0, []
    n = p['parallel_worlds']

    def publish(folder, value):
        nonlocal clock, last
        write(folder/'evaluation.json', value)
        count, instant = counts['completed_controls'], perf_counter()
        if count > last:
            rows.append(dict(control=count, transitions_per_second=(count-last)*n/(instant-clock)))
            clock, last = instant, count
        video = folder/'evaluation-backward'
        clips = read(video/'clips.json') if (video/'clips.json').is_file() else []
        clip = next((video/row['file'] for row in clips if row['complete_clip']), None)
        frame = video/'frame-step-0.jpg'
        live.publish(folder, dict(status=value['status'], run_id=p['run_id'], purpose='后退启动 · 固定策略',
                     updated_at=datetime.now(TZ).isoformat(), controls=count, maximum_controls=p['maximum_control_steps'],
                     transitions=n*count, learning_updates=0, updates=rows,
                     latest_clip=clip.relative_to(folder).as_posix() if clip else None,
                     frame=frame.relative_to(folder).as_posix() if frame.is_file() else None))

    with (record/'execution.log').open('x', buffering=1) as log, visible_log(log), patch.object(live, 'PAGE', page(seed)):
        publish(record, state)
        try:
            configure_torch_backends(allow_tf32=False)
            torch.set_num_threads(4)
            cfg, display = config(p)
            write(record/'display.json', display)
            dump_yaml(record/'env.yaml', asdict(cfg))
            _, desired = prior.groups_and_commands(p)
            env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode=None)
            try:
                obs, _ = env.reset(seed=seed)
                counts['initial_resets'] = 1
                env.command_manager.get_command('twist').copy_(torch.from_numpy(desired).to(env.device))
                actor = make_actor(p, env.device)
                startup(env, obs, actor, p, desired, spec, record)
                print('WALK_BACKWARD_ONSET_CUDA_ZERO', seed, flush=True)
                with patch.object(prior, 'ROLES', ('backward',)), patch.object(prior, 'ForwardPhase', BackwardController), \
                     patch.object(prior, 'publish', publish), patch.object(prior, 'available', available):
                    prior.collect(env, actor, spec, record, state, desired)
            finally:
                env.close()
            state.update(status='collected-review-required', branch='verification')
        except BaseException as exc:
            state.update(status='stopped-review-required', error=repr(exc))
            raise
        finally:
            for row in state['conditions']:
                row['condition'] = 'backward'
            publish(record, state)


if __name__ == '__main__':
    check_project()
    if sys.argv[1:] == ['--freeze']:
        freeze()
    else:
        assert len(sys.argv) == 3 and sys.argv[1] == '--sample'
        sample(int(sys.argv[2]))
