"""Read back seven-command CUDA targets without advancing the physical scene."""

import fcntl
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from check_routed_seven_command_cpu import DRAFT, OUT, ROOT, prior, read, sha, sources, write
from command_input_mapping import transform
from routed_getup_sequence import FORWARD, GETUP, STAND, ForwardFeedback
from routed_seven_command_controller import SevenCommandSequenceController


def target_fixture(env, sensed, spec, delivery, seed):
    p, s = spec['environment'], spec['sequence']
    commands, routes = delivery['commands'], delivery['route_by_command']
    groups = {name: [i] for i, name in enumerate(commands)}
    groups['forward'].append(7)
    home = spec['switch_policies']['stand']['home']
    sessions, bank = {}, {}
    for role, details in delivery['models'].items():
        model = details['files']['onnx']
        sessions[role] = prior.original.session(prior.route.pinned((model['path'], model['sha256'])))
        bank[role] = prior.route.guarded_actor(dict(model=model['path'], model_sha256=model['sha256'],
            seed=seed, home=home), env.device)
    ctrl = SevenCommandSequenceController(8, env.device, s['controller'], [p['model_home'], home, home],
        p['joint_limits'], ForwardFeedback(spec['walking'], 8), commands=commands, groups=groups, routes=routes,
        ramp=spec['ramped_forward']['ramp'])
    ctrl.mode.fill_(STAND)
    ctrl.dwell.fill_(ctrl.dwell_steps)
    ctrl.history[STAND] = torch.linspace(-.08, .08, 8*14, device=env.device).reshape(8, 14)
    carried = ctrl.history[STAND].clone()
    # Retain the actual sensor layout and joint values. Set the IMU fixture
    # so seven worlds are ready to switch and one stays in standing.
    x0 = sensed[:8].clone()
    x0[:, :6] = torch.tensor([0., 0., .1, 0., 0., -1.], device=env.device)
    x0[7, 0] = 2.
    native = env.action_manager.get_term('joint_pos')
    fields = ('requested_target', 'projected_destination', '_processed_actions', 'previous_target')
    term = SimpleNamespace(**{key: getattr(native, key)[:8] for key in fields})
    saved = {key: getattr(term, key).clone() for key in fields}
    previous = ctrl.homes[STAND].expand(8, 14).clone()
    actors = {GETUP: lambda data: torch.zeros_like(data['actor'][:, :14]), STAND: bank['base'], FORWARD: bank}
    start = ctrl.forward_at
    worst = max_step = 0.
    try:
        with torch.inference_mode():
            for k in range(start, start + 26):
                x, bridge = ctrl.advance(k, x0)
                assert ctrl.mode.tolist() == [FORWARD]*7 + [STAND]
                assert bridge.tolist() == [k < start + 25]*7 + [False]
                raw, yaw, alternate = ctrl.infer(k, x, actors)
                assert not yaw.any() and not alternate.any()
                if k in (start, start + 24, start + 25):
                    for row, name in enumerate((*commands, 'stand')):
                        external = np.asarray(commands[name], dtype=float)
                        if name == 'forward':
                            external *= min(1., (k - start + 1) / 25.)
                        np.testing.assert_array_equal(x[row, 48:51].cpu(), transform(external))
                        if k == start:
                            expected_history = carried[row] if routes[name] == 'base' else torch.zeros_like(carried[row])
                            torch.testing.assert_close(x[row, 34:48], expected_history, atol=0., rtol=0.)
                        expected = sessions[routes[name]].run(None, {'obs': x[row:row+1].cpu().numpy()})[0][0]
                        error = abs(raw[row].cpu().numpy() - expected).max()
                        assert error <= 2e-5
                        worst = max(worst, float(error))
                ctrl.apply(term, previous, raw, alternate, bridge, k, .02)
                if k == start:
                    max_step = float((term._processed_actions[:7] - previous[:7]).abs().max())
                    assert max_step <= .0400002
                previous.copy_(term._processed_actions)
    finally:
        for key, value in saved.items():
            getattr(term, key).copy_(value)
            torch.testing.assert_close(getattr(term, key), value, atol=0., rtol=0.)
    return dict(actor_queries=24, maximum_onnx_error_rad=worst, maximum_target_step_rad=max_step,
        native_target_storage_restored=True, fixture_control_calls=26)


def main():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    d, spec, delivery, paths = sources()
    prior.original.require_stage(DRAFT, d['stage'])
    target, cpu = OUT/'cuda-zero.json', OUT/'preflight.json'
    assert not target.exists() and not torch.cuda.is_initialized()
    witness = read(cpu)
    assert witness['status'] == 'passed-cpu-seven-route-handoff'
    for name, expected in witness['frozen_inputs'].items():
        assert sha(ROOT/name) == expected, name
    seed = d['development_seed']
    poses = np.random.default_rng(seed).permutation(np.repeat(np.arange(4), d['requested_pose_counts']))
    base = prior.base
    groups = {name: np.flatnonzero(poses == i).tolist() for i, name in enumerate(base.shared.POSES)}
    environment = dict(spec['environment'], stage=d['stage'], seed=seed,
        run_id=f'duck_switch_routed-seven-command_cuda-zero_s{seed}', split=dict(pose_codes=poses.tolist(), groups=groups))
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg, _, _ = prior.original.prepare(environment)
    watch = base.MemoryWatch(d['maximum_gpu_memory_fraction'])
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock, watch.active():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
        try:
            obs, _ = env.reset(seed=seed)
            assert env.num_envs == d['worlds'] and obs['actor'].shape == (d['worlds'], 61)
            assert env.common_step_counter == env._sim_step_counter == 0
            before = obs['actor'].clone()
            proof = target_fixture(env, obs['actor'], spec, delivery, seed)
            torch.testing.assert_close(obs['actor'], before, atol=0., rtol=0.)
            assert env.common_step_counter == env._sim_step_counter == 0
            prior.original.require_stage(DRAFT, d['stage'])
            watch.check()
            watch.sample()
        finally:
            env.close()
        write(target, dict(status='passed-cuda-seven-route-zero', physical_substeps=0, world_transitions=0,
            resets=1, worlds=d['worlds'], development_seed=seed, hardware_io=False, heldout_used=False,
            proof=proof, gpu=watch.summary(),
            source_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in (*paths, cpu, Path(__file__).resolve())}))
    print('ROUTED_SEVEN_COMMAND_CUDA_ZERO', sha(target), proof, flush=True)


if __name__ == '__main__':
    main()
