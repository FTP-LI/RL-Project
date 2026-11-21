"""Freeze and capture a bounded impulse on the seven-command common plant."""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil
import sys
from time import perf_counter
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import common_walk_forward_phase_screen as prior
from common_walk_forward_phase_screen import check_sources, groups_and_commands, make_config, snapshot_probe
from getup_anchor_same_release import visible_log
from record_io import write
from run_local_getup_ppo import check_project
from s1_reference_inputs import ROOT, read, sha
import training_live_page as live
from walking_forward_yaw_pulse import ForwardYawPulse


PLAN = ROOT/'configs/experiments/common_walk_yaw_pulse.json'
OUT = ROOT/'records/reports/common-walk-yaw-pulse'
PAGE = ROOT/'records/walk-yaw-pulse-live.html'
STAGE = 'S2-forward-yaw-impulse-development'
ROLES = ('reference-a', 'reference-b', 'pulse-yaw')


def component_probe(p, settings, desired):
    a, b = (ForwardYawPulse(p, settings, desired) for _ in range(2))
    obs = np.random.default_rng(104500).normal(size=(len(desired), 61)).astype(np.float32)
    active = np.ones(len(desired), dtype=bool)
    active[p['groups']['forward'][-1]] = False
    step = settings['warmup_controls'] + 1
    expected = np.zeros_like(desired)
    expected[active, 2] = np.asarray(settings['pulse_signs'], np.float32)[active]*np.float32(settings['peak_rad_s'])
    for k in (step-1, step, step+1, step+6):
        reference = a.map(obs, k, False, active)
        before = b.snapshot()
        observed = b.map(obs, k, True, active)
        np.testing.assert_array_equal(observed-reference, expected if k == step else 0.)
        b.restore(before)
        np.testing.assert_array_equal(observed, b.map(obs, k, True, active))
    return dict(pulse_controls=1, disabled_world_unchanged=True, exact_other_commands=True,
                controller_restore_exact=True)


def freeze():
    from mjlab.envs import ManagerBasedRlEnv

    assert not PLAN.exists() and not OUT.exists()
    spec = read(prior.PLAN)
    basis = ROOT/'records/reports/common-walk-confirmation/development-yaw-forecast.json'
    screen = prior.OUT/'analysis.json'
    previous = read(screen)
    assert previous['protocol_sha256'] == sha(prior.PLAN) and not previous['continued_admission']
    forecast = read(basis)
    assert not forecast['heldout_used'] and not forecast['combined_20ms_predictor_nominated']
    assert all(x['paired_mean_gain_over_gyro_rad_s'] >= .01 for x in forecast['results']['0.06']['cohorts'].values())
    p = spec['environment']
    seed, h, k = 1045, 200, 50
    assert not any(f'_s{seed}_' in item.name for item in (ROOT/'records/evaluations').iterdir())
    rng = np.random.default_rng(70000+seed)
    n = p['parallel_worlds']
    permutation = rng.permutation(n)
    slot = list(p['commands']).index('forward')*32
    origin = np.flatnonzero(permutation == 0).item()
    permutation[slot], permutation[origin] = permutation[origin], permutation[slot]
    groups = {name: permutation[i*32:(i+1)*32].tolist() for i, name in enumerate(p['commands'])}
    signs = np.zeros(n, dtype=np.int8)
    signs[groups['forward']] = rng.permutation(np.repeat([-1, 1], 16))
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    p.update(seed=seed, assignment_seed=70000+seed, stage=STAGE, groups=groups, maximum_control_steps=h,
             run_id=f'duck_walk_yaw-pulse_paired_s{seed}_{stamp}')
    controls = k+len(ROLES)*(h-k)
    spec.update(status='frozen-before-physical-controls', stage=STAGE, roles=list(ROLES),
                phase=dict(warmup_controls=k, peak_rad_s=.06, pulse_signs=signs.tolist()),
                maximum_control_transitions=n*controls, maximum_physics_substeps=5*controls,
                maximum_sampling_wall_seconds=300, learning_updates=0, heldout_used=False,
                automatic_retry=False, automatic_extension=False,
                gate=dict(bootstrap_seed=104501, bootstrap_samples=10000, minimum_positive_worlds=24,
                          minimum_effect_rad_s=.001, maximum_placebo_rad_s=.001,
                          minimum_60ms_fraction_of_20ms=.5, tracking_gate_unchanged_rad_s=.01),
                scope='One signed 20 ms perturbation per forward world. Short causal diagnostic; original full seven-command scores and heldout kept separate.')
    for path in (prior.PLAN, screen, basis):
        spec['reference_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (Path(__file__), Path(__file__).with_name('verify_common_walk_yaw_pulse.py'),
                 Path(__file__).with_name('walking_forward_yaw_pulse.py'), ROOT/'scripts/training/training_live_page.py'):
        spec['source_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    check_sources(spec)
    _, desired = groups_and_commands(p)
    component = component_probe(p, spec['phase'], desired)
    torch.set_num_threads(4)
    cfg, _ = make_config(p)
    env = ManagerBasedRlEnv(cfg=cfg, device='cpu', render_mode=None)
    try:
        env.reset(seed=seed)
        env.command_manager.get_command('twist').copy_(torch.from_numpy(desired))
        snapshot = snapshot_probe(env, spec['maximum_warp_snapshot_bytes'])
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT/'preflight.json', dict(status='passed-cpu-zero-step', protocol_sha256=sha(PLAN),
          component=component, snapshot=snapshot, physical_substeps=0, learning_updates=0, heldout_used=False))
    print('WALK_YAW_PULSE_FROZEN', sha(PLAN), p['run_id'], spec['maximum_control_transitions'], flush=True)


def checked():
    spec, pre = read(PLAN), read(OUT/'preflight.json')
    assert pre['status'] == 'passed-cpu-zero-step' and pre['protocol_sha256'] == sha(PLAN)
    check_sources(spec)
    return spec


def sample(spec):
    import os
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    assert os.isatty(1) and not (OUT/'coordinator.json').exists()
    p = spec['environment']
    record = ROOT/'records/evaluations'/p['run_id']
    record.mkdir()
    shutil.copy2(PLAN, record/'protocol.json')
    (record/'README.md').write_text('# 前进偏航单步脉冲\n\n[冻结方案](../../../configs/experiments/common_walk_yaw_pulse.json) · '
        '[试验记录](../../reports/common-walk-yaw-pulse/README.md) · [原始状态](evaluation.json)\n', encoding='utf-8')
    counts = dict(attempted_controls=0, completed_controls=0, attempted_substeps=0,
                  completed_substeps=0, initial_resets=0)
    state = dict(status='running', branch='gpu-zero-step', actual=counts, conditions=[], learning_updates=0,
                 heldout_used=False, protocol_sha256=sha(PLAN), record=record.relative_to(ROOT).as_posix())
    last, clock = 0, perf_counter()
    rows = []
    n = p['parallel_worlds']

    def publish(folder, value):
        nonlocal last, clock
        write(folder/'evaluation.json', value)
        write(OUT/'coordinator.json', value)
        count, instant = value['actual']['completed_controls'], perf_counter()
        if count > last:
            rows.append(dict(control=count, transitions_per_second=(count-last)*n/(instant-clock)))
            last, clock = count, instant
        clip = next((r['video']['path'] for r in reversed(value['conditions']) if r['video']['complete_clip']), None)
        frame = next(iter(folder.glob('evaluation-pulse-yaw/frame-step-0.jpg')), None)
        live.publish(folder, dict(status=value['status'], run_id=p['run_id'],
            purpose=f"偏航单步脉冲 · {value['branch']}", updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            controls=count, maximum_controls=spec['maximum_control_transitions']//n, transitions=count*n,
            learning_updates=0, updates=rows, latest_clip=clip,
            frame=frame.relative_to(folder).as_posix() if frame else None))

    def available():
        check_project()
        request = read(ROOT/'project.json')['staged_training_request']
        assert request['current_stage'] == STAGE and request['current_protocol_frozen']
        assert ROOT/request['current_protocol'] == PLAN

    available()
    write(record/'command.json', dict(argv=sys.argv, visible_tty=os.isatty(1)))
    with (record/'execution.log').open('x', buffering=1) as log, visible_log(log), patch.object(live, 'PAGE', PAGE):
        publish(record, state)
        try:
            configure_torch_backends(allow_tf32=False)
            torch.set_num_threads(4)
            cfg, display = make_config(p)
            write(record/'display.json', display)
            dump_yaml(record/'env.yaml', asdict(cfg))
            _, desired = groups_and_commands(p)
            env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode=None)
            try:
                obs, _ = env.reset(seed=p['seed'])
                counts['initial_resets'] = 1
                env.command_manager.get_command('twist').copy_(torch.as_tensor(desired, device=env.device))
                actor = prior.make_actor(p, env.device)
                prior.startup(env, obs, actor, p, record, prior.PARENT, desired)
                snapshot = snapshot_probe(env, spec['maximum_warp_snapshot_bytes'])
                write(record/'zero-snapshot.json', dict(status='passed', physical_substeps=0, snapshot=snapshot))
                print('WALK_YAW_PULSE_GPU_ZERO_OK', snapshot['fields'], flush=True)
                with patch.object(prior, 'ForwardPhase', ForwardYawPulse), patch.object(prior, 'ROLES', ROLES), \
                     patch.object(prior, 'available', available), patch.object(prior, 'publish', publish):
                    prior.collect(env, actor, spec, record, state, desired)
            finally:
                env.close()
            state.update(status='collected-review-required', branch='verification')
        except BaseException as exc:
            state.update(status='stopped-review-required', error=repr(exc))
            raise
        finally:
            publish(record, state)


if __name__ == '__main__':
    check_project()
    assert sys.argv[1:] in (['--freeze'], ['--sample'])
    if sys.argv[1] == '--freeze':
        freeze()
    else:
        sample(checked())
