"""Freeze, qualify and sample the full shared-standing turn feedback pair."""

from argparse import ArgumentParser
from datetime import datetime
from math import ceil
from pathlib import Path
from shutil import copyfile, disk_usage
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_routed_forward_entry_ramp as pair_runner
import run_routed_seven_command as prior
from collect_routed_turn_feedback_pair import ROLES, collect, controllers
from routed_getup_sequence import FORWARD, STAND
from routed_seven_command_controller import balanced_groups
from walking_command_controller import create_controller


ROOT, read, sha, write = prior.ROOT, prior.read, prior.sha, prior.write
base, route, original = prior.base, prior.route, prior.original
DRAFT = ROOT/'configs/experiments/continuous_routed_turn_feedback_pair_draft.json'
PLAN = DRAFT.with_stem('continuous_routed_turn_feedback_pair')
OUT = ROOT/'records/reports/continuous-routed-turn-feedback-pair'
PAGE = ROOT/'records/continuous-routed-turn-feedback-pair-live.html'
STAGE = 'S5-routed-turn-feedback-pair'
PREPARATION = 'S5-routed-turn-feedback-recovery-preparation'


def phase(protocol):
    original.require_stage(protocol, PREPARATION if protocol == DRAFT else STAGE)


def feedback_fixture(spec, device):
    """Use the native device for all seven boundary inputs and read back both turn maps."""
    seven = spec['seven_command']
    d = spec['sequence']['boundary_controls'][-1]
    banks = controllers(spec, device)
    modes = banks[ROLES[0]].mode
    sensed = torch.zeros((len(modes), 61), device=device)
    sensed[:, 5] = -1.
    expected = create_controller(len(modes), .02)
    requests = np.empty((len(modes),3), np.float32)
    turn = np.zeros(len(modes), bool)
    for name, ids in seven['groups'].items():
        requests[ids] = seven['commands'][name]
        turn[ids] = name in ('turn-left', 'turn-right')
    for bank in banks.values():
        bank.mode.fill_(STAND)
        bank.dwell.fill_(bank.dwell_steps)
    turn_device = torch.as_tensor(turn, device=device)
    with torch.inference_mode():
        for k, gyro in enumerate((.15, -.26, .51)):
            sensed[:, 2] = gyro
            a, b = (bank.advance(d+k, sensed) for bank in banks.values())
            assert all(bool((bank.mode == FORWARD).all()) for bank in banks.values())
            torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
            torch.testing.assert_close(a[0][~turn_device], b[0][~turn_device], atol=0, rtol=0)
            torch.testing.assert_close(a[0][turn_device, :50], b[0][turn_device, :50], atol=0, rtol=0)
            torch.testing.assert_close(a[0][turn_device, 51:], b[0][turn_device, 51:], atol=0, rtol=0)
            ref = expected.map(requests, sensed.cpu().numpy())[turn, 2]
            np.testing.assert_array_equal(b[0][turn_device, 50].cpu(), ref)
    return dict(calls=3, turn_worlds=int(turn.sum()), other_worlds=int((~turn).sum()),
                identical_other_inputs=True, exact_accepted_turn_input=True)


def freeze():
    from mjlab.scene import Scene

    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE)) and not torch.cuda.is_initialized()
    d = read(DRAFT)
    assert d['stage'] == PREPARATION and tuple(d['roles']) == ROLES
    assert not any(d[name] for name in ('physical_substeps_authorized', 'learning_updates', 'heldout_used',
                                       'hardware_io', 'automatic_retry', 'automatic_extension'))
    for name, checksum in d['pinned'].items():
        assert sha(ROOT/name) == checksum, name
    source, score = prior.PLAN, prior.OUT/'analysis.json'
    assert read(score)['protocol_sha256'] == sha(source) and not read(score)['score']['passed_development_gate']
    cpu = read(prior.OUT/'turn-feedback-cpu.json')
    assert cpu['status'] == 'passed-restored-turn-feedback-cpu-readback'
    assert all(sha(ROOT/name) == checksum for name, checksum in cpu['source_sha256'].items())
    benchmark = read(prior.OUT/'trace-compression-benchmark.json')
    assert benchmark['status'] == 'passed-lossless-npz-benchmark'
    assert len([row for row in benchmark['results'] if row['level'] == d['trace_zip_level']]) == 2
    spec = read(source)
    p, s, seven = (spec[key] for key in ('environment', 'sequence', 'seven_command'))
    seed, n, total = (d[key] for key in ('development_seed', 'worlds', 'maximum_unique_controls'))
    assert seed != p['seed'] and p['num_envs'] == n and not spec['reserved_holdout']
    assert s['boundary_controls'][-1] == d['shared_controls']
    assert d['terminal_controls'] == 1000 and total == d['shared_controls']+2*d['terminal_controls']
    assert d['maximum_transitions'] == n*total and d['maximum_batched_physics_substeps'] == total*5
    assert d['maximum_gpu_memory_fraction'] == spec['draft']['gpu_memory_fraction_cap']
    assert d['maximum_warp_snapshot_bytes'] == spec['maximum_warp_snapshot_bytes']
    assert disk_usage(ROOT).free >= 12*1024**3
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{seed}_' in path.name for path in (ROOT/folder).iterdir())
    codes = np.random.default_rng(seed).permutation(np.repeat(np.arange(4), spec['dual_route']['requested_pose_counts']))
    pose_groups = {pose:np.flatnonzero(codes == i).tolist() for i, pose in enumerate(base.shared.POSES)}
    groups = balanced_groups(codes, seven['commands'], seed+70000)
    camera_source = next(name for name, ids in groups.items() if 0 in ids)
    if camera_source != 'turn-left':
        other = min(i for i in groups['turn-left'] if codes[i] == codes[0])
        groups[camera_source] = sorted(other if i == 0 else i for i in groups[camera_source])
        groups['turn-left'] = sorted(0 if i == other else i for i in groups['turn-left'])
    run_id = f'duck_switch_routed-turn-gyro_pair_s{seed}_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    record = ROOT/'records/evaluations'/run_id
    p.update(seed=seed, stage=STAGE, run_id=run_id, split=dict(pose_codes=codes.tolist(), groups=pose_groups))
    spec['dual_route']['seed'] = seed
    for policy in (*spec['policies'].values(), spec['common_fallback'], *spec['switch_policies'].values(),
                   *seven['policies'].values()):
        policy['seed'] = seed
    seven.update(groups=groups, minimum_by_command={name:ceil(.95*len(ids)) for name, ids in groups.items()},
                 video_command='turn-left')
    spec.update(status='frozen-before-dynamics', stage=STAGE, record=base.rel(record), page=base.rel(PAGE),
        maximum_unique_controls=total, maximum_physics_substeps=total*5,
        maximum_sampling_wall_seconds=d['maximum_sampling_wall_seconds'], turn_pair=d,
        per_branch_horizon=d['shared_controls']+d['terminal_controls'])
    spec['references'].update(d['pinned'])
    spec['references'][base.rel(DRAFT)] = sha(DRAFT)
    for name in ('run_routed_turn_feedback_pair.py', 'collect_routed_turn_feedback_pair.py',
                 'verify_routed_turn_feedback_pair.py', 'routed_seven_turn_feedback_controller.py'):
        path = Path(__file__).with_name(name).resolve()
        spec['source_sha256'][base.rel(path)] = sha(path)
    base.shared.assert_sources(spec)
    cfg, _, _ = original.prepare(p)
    assert cfg.env.scene.num_envs == n and cfg.env.episode_length_s == 32 and not cfg.env.auto_reset
    base.exact_scene(Scene(cfg.env.scene, device='cpu').compile())
    torch.set_num_threads(4)
    fixture = feedback_fixture(spec, 'cpu')
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    record.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record/'protocol.json')
    write(OUT/'freeze.json', dict(status='passed-cpu-freeze', protocol_sha256=sha(PLAN), physical_substeps=0,
        seed=seed, exact_cpu_scene=True, turn_fixture=fixture, camera_world=0, camera_original_assignment=camera_source))
    (record/'README.md').write_text('# 同站立状态转向反馈对照\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../continuous-routed-turn-feedback-pair-live.html)\n', encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='同站立状态的七指令转向反馈对照', run_id=run_id,
        controls=0, maximum_controls=total, transitions=0, updates=[], learning_updates=0))
    print('ROUTED_TURN_PAIR_FROZEN', base.rel(record), sha(PLAN), flush=True)


def snapshot(env, obs, memory, spec, target):
    with patch.object(original, 'switch_models', prior.ORIGINAL_SWITCH):
        actors, proof, fields, inputs = prior.ORIGINAL_SNAPSHOT(env, obs, memory, spec, target)
    check = feedback_fixture(spec, env.device)
    assert check == read(OUT/'freeze.json')['turn_fixture']
    proof['sequence_boundaries']['turn_input_restore'] = check
    assert env.common_step_counter == env._sim_step_counter == 0
    return actors, proof, fields, inputs


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze','zero','sample'))
    mode = parser.parse_args().mode
    if mode == 'freeze':
        return freeze()
    phase(PLAN)
    seed = read(PLAN)['environment']['seed']
    with patch.multiple(pair_runner, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=seed, phase=phase, collect=collect), \
         patch.multiple(route.prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=seed, phase=phase, snapshot=snapshot), \
         patch.object(base.shared, 'prepare', original.prepare), \
         patch.object(base.shared, 'actor_model', route.guarded_actor), \
         patch.object(original, 'switch_models', prior.route_models):
        (route.prior.zero if mode == 'zero' else pair_runner.sample)()


if __name__ == '__main__':
    main()
