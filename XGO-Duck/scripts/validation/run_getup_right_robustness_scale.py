"""Measure two GPU scales with the unchanged full-trace collector and short clips."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from shutil import copyfile, disk_usage
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import evaluate_getup_dual_pose_route as route
from run_getup_latched_head_corrective_ppo import require_stage
from run_getup_teacher_head_evaluation import pinned


base, ROOT, read, sha, write = route.base, route.ROOT, route.read, route.sha, route.write
DRAFT = ROOT / 'configs/experiments/getup_right_robustness_development_draft.json'
PLAN = DRAFT.with_stem('getup_right_robustness_scale')
OUT = ROOT / 'records/reports/getup-right-robustness/scale'
STAGE = 'S3-getup-right-robustness-scale'
SOURCE_PREPARE = base.shared.prepare
MIB = 1024 ** 2


def prepare(environment):
    cfg, display, meta = SOURCE_PREPARE(environment)
    seconds = environment['episode_seconds']
    # The inherited evaluator fixes an 11 s horizon. Only the benchmark horizon is shortened.
    cfg.env.episode_length_s = seconds
    cfg.env.observations['critic'].terms['remaining_time'].params['horizon_seconds'] = seconds
    return cfg, display, meta


def freeze():
    from mjlab.scene import Scene

    require_stage(DRAFT, 'S3-getup-right-robustness-development-preparation')
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    d = read(DRAFT)
    probe = d['resource_probe']
    parent = pinned(d['source_protocol'])
    for reference in d['development_evidence']:
        pinned(reference)
    spec = read(parent)
    assert tuple(d['requested_pose_order']) == base.shared.POSES
    assert tuple(d['branches']) == tuple(spec['branches']) == route.ROLES
    assert spec['prefix_controls'] == d['release_counter'] and not spec['heldout_used']
    seed = probe['seed']
    assert seed != d['proposed_development_seed'] and seed not in d['excluded_for_tuning_and_training']
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{seed}_' in path.name for path in (ROOT / folder).iterdir())
    n_choices = {choice['worlds']: choice for choice in d['scale_options']}
    assert set(probe['order']) == set(n_choices)
    controls = d['release_counter'] + len(route.ROLES) * (probe['episode_controls'] - d['release_counter'])
    assert controls * 5 == probe['maximum_physics_substeps_per_scale']
    assert disk_usage(ROOT).free >= probe['minimum_free_storage_gib'] * 1024 ** 3
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    OUT.mkdir(parents=True)
    trials = {}
    for worlds in probe['order']:
        trial = deepcopy(spec)
        settings = trial['environment']
        counts = n_choices[worlds]['requested_pose_counts']
        assert sum(counts) == worlds
        codes = np.random.default_rng(seed).permutation(np.repeat(np.arange(len(counts)), counts))
        groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(base.shared.POSES)}
        run_id = f'duck_getup_right-robustness_scale-{worlds}_s{seed}_{stamp}'
        target = OUT / f'n{worlds}'
        record = ROOT / 'records/benchmarks' / run_id
        page = ROOT / f'records/getup-right-robustness-scale-{worlds}-live.html'
        settings.update(seed=seed, num_envs=worlds, stage=STAGE, run_id=run_id,
                        split=dict(pose_codes=codes.tolist(), groups=groups),
                        episode_seconds=probe['episode_controls'] / d['control_hz'],
                        maximum_control_steps=probe['episode_controls'],
                        maximum_control_transitions=worlds * probe['episode_controls'])
        trial['dual_route'].update(seed=seed, worlds=worlds, requested_pose_counts=counts,
                                   reserved_holdout_seed=None)
        for policy in (*trial['policies'].values(), trial['common_fallback']):
            policy['seed'] = seed
        trial['draft']['gpu_memory_fraction_cap'] = d['maximum_gpu_memory_fraction']
        trial['references'].update({base.rel(path): sha(path) for path in (DRAFT, parent)})
        trial['source_sha256'][base.rel(Path(__file__).resolve())] = sha(Path(__file__))
        trial.update(stage=STAGE, record=base.rel(record), page=base.rel(page), resource_probe=probe,
                     maximum_unique_controls=controls,
                     maximum_physics_substeps=probe['maximum_physics_substeps_per_scale'],
                     maximum_warp_snapshot_bytes=probe['maximum_snapshot_bytes_per_scale'],
                     maximum_sampling_wall_seconds=probe['maximum_wall_seconds_per_scale'],
                     cameras={role: {'right-side': groups['right-side'][0]} for role in route.ROLES})
        base.shared.assert_sources(trial)
        cfg, _, _ = prepare(settings)
        assert cfg.env.scene.num_envs == worlds and cfg.env.episode_length_s == settings['episode_seconds']
        base.exact_scene(Scene(cfg.env.scene, device='cpu').compile())
        target.mkdir()
        record.mkdir()
        assert not page.exists()
        protocol = target / 'protocol.json'
        write(protocol, trial)
        copyfile(protocol, record / 'protocol.json')
        cpu = read(route.OUT / 'replication/freeze.json')
        assert cpu['protocol_sha256'] == sha(parent)
        write(target / 'freeze.json', dict(cpu, protocol_sha256=sha(protocol), cpu_scene_verified=True,
            cpu_only=True, source_protocol_sha256=sha(parent), functional_scoring=False))
        base.publish(record, page, dict(status='starting', purpose=f'起身并行规模短测 · {worlds} 环境',
            run_id=run_id, controls=0, maximum_controls=controls, transitions=0, updates=[], learning_updates=0))
        trials[str(worlds)] = dict(protocol=base.rel(protocol), sha256=sha(protocol))
    assert not torch.cuda.is_initialized()
    write(PLAN, dict(stage=STAGE, status='frozen-resource-probe', seed=seed, order=probe['order'], trials=trials,
                     draft_sha256=sha(DRAFT), heldout_used=False, functional_scoring=False))
    print('GETUP_RIGHT_SCALE_FROZEN', sha(PLAN), flush=True)


def run(mode, worlds):
    def phase(protocol):
        require_stage(PLAN, STAGE)
        assert sha(DRAFT) == suite['draft_sha256']
        assert sha(protocol) == trial['sha256']

    suite = read(PLAN)
    trial = suite['trials'][str(worlds)]
    protocol = ROOT / trial['protocol']
    phase(protocol)
    spec = read(protocol)
    record, page = ROOT / spec['record'], ROOT / spec['page']
    assert disk_usage(ROOT).free >= spec['resource_probe']['minimum_free_storage_gib'] * 1024 ** 3
    with patch.multiple(route.prior, PLAN=protocol, OUT=protocol.parent, PAGE=page, SEED=suite['seed'],
                        phase=phase, snapshot=route.snapshot), \
         patch.object(base.shared, 'prepare', prepare), \
         patch.object(base.shared, 'actor_model', route.guarded_actor), \
         patch.object(base.acquisition, 'collect', route.collect):
        getattr(route.prior, mode)()
    if mode == 'sample':
        evaluation = read(record / 'evaluation.json')
        telemetry = read(record / 'gpu-telemetry.json')
        trace_bytes = sum(path.stat().st_size for path in record.glob('trace-*.npz'))
        timings = getrusage(RUSAGE_SELF)
        rss_mib = timings.ru_maxrss / 1024
        assert evaluation['status'] == 'completed' and evaluation['actor_weights_unchanged']
        assert evaluation['actual_control_transitions'] == worlds * spec['maximum_unique_controls']
        assert trace_bytes > 0 and len(evaluation['conditions']) == len(route.ROLES)
        assert rss_mib <= spec['resource_probe']['maximum_rss_mib_per_scale']
        assert all(sha(record / item['video']['path']) == item['video']['sha256']
                   for item in evaluation['conditions'])
        wall = evaluation['sampling_wall_seconds']
        write(protocol.parent / 'result.json', dict(status='completed-resource-probe', worlds=worlds,
            controls=evaluation['actual']['controls'], substeps=evaluation['actual']['substeps'],
            transitions=evaluation['actual_control_transitions'], seconds_including_trace_write=wall,
            transitions_per_second=evaluation['actual_control_transitions'] / wall,
            gpu_peak_mib=evaluation['gpu']['peak_mib'], gpu_activity_mean=sum(row['gpu_percent'] for row in telemetry) / len(telemetry),
            ram_peak_mib=rss_mib, cpu_user_seconds=timings.ru_utime, cpu_system_seconds=timings.ru_stime,
            trace_mib=trace_bytes / MIB, evaluation_sha256=sha(record / 'evaluation.json'),
            learning_updates=0, functional_scoring=False, heldout_used=False))
        print('GETUP_RIGHT_SCALE_COMPLETED', worlds, flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample'))
    parser.add_argument('worlds', type=int, nargs='?', choices=(2048, 3072))
    args = parser.parse_args()
    if args.mode == 'freeze':
        assert args.worlds is None
        freeze()
    else:
        assert args.worlds is not None
        run(args.mode, args.worlds)
