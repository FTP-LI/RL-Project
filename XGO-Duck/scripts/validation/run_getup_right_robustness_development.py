"""Evaluate fixed getup routes on one fresh, right-side-enriched release batch."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from shutil import copyfile, disk_usage
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import evaluate_getup_dual_pose_route as route
import replicate_getup_dual_pose_route as replication
from run_getup_latched_head_corrective_ppo import require_stage
from run_getup_teacher_head_evaluation import pinned
from standup_metrics import classify_release, projected_gravity


base, ROOT, read, sha, write = route.base, route.ROOT, route.read, route.sha, route.write
DRAFT = ROOT / 'configs/experiments/getup_right_robustness_development_draft.json'
PLAN = DRAFT.with_stem('getup_right_robustness_development_s1168')
OUT = ROOT / 'records/reports/getup-right-robustness/development'
PAGE = ROOT / 'records/getup-right-robustness-development-live.html'
STAGE, SEED = 'S3-getup-right-robustness-development', 1168
SELECTION = ('records/reports/getup-right-robustness/scale/selection.json',
             '49c96ca240fa0b3417f2eb0208f4537dbc6b983c3c7a8a9a9c0e231b4e166475')
RESOURCE = ROOT / 'records/reports/getup-right-robustness/scale/n3072-snapshot-recovery/protocol.json'


def phase(protocol):
    require_stage(protocol, STAGE + '-preparation' if protocol == DRAFT else STAGE)


def check_storage(draft):
    assert disk_usage(ROOT).free >= draft['resource_probe']['minimum_free_storage_gib'] * 1024 ** 3


def freeze():
    from mjlab.scene import Scene

    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE)) and not torch.cuda.is_initialized()
    d = read(DRAFT)
    check_storage(d)
    assert d['proposed_development_seed'] == SEED and SEED not in d['excluded_for_tuning_and_training']
    assert d['reserved_holdout_seed'] is None
    assert not any(d[key] for key in ('learning_updates', 'physical_substeps_authorized', 'heldout_used',
                                      'hardware_io', 'automatic_retry', 'automatic_extension'))
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{SEED}_' in path.name for path in (ROOT / folder).iterdir())
    source, selection = pinned(d['source_protocol']), pinned(SELECTION)
    evidence = [pinned(item) for item in d['development_evidence']]
    selected = read(selection)
    worlds = selected['selected_worlds']
    resource, = (trial for trial in selected['trials'] if trial['worlds'] == worlds)
    resource_result = pinned((resource['result'], resource['result_sha256']))
    assert worlds == 3072 and selected['status'] == 'passed-two-resource-probes'
    assert resource['gpu_peak_mib'] / resource['gpu_total_mib'] < d['maximum_gpu_memory_fraction']
    resource_protocol = resource_result.with_name('protocol.json')
    assert resource_protocol == RESOURCE
    resource_zero = resource_result.parent / 'independent-zero/proof.json'
    assert read(resource_zero)['protocol_sha256'] == sha(resource_protocol)
    assert read(resource_zero)['physical_substeps'] == 0
    option, = (item for item in d['scale_options'] if item['worlds'] == worlds)
    spec = read(source)
    p, routing = spec['environment'], spec['dual_route']
    gates = dict(d['required_original_gates'])
    gates['minimum_actual_pose_count'] = gates.pop('minimum_each_actual_pose_count')
    assert gates == routing['gate']
    assert gates['minimum_actual_pose_count'] == p['actual_release_minimum']
    assert d['recorded_seconds'] == p['episode_seconds'] == p['criteria']['max_seconds'] == 11.
    assert d['control_hz'] * d['recorded_seconds'] == p['maximum_control_steps']
    assert d['maximum_physics_substeps'] == spec['maximum_physics_substeps']
    assert worlds * spec['maximum_unique_controls'] == option['maximum_transitions']
    assert tuple(d['requested_pose_order']) == base.shared.POSES
    counts = option['requested_pose_counts']
    codes = np.random.default_rng(SEED).permutation(np.repeat(np.arange(len(counts)), counts))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(base.shared.POSES)}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_right-robustness_development_s{SEED}_{stamp}'
    record = ROOT / 'records/evaluations' / run_id
    p.update(num_envs=worlds, seed=SEED, stage=STAGE, run_id=run_id,
             maximum_control_transitions=worlds * p['maximum_control_steps'],
             split=dict(pose_codes=codes.tolist(), groups=groups))
    routing.update(seed=SEED, worlds=worlds, requested_pose_counts=counts, reserved_holdout_seed=None)
    for policy in (*spec['policies'].values(), spec['common_fallback']):
        policy['seed'] = SEED
    spec.update(stage=STAGE, record=base.rel(record), page=base.rel(PAGE), right_robustness=d,
                maximum_warp_snapshot_bytes=read(resource_protocol)['maximum_warp_snapshot_bytes'],
                cameras={role: {'right-side': groups['right-side'][0]} for role in route.ROLES})
    spec['draft']['gpu_memory_fraction_cap'] = d['maximum_gpu_memory_fraction']
    old_cpu = replication.OUT / 'freeze.json'
    proof = read(old_cpu)
    assert proof['protocol_sha256'] == sha(source)
    refs = (DRAFT, source, old_cpu, selection, resource_result, resource_protocol, resource_zero, *evidence)
    spec['references'].update({base.rel(path): sha(path) for path in refs})
    source_code = Path(__file__).resolve()
    spec['source_sha256'][base.rel(source_code)] = sha(source_code)
    base.shared.assert_sources(spec)
    cfg, _, _ = base.shared.prepare(p)
    assert cfg.env.scene.num_envs == worlds and cfg.env.episode_length_s == 11.
    base.exact_scene(Scene(cfg.env.scene, device='cpu').compile())
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    record.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(proof, protocol_sha256=sha(PLAN), source_protocol_sha256=sha(source),
        scale_selection_sha256=sha(selection), cpu_scene_verified=True, seed=SEED, worlds=worlds))
    (record / 'README.md').write_text('# 右侧起身开发评估\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../getup-right-robustness-development-live.html)\n',
        encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='右侧起身开发评估', run_id=run_id,
        controls=0, maximum_controls=spec['maximum_unique_controls'], transitions=0, updates=[], learning_updates=0))
    print('RIGHT_ROBUSTNESS_DEVELOPMENT_FROZEN', base.rel(record), sha(PLAN), flush=True)


def collect(env, actors, memory, meta, spec, record, state, publish, check_active):
    original = base.acquisition.SceneReleaseSnapshot
    minimum = spec['right_robustness']['minimum_actual_right_development_coverage']
    release = spec['prefix_controls']

    class CoverageSnapshot(original):
        def __init__(self, env, memory, cap):
            # Apply the scorer's physical release classifier before any active
            # policy step. Choose one world without seeing any branch outcome.
            assert env.common_step_counter == release and env._sim_step_counter == release * 5
            gravity = projected_gravity(base.shared.cpu(env.scene['robot'].data.root_link_quat_w))
            poses = np.asarray(classify_release(gravity, spec['environment']['criteria']['release_alignment_min']))
            counts = {pose: int((poses == pose).sum()) for pose in (*base.shared.POSES, 'other')}
            right = np.flatnonzero(poses == 'right-side')
            chosen = int(right[0]) if right.size else None
            passed = len(right) >= minimum and all(
                counts[pose] >= spec['environment']['actual_release_minimum'] for pose in base.shared.POSES)
            np.savez_compressed(record / 'release-coverage.npz', poses=poses)
            proof = dict(status='passed' if passed else 'stopped-insufficient-release-coverage', counts=counts,
                world_for_actual_right_replay=chosen, controls=release, physical_substeps=release * 5,
                post_release_controls=0, minimum_right=minimum, learning_updates=0,
                poses_sha256=sha(record / 'release-coverage.npz'))
            write(record / 'release-coverage.json', proof)
            state['release_coverage'] = proof
            publish()
            print('RIGHT_ROBUSTNESS_RELEASE', counts, 'representative', chosen, flush=True)
            assert passed, 'Actual release coverage did not meet the fixed minimum'
            super().__init__(env, memory, cap)

    with patch.object(base.acquisition, 'SceneReleaseSnapshot', CoverageSnapshot):
        route.collect(env, actors, memory, meta, spec, record, state, publish, check_active)


def run(mode):
    phase(PLAN)
    spec = read(PLAN)
    check_storage(spec['right_robustness'])
    record = ROOT / spec['record']
    if mode == 'verify':
        proof = read(record / 'release-coverage.json')
        assert proof['status'] == 'passed' and proof['post_release_controls'] == 0
        assert proof['poses_sha256'] == sha(record / 'release-coverage.npz')
        with patch.multiple(route, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED):
            route.verify()
        with np.load(record / 'release-coverage.npz', allow_pickle=False) as arrays:
            poses = arrays['poses']
        scores = read(OUT / 'analysis.json')
        for arm in scores['arms'].values():
            np.testing.assert_array_equal([row['release_pose'] for row in arm['timely']['episodes']], poses)
        assert proof['world_for_actual_right_replay'] == int(np.flatnonzero(poses == 'right-side')[0])
        write(OUT / 'release-verification.json', dict(status='passed-independent-release-match',
            counts=proof['counts'], selected_world=proof['world_for_actual_right_replay'],
            release_sha256=sha(record / 'release-coverage.json'), analysis_sha256=sha(OUT / 'analysis.json'),
            heldout_used=False, learning_updates=0))
        return
    with patch.multiple(route.prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase, snapshot=route.snapshot), \
         patch.object(base.acquisition, 'collect', collect), patch.object(base.shared, 'actor_model', route.guarded_actor):
        getattr(route.prior, mode)()


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample', 'verify'))
    args = parser.parse_args()
    (freeze if args.mode == 'freeze' else lambda: run(args.mode))()
