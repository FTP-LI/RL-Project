"""Run the one-time, original-mixture confirmation of the fixed getup route."""

from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_getup_right_robustness_development as study


base, route, ROOT, read, sha, write = study.base, study.route, study.ROOT, study.read, study.sha, study.write
DRAFT = ROOT / 'configs/experiments/getup_right_robustness_holdout_draft.json'
PLAN = DRAFT.with_stem('getup_right_robustness_holdout_s2065')
OUT = ROOT / 'records/reports/getup-right-robustness/holdout'
PAGE = ROOT / 'records/getup-right-robustness-holdout-live.html'
STAGE, SEED = 'S3-getup-right-robustness-holdout', 2065


def phase(protocol):
    study.require_stage(protocol, STAGE + '-preparation' if protocol == DRAFT else STAGE)


def freeze():
    from mjlab.scene import Scene

    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE)) and not torch.cuda.is_initialized()
    d = read(DRAFT)
    assert d['holdout_seed'] == SEED != d['excluded_seed']
    assert not any(d[key] for key in ('learning_updates', 'physical_substeps_authorized', 'heldout_used',
        'holdout_eligible_for_tuning', 'hardware_io', 'automatic_retry', 'automatic_extension'))
    for folder in ('records/evaluations', 'records/experiments', 'records/benchmarks',
                   'records/training-archive/evaluations', 'records/training-archive/experiments'):
        assert not any(f'_s{SEED}_' in path.name for path in (ROOT / folder).iterdir())
    source, analysis, decision, release, mix, scale = (study.pinned(d[key]) for key in ('source_protocol',
        'development_analysis', 'development_decision', 'release_verification', 'original_mix_source', 'resource_selection'))
    spec, old, score, result, coverage, resource = map(read, (source, mix, analysis, decision, release, scale))
    assert score['protocol_sha256'] == sha(source) and score['seed'] == study.SEED
    assert result['analysis_sha256'] == coverage['analysis_sha256'] == sha(analysis)
    assert result['admitted_to_independent_development'] and score['arms']['candidate']['combined']['functional_coverage_gate']
    assert not any(item['heldout_used'] for item in (spec, score, result, coverage, resource))
    assert coverage['counts']['right-side'] == score['arms']['candidate']['combined']['actual']['right-side']['episodes']
    assert coverage['counts']['right-side'] >= spec['right_robustness']['minimum_actual_right_development_coverage']
    p, routing = spec['environment'], spec['dual_route']
    worlds = d['worlds']
    assert worlds == p['num_envs'] == resource['selected_worlds']
    assert d['episode_seconds'] == p['episode_seconds'] == 11
    assert d['required_gate'] == routing['gate'] == old['dual_route']['gate']
    assert d['minimum_each_other_actual_pose_coverage'] == p['actual_release_minimum']
    assert d['maximum_gpu_memory_fraction'] == spec['draft']['gpu_memory_fraction_cap']
    assert d['maximum_physics_substeps'] == spec['maximum_physics_substeps']
    assert d['maximum_transitions'] == spec['maximum_unique_controls'] * worlds
    assert tuple(d['requested_pose_order']) == base.shared.POSES
    counts = d['requested_pose_counts']
    previous = old['dual_route']['requested_pose_counts']
    assert sum(counts) == worlds and all(a * sum(previous) == b * worlds for a, b in zip(counts, previous, strict=True))
    for role in route.ROLES:
        assert spec['policies'][role]['model_sha256'] == old['policies'][role]['model_sha256']
    assert spec['common_fallback']['model_sha256'] == old['common_fallback']['model_sha256']
    study.check_storage(spec['right_robustness'])
    codes = np.random.default_rng(SEED).permutation(np.repeat(np.arange(len(counts)), counts))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(base.shared.POSES)}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_right-robustness_holdout_s{SEED}_{stamp}'
    record = ROOT / 'records/evaluations' / run_id
    p.update(seed=SEED, stage=STAGE, run_id=run_id, split=dict(pose_codes=codes.tolist(), groups=groups))
    routing.update(seed=SEED, requested_pose_counts=counts, reserved_holdout_seed=SEED)
    # The common release guard reads this field before any policy-controlled step.
    spec['right_robustness']['minimum_actual_right_development_coverage'] = d['minimum_actual_right_coverage']
    for policy in (*spec['policies'].values(), spec['common_fallback']):
        policy['seed'] = SEED
    spec.update(stage=STAGE, record=base.rel(record), page=base.rel(PAGE), data_scope='reserved-holdout',
        reserved_holdout=True, holdout_admission=d, cameras={role: {'right-side': groups['right-side'][0]} for role in route.ROLES})
    cpu_file, cuda_file = study.OUT / 'freeze.json', study.OUT / 'independent-zero/proof.json'
    cpu, cuda = read(cpu_file), read(cuda_file)
    assert cpu['protocol_sha256'] == cuda['protocol_sha256'] == sha(source)
    assert cuda['physical_substeps'] == 0 and cuda['snapshot_bytes'] <= spec['maximum_warp_snapshot_bytes']
    refs = (DRAFT, source, analysis, decision, release, mix, scale, cpu_file, cuda_file)
    spec['references'].update({base.rel(path): sha(path) for path in refs})
    code = Path(__file__).resolve()
    spec['source_sha256'][base.rel(code)] = sha(code)
    base.shared.assert_sources(spec)
    cfg, _, _ = base.shared.prepare(p)
    assert cfg.env.scene.num_envs == worlds and cfg.env.episode_length_s == d['episode_seconds']
    base.exact_scene(Scene(cfg.env.scene, device='cpu').compile())
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    record.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(cpu, protocol_sha256=sha(PLAN), data_scope='reserved-holdout',
        source_cpu_sha256=sha(cpu_file), cpu_scene_verified=True, seed=SEED, heldout_physics_started=False))
    (record / 'README.md').write_text('# 右侧起身独立留出\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../getup-right-robustness-holdout-live.html)\n',
        encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='四姿态起身独立留出', run_id=run_id,
        controls=0, maximum_controls=spec['maximum_unique_controls'], transitions=0, updates=[], learning_updates=0))
    print('RIGHT_ROBUSTNESS_HOLDOUT_FROZEN', base.rel(record), sha(PLAN), flush=True)


def verify(record, spec, receipt):
    used = read(receipt)
    assert used['heldout_used'] and used['status'] == 'completed' and used['protocol_sha256'] == sha(PLAN)
    proof = read(record / 'release-coverage.json')
    assert proof['status'] == 'passed' and proof['post_release_controls'] == 0
    assert proof['counts']['right-side'] >= spec['holdout_admission']['minimum_actual_right_coverage']
    assert proof['poses_sha256'] == sha(record / 'release-coverage.npz')
    assert not (OUT / 'decision.json').exists()
    decisions = []

    def scoped(path, result):
        result = dict(result, heldout_used=True, data_scope='reserved-holdout', holdout_eligible_for_tuning=False)
        if path.name == 'decision.json':
            result['final_holdout_passed'] = result.pop('admitted_to_independent_development')
            result['status'] = 'completed-reserved-holdout'
            decisions.append(result)
        else:
            write(path, result)

    with patch.multiple(route, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, write=scoped):
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
        heldout_used=True, holdout_eligible_for_tuning=False, learning_updates=0))
    decision, = decisions
    write(OUT / 'decision.json', decision)
    print('RIGHT_ROBUSTNESS_FINAL_HOLDOUT', decision['final_holdout_passed'], flush=True)


def run(mode):
    phase(PLAN)
    spec = read(PLAN)
    assert spec['reserved_holdout'] and spec['data_scope'] == 'reserved-holdout'
    study.check_storage(spec['right_robustness'])
    record = ROOT / spec['record']
    receipt = OUT / 'consumption.json'
    if mode == 'verify':
        return verify(record, spec, receipt)
    scope = dict(protocol_sha256=sha(PLAN), data_scope='reserved-holdout', seed=SEED, holdout_eligible_for_tuning=False)
    if mode == 'sample':
        assert not (record / 'evaluation.json').exists()
        assert read(OUT / 'independent-zero/proof.json')['protocol_sha256'] == sha(PLAN)
        assert read(record / 'windows-observer-ready.json')['protocol_sha256'] == sha(PLAN)
        # Exclusive claim precedes reset or any physics and survives a process crash.
        with receipt.open('x', encoding='utf-8') as file:
            json.dump(dict(scope, status='claimed-before-physics', heldout_used=None), file, indent=2)
    try:
        with patch.multiple(route.prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase, snapshot=route.snapshot), \
             patch.object(base.acquisition, 'collect', study.collect), patch.object(base.shared, 'actor_model', route.guarded_actor):
            getattr(route.prior, mode)()
    finally:
        if mode == 'sample' and (record / 'evaluation.json').exists():
            status = read(record / 'evaluation.json')
            write(receipt, dict(scope, status=status['status'], heldout_used=status['actual']['attempted_substeps'] > 0,
                actual=status['actual']))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample', 'verify'))
    mode = parser.parse_args().mode
    (freeze if mode == 'freeze' else lambda: run(mode))()
