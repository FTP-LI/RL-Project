"""Repeat the admitted dual-pose route with a fresh development seed."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import evaluate_getup_dual_pose_route as first
from run_getup_latched_head_corrective_ppo import require_stage
from run_getup_teacher_head_evaluation import pinned


base, ROOT, read, sha, write = first.base, first.ROOT, first.read, first.sha, first.write
DRAFT = ROOT / 'configs/experiments/getup_dual_pose_route_replication_draft.json'
PLAN = DRAFT.with_stem('getup_dual_pose_route_s1167')
OUT = ROOT / 'records/reports/getup-dual-pose-route/replication'
PAGE = ROOT / 'records/getup-dual-pose-route-s1167-live.html'
STAGE, SEED = 'S3-getup-dual-pose-route-replication', 1167


def phase(protocol):
    require_stage(protocol, STAGE + '-preparation' if protocol == DRAFT else STAGE)


def freeze():
    phase(DRAFT)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE))
    d = read(DRAFT)
    assert d['development_seed'] == SEED != d['reserved_holdout_seed']
    assert not any(d[key] for key in ('learning_updates', 'physical_substeps_authorized', 'heldout_used', 'hardware_io'))
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{SEED}_' in path.name for path in (ROOT / folder).iterdir())
    source, analysis, decision = (pinned(d[key]) for key in ('first_protocol', 'first_analysis', 'first_decision'))
    score, selected = read(analysis), read(decision)
    assert score['protocol_sha256'] == sha(source) and selected['analysis_sha256'] == sha(analysis)
    assert selected['admitted_to_independent_development'] and score['arms']['candidate']['combined']['functional_coverage_gate']
    assert not selected['heldout_used'] and score['seed'] == first.SEED
    old_cpu, old_cuda = first.OUT / 'freeze.json', first.OUT / 'independent-zero/proof.json'
    proof = read(old_cpu)
    assert proof['status'] == 'passed-cpu-zero' and proof['protocol_sha256'] == sha(source)
    assert read(old_cuda)['protocol_sha256'] == sha(source)
    spec = read(source)
    p, route = spec['environment'], spec['dual_route']
    assert route['reserved_holdout_seed'] == d['reserved_holdout_seed']
    route['seed'] = SEED
    codes = np.random.default_rng(SEED).permutation(np.repeat(np.arange(4), route['requested_pose_counts']))
    groups = {pose: np.flatnonzero(codes == k).tolist() for k, pose in enumerate(base.shared.POSES)}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_dual-pose-route_development_s{SEED}_{stamp}'
    record = ROOT / 'records/evaluations' / run_id
    p.update(seed=SEED, run_id=run_id, stage=STAGE, split=dict(pose_codes=codes.tolist(), groups=groups))
    for policy in (*spec['policies'].values(), spec['common_fallback']):
        policy['seed'] = SEED
    spec.update(stage=STAGE, record=base.rel(record), page=base.rel(PAGE), replication=d,
                cameras={role: {'right-side': groups['right-side'][0]} for role in first.ROLES})
    spec['references'].update({base.rel(path): sha(path) for path in (DRAFT, source, analysis, decision, old_cpu, old_cuda)})
    code = Path(__file__).resolve()
    spec['source_sha256'][base.rel(code)] = sha(code)
    base.shared.assert_sources(spec)
    OUT.mkdir()
    record.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(proof, protocol_sha256=sha(PLAN), same_cpu_scene_source_sha256=sha(old_cpu),
                                    source_protocol_sha256=sha(source)))
    (record / 'README.md').write_text('# 双姿态起身独立复验\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../getup-dual-pose-route-s1167-live.html) · [日志](execution.log)\n',
        encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='双姿态起身独立复验', run_id=run_id,
        controls=0, maximum_controls=spec['maximum_unique_controls'], transitions=0, updates=[], learning_updates=0))
    print('DUAL_POSE_ROUTE_REPLICATION_FROZEN', base.rel(record), sha(PLAN), flush=True)


def run(mode):
    if mode == 'verify':
        with patch.multiple(first, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED):
            return first.verify()
    with patch.multiple(first.prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase, snapshot=first.snapshot), \
         patch.object(base.acquisition, 'collect', first.collect), patch.object(base.shared, 'actor_model', first.guarded_actor):
        return getattr(first.prior, mode)()


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample', 'verify'))
    mode = parser.parse_args().mode
    (freeze if mode == 'freeze' else lambda: run(mode))()
