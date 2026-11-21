"""Confirm the frozen dual-pose route once on the reserved seed."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import evaluate_getup_dual_pose_route as first
import replicate_getup_dual_pose_route as replication
from run_getup_latched_head_corrective_ppo import require_stage
from run_getup_teacher_head_evaluation import pinned


base, ROOT, read, sha, write = first.base, first.ROOT, first.read, first.sha, first.write
DRAFT = ROOT / 'configs/experiments/getup_dual_pose_route_holdout_draft.json'
PLAN = DRAFT.with_stem('getup_dual_pose_route_holdout_s2064')
OUT = ROOT / 'records/reports/getup-dual-pose-route/holdout'
PAGE = ROOT / 'records/getup-dual-pose-route-holdout-live.html'
STAGE, SEED = 'S3-getup-dual-pose-route-holdout', 2064


def phase(protocol):
    require_stage(protocol, STAGE + '-preparation' if protocol == DRAFT else STAGE)


def freeze():
    phase(DRAFT)
    d = read(DRAFT)
    assert d['holdout_seed'] == SEED and not d['heldout_used']
    assert not any(path.exists() for path in (PLAN, OUT, PAGE))
    for folder in ('records/evaluations', 'records/experiments', 'records/training-archive/evaluations'):
        assert not any(f'_s{SEED}_' in path.name for path in (ROOT / folder).iterdir())
    refs = [DRAFT]
    for source, batch in zip((first, replication), d['development'], strict=True):
        protocol, score_path, decision_path = (pinned(batch[key]) for key in ('protocol', 'analysis', 'decision'))
        assert source.PLAN == protocol
        score, decision = read(score_path), read(decision_path)
        assert score['seed'] == source.SEED and score['protocol_sha256'] == sha(protocol)
        assert decision['analysis_sha256'] == sha(score_path) and not decision['heldout_used']
        assert decision['admitted_to_independent_development'] and score['arms']['candidate']['combined']['functional_coverage_gate']
        refs.extend((protocol, score_path, decision_path))
    spec = read(replication.PLAN)
    prior = read(first.PLAN)
    for role in first.ROLES:
        assert spec['policies'][role]['model_sha256'] == prior['policies'][role]['model_sha256']
    assert spec['common_fallback']['model_sha256'] == prior['common_fallback']['model_sha256']
    assert spec['dual_route']['gate'] == prior['dual_route']['gate']
    p, route = spec['environment'], spec['dual_route']
    assert route['reserved_holdout_seed'] == SEED and not spec['heldout_used']
    route['seed'] = SEED
    codes = np.random.default_rng(SEED).permutation(np.repeat(np.arange(4), route['requested_pose_counts']))
    groups = {pose: np.flatnonzero(codes == k).tolist() for k, pose in enumerate(base.shared.POSES)}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_dual-pose-route_holdout_s{SEED}_{stamp}'
    record = ROOT / 'records/evaluations' / run_id
    p.update(seed=SEED, run_id=run_id, stage=STAGE, split=dict(pose_codes=codes.tolist(), groups=groups))
    for policy in (*spec['policies'].values(), spec['common_fallback']):
        policy['seed'] = SEED
    spec.update(stage=STAGE, record=base.rel(record), page=base.rel(PAGE), data_scope='reserved-holdout',
        reserved_holdout=True, holdout_admission=d, cameras={role: {'right-side': groups['right-side'][0]} for role in first.ROLES})
    old_cpu, old_cuda = replication.OUT / 'freeze.json', replication.OUT / 'independent-zero/proof.json'
    cpu, cuda = read(old_cpu), read(old_cuda)
    assert cpu['protocol_sha256'] == cuda['protocol_sha256'] == sha(replication.PLAN)
    refs.extend((old_cpu, old_cuda))
    spec['references'].update({base.rel(path): sha(path) for path in refs})
    path = Path(__file__).resolve()
    spec['source_sha256'][base.rel(path)] = sha(path)
    base.shared.assert_sources(spec)
    OUT.mkdir()
    record.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(cpu, protocol_sha256=sha(PLAN), data_scope='reserved-holdout',
        source_cpu_sha256=sha(old_cpu), heldout_physics_started=False))
    (record / 'README.md').write_text('# 双姿态起身最终留出\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../getup-dual-pose-route-holdout-live.html) · [日志](execution.log)\n',
        encoding='utf-8')
    base.publish(record, PAGE, dict(status='starting', purpose='双姿态起身最终留出', run_id=run_id,
        controls=0, maximum_controls=spec['maximum_unique_controls'], transitions=0, updates=[], learning_updates=0))
    print('DUAL_POSE_ROUTE_HOLDOUT_FROZEN', base.rel(record), sha(PLAN), flush=True)


def run(mode):
    phase(PLAN)
    spec = read(PLAN)
    assert spec['reserved_holdout'] and spec['data_scope'] == 'reserved-holdout'
    record = ROOT / spec['record']
    receipt = OUT / 'consumption.json'
    if mode == 'verify':
        used = read(receipt)
        assert used['heldout_used'] and used['status'] == 'completed' and used['protocol_sha256'] == sha(PLAN)

        def scoped(path, result):
            result = dict(result, heldout_used=True, data_scope='reserved-holdout')
            if path.name == 'decision.json':
                result['final_holdout_passed'] = result.pop('admitted_to_independent_development')
                result['status'] = 'completed-reserved-holdout'
                result['holdout_eligible_for_tuning'] = False
            write(path, result)

        with patch.multiple(first, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, write=scoped):
            first.verify()
        print('DUAL_POSE_FINAL_HOLDOUT', read(OUT / 'decision.json')['final_holdout_passed'], flush=True)
        return
    if mode == 'sample':
        assert not receipt.exists() and not (record / 'evaluation.json').exists()
    try:
        with patch.multiple(first.prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=SEED, phase=phase, snapshot=first.snapshot), \
             patch.object(base.acquisition, 'collect', first.collect), patch.object(base.shared, 'actor_model', first.guarded_actor):
            getattr(first.prior, mode)()
    finally:
        if mode == 'sample' and (record / 'evaluation.json').exists():
            status = read(record / 'evaluation.json')
            write(receipt, dict(status=status['status'], heldout_used=status['actual']['attempted_substeps'] > 0,
                protocol_sha256=sha(PLAN), data_scope='reserved-holdout', seed=SEED, actual=status['actual'],
                holdout_eligible_for_tuning=False))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample', 'verify'))
    mode = parser.parse_args().mode
    (freeze if mode == 'freeze' else lambda: run(mode))()
