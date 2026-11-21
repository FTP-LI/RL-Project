"""Run the reserved held-out comparison after both development batches pass."""
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import evaluate_getup_pose_router as development
from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_pose_router_holdout.json'
OUT = development.OUT/'heldout'
STAGE = 'S3-pose-routed-recovery-heldout'
SEED = 2062


def development_passes():
    master = read(development.PLAN)
    digest = sha(development.PLAN)
    paths = []
    for seed in development.SEEDS:
        path = development.OUT/f'seed{seed}-analysis.json'
        result = read(path)
        assert result['seed'] == seed and result['master_sha256'] == digest
        assert result['status'] == 'passed-independent-verification' and result['selection']['passes_gate']
        assert not result['heldout_used']
        paths.append(path)
    return master, paths


def heldout_spec(master):
    with patch.object(development, 'PLAN', PLAN), patch.object(development, 'STAGE', STAGE):
        spec = development.bound(master, SEED)
    return dict(spec, heldout_used=True, data_scope='reserved-heldout')


def freeze():
    master, passes = development_passes()
    assert not PLAN.exists() and not OUT.exists()
    assert not any(f'_s{SEED}_' in path.name for path in (ROOT/'records/evaluations').iterdir())
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    codes = np.concatenate(([1], np.random.default_rng(SEED).permutation(np.repeat(np.arange(4), [32, 31, 128, 128]))))
    master['source_sha256'][Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    for path in [development.PLAN, *passes]:
        master['references'][path.relative_to(ROOT).as_posix()] = sha(path)
    master.update(status='frozen-before-heldout-physical-steps', stage=STAGE, heldout_used=True,
                  batches={str(SEED): dict(run_id=f'duck_getup_pose-route_heldout_s{SEED}_{stamp}', pose_codes=codes.tolist())},
                  gate='One reserved heldout; timely and final original gates pass, combined successes no lower than either pure policy.')
    spec = heldout_spec(master)
    development.shared.assert_sources(spec)
    scene = development.cpu_scene_readback(spec['environment'], spec)
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN, master)
    write(OUT/'preflight.json', dict(status='passed-cpu-scene-zero-step', seed=SEED, protocol_sha256=sha(PLAN),
          scene=scene, physical_substeps=0, learning_updates=0, data_scope='reserved-heldout',
          heldout_physics_started=False))
    print('POSE_ROUTE_HELDOUT_FROZEN', sha(PLAN), spec['environment']['run_id'], flush=True)


def main():
    development.check_project()
    if sys.argv[1:] == ['--freeze']:
        return freeze()
    assert sys.argv[1:] in (['--sample'], ['--verify'])
    development_passes()
    pre = read(OUT/'preflight.json')
    assert pre['status'] == 'passed-cpu-scene-zero-step' and pre['protocol_sha256'] == sha(PLAN)
    spec = heldout_spec(read(PLAN))
    development.shared.assert_sources(spec)
    record = ROOT/'records/evaluations'/spec['environment']['run_id']
    result = OUT/f'seed{SEED}-analysis.json'

    def write_scope(path, value):
        if path in (record/'evaluation.json', result):
            value = dict(value, heldout_used=True, data_scope='reserved-heldout')
        write(path, value)

    with patch.object(development, 'PLAN', PLAN), patch.object(development, 'OUT', OUT), \
         patch.object(development, 'write', write_scope):
        if sys.argv[1:] == ['--sample']:
            try:
                development.sample(spec, record)
            finally:
                readme = record/'README.md'
                if readme.exists():
                    content = readme.read_text(encoding='utf-8').replace(
                        'getup_pose_router.json', 'getup_pose_router_holdout.json')
                    readme.write_text(content+'\n本批为独立留出，记录一次使用。\n', encoding='utf-8')
        else:
            assert not result.exists() and read(record/'evaluation.json')['heldout_used']
            development.verify(spec, record)
            assert read(result)['heldout_used']


if __name__ == '__main__':
    main()
