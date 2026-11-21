"""Resume the 3072-world sizing study with its measured snapshot allowance."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_getup_right_robustness_scale as scale


base, ROOT, read, sha, write = scale.base, scale.ROOT, scale.read, scale.sha, scale.write
PLAN = scale.PLAN.with_stem('getup_right_robustness_scale_recovery')
STAGE = scale.STAGE + '-snapshot-recovery'
TARGET = scale.OUT / 'n3072-snapshot-recovery'
SNAPSHOT_BYTES = 1175359520
SNAPSHOT_CAP = 1280 * scale.MIB


def freeze():
    scale.require_stage(scale.PLAN, scale.STAGE)
    assert not PLAN.exists() and not TARGET.exists()
    old_suite = read(scale.PLAN)
    old_trial = old_suite['trials']['3072']
    old_protocol = ROOT / old_trial['protocol']
    assert sha(old_protocol) == old_trial['sha256']
    old_target = old_protocol.parent
    source_log = old_target / 'zero.log'
    log = source_log.read_text(encoding='utf-8')
    assert f'AssertionError: {SNAPSHOT_BYTES}' in log and 'self.warp_bytes <= maximum_warp_bytes' in log
    assert 'SHARED_RELEASE_GPU_ZERO_OK' not in log and not (old_target / 'independent-zero/proof.json').exists()
    spec = read(old_protocol)
    old_record, old_page = ROOT / spec['record'], ROOT / spec['page']
    state = read(old_record / 'state.json')
    assert state['status'] == 'starting' and not (old_record / 'evaluation.json').exists()
    assert spec['maximum_warp_snapshot_bytes'] < SNAPSHOT_BYTES < SNAPSHOT_CAP
    control_protocol = ROOT / old_suite['trials']['2048']['protocol']
    control = control_protocol.with_name('result.json')
    assert read(control)['status'] == 'completed-resource-probe'
    stop = old_target / 'stopped.json'
    write(stop, dict(status='stopped-before-physics-snapshot-budget', physical_substeps=0, learning_updates=0,
                     original_snapshot_cap_bytes=spec['maximum_warp_snapshot_bytes'],
                     measured_snapshot_bytes=SNAPSHOT_BYTES, zero_log_sha256=sha(source_log)))
    base.publish(old_record, old_page, dict(state, status='stopped-review-required', phase='zero-snapshot-cap',
                error='完整快照超过短测预设上限，物理步数为零'))
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_right-robustness_scale-3072-snapshot_s{old_suite["seed"]}_{stamp}'
    record = ROOT / 'records/benchmarks' / run_id
    page = ROOT / 'records/getup-right-robustness-scale-3072-snapshot-live.html'
    assert not record.exists() and not page.exists()
    spec.update(stage=STAGE, record=base.rel(record), page=base.rel(page), maximum_warp_snapshot_bytes=SNAPSHOT_CAP)
    spec['environment'].update(stage=STAGE, run_id=run_id)
    spec['resource_probe']['maximum_snapshot_bytes_per_scale'] = SNAPSHOT_CAP
    spec['source_sha256'][base.rel(Path(__file__).resolve())] = sha(Path(__file__))
    spec['references'].update({base.rel(path): sha(path) for path in (scale.PLAN, old_protocol, source_log, stop, control)})
    base.shared.assert_sources(spec)
    TARGET.mkdir()
    record.mkdir()
    trial = TARGET / 'protocol.json'
    write(trial, spec)
    copyfile(trial, record / 'protocol.json')
    write(TARGET / 'freeze.json', dict(read(old_target / 'freeze.json'), protocol_sha256=sha(trial),
          original_protocol_sha256=sha(old_protocol), measured_snapshot_bytes=SNAPSHOT_BYTES,
          allowed_snapshot_bytes=SNAPSHOT_CAP))
    write(PLAN, dict(old_suite, stage=STAGE, order=[3072], original_suite_sha256=sha(scale.PLAN),
                     trials={'3072': dict(protocol=base.rel(trial), sha256=sha(trial))}))
    base.publish(record, page, dict(status='starting', purpose='起身并行规模短测 · 3,072 环境快照复核',
        run_id=run_id, controls=0, maximum_controls=spec['maximum_unique_controls'], transitions=0,
        updates=[], learning_updates=0))
    print('GETUP_RIGHT_SCALE_SNAPSHOT_FROZEN', sha(PLAN), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample'))
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze()
    else:
        with patch.multiple(scale, PLAN=PLAN, STAGE=STAGE):
            scale.run(args.mode, 3072)
