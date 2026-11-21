"""Narrow the memory-limited interval without repeating the closed scale screen."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import torch

import local_getup_scale as bench
from s1_reference_inputs import ROOT, read, sha
from record_io import write
from training_live_page import publish

BASE = bench.PLAN
PLAN = ROOT/'configs/experiments/local_getup_scale_refine.json'


def freeze():
    assert not PLAN.exists()
    first = read(BASE)
    result_path = ROOT/'records/benchmarks'/first['run_id']/'summary.json'
    result = read(result_path)
    last = result['trials'][-1]
    assert result['status'] == 'closed-after-trial-stop' and result['selected_worlds'] == 2048
    assert last['worlds'] == 4096 and last['error'] == 'AssertionError: Observed GPU memory ceiling'
    assert result['protocol_sha256'] == sha(BASE)
    baseline = [r['net_transitions_per_second'] for r in result['trials'] if r['worlds']==2048]
    assert len(baseline) == 2
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    spec = dict(first, run_id=f'duck_getup_scale-refine_s3062_{stamp}', passes=[[3072,3584]], expansion=[],
                parent_result=result_path.relative_to(ROOT).as_posix(), parent_result_sha256=sha(result_path),
                anchor_worlds=2048, anchor_rates=baseline, repeats_per_size=2,
                rule='Measure 3072 twice. Test 3584 only when both 3072 passes beat both 2048 passes.')
    spec['source_sha256'].update({Path(__file__).relative_to(ROOT).as_posix():sha(Path(__file__)),
                                 BASE.relative_to(ROOT).as_posix():sha(BASE)})
    for n in spec['passes'][0]:
        bench.make_config(spec, n)
    assert not torch.cuda.is_initialized()
    write(PLAN, spec)
    print('LOCAL_SCALE_REFINE_FROZEN', spec['run_id'], sha(PLAN), flush=True)


def run():
    spec = bench.checked_plan()
    assert sha(ROOT/spec['parent_result']) == spec['parent_result_sha256']
    suite = ROOT/'records/benchmarks'/spec['run_id']
    suite.mkdir()
    (suite/'README.md').write_text(
        '# 本机起身规模补测\n\n4,096 环境在零步显存检查停止。沿用原采样场景、固定权重和种子，'
        '从 3,072 两次测起，再按收益决定是否测试 3,584。录像链接指向此前观察样例。\n\n'
        '[冻结方案](../../../configs/experiments/local_getup_scale_refine.json) · [结果](summary.json)\n',
        encoding='utf-8')
    results = []
    best, rates = spec['anchor_worlds'], spec['anchor_rates']
    stop = None
    for worlds in spec['passes'][0]:
        comparison = []
        for _ in range(spec['repeats_per_size']):
            assert read(ROOT/'project.json')['goal_execution']['status'] == 'active'
            i = len(results)+1
            record = suite/f'trial-{i:02d}-n{worlds}'
            record.mkdir()
            command = [sys.executable,'-B',str(Path(__file__)),'--trial',str(i),str(worlds)]
            write(record/'command.json', dict(argv=command, protocol_sha256=sha(PLAN), started_at=bench.now()))
            with (record/'execution.log').open('x') as log:
                try:
                    exit_code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                               timeout=spec['trial_limit_seconds']).returncode
                    error = f'Exit {exit_code}'
                except subprocess.TimeoutExpired:
                    error = 'Wall time cap'
            receipt = record/'result.json'
            if not receipt.exists():
                write(receipt, dict(status='stopped', worlds=worlds, trial=i, error=error))
            value = read(receipt)
            results.append(value)
            write(suite/'summary.json', dict(status='running', trials=results))
            print('LOCAL_SCALE_REFINE_TRIAL', i, worlds, value['status'], value.get('net_transitions_per_second'), flush=True)
            if value['status'] != 'completed':
                stop = value.get('error', 'trial stopped')
                break
            comparison.append(value['net_transitions_per_second'])
        if stop or min(comparison) <= max(rates):
            break
        best, rates = worlds, comparison
    status = 'closed-after-trial-stop' if stop else 'completed'
    write(suite/'summary.json', dict(status=status, trials=results, selected_worlds=best,
          selected_net_rates=rates, stop_reason=stop, learning_updates=0, protocol_sha256=sha(PLAN)))
    if (record/'state.json').exists():
        state = read(record/'state.json')
        state.update(status='stopped' if stop else 'completed', updated_at=bench.now(),
                     purpose=f'规模补测结束 · 入选 {best} 环境 · 录像来自此前观察检查')
        publish(record, state)
    print('LOCAL_SCALE_REFINE_FINISHED', best, status, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--trial', nargs=2, type=int)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        bench.PLAN = PLAN  # Reuse the unchanged worker with the new frozen inputs.
        if args.trial:
            bench.trial(*args.trial)
        else:
            run()
