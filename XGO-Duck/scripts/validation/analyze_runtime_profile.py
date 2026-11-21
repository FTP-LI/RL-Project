"""Summarize an independently verified CPU/CUDA diagnostic pair.

Only saved files are read: no simulation, inference or optimizer runs. Compare
the five prespecified updates; do not treat one ordered pair as a speedup test.
CPU scopes include waiting and nested children. GPU activity is reported as a
union of recorded intervals, not utilization or an additive wall-time budget.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_duration(events):
    intervals = sorted((e['ts'], e['ts'] + e['dur']) for e in events)
    total, end = 0.0, float('-inf')
    for left, right in intervals:
        total += max(0.0, right - max(left, end))
        end = max(end, right)
    return total


def analyze(pair):
    checked = read(pair/'verification.json')
    assert checked['status'] == 'passed'
    for name, expected in checked['input_sha256'].items():
        assert sha(pair/name) == expected, name
    trace_path = pair/'traced/trace.json'
    assert sha(trace_path) == checked['trace_sha256']
    timing = {}
    for mode in ('reference', 'traced'):
        rows = read(pair/mode/'iteration-timing.json')
        selected = rows[22:27]
        assert [r['iteration'] for r in selected] == list(range(5022, 5027))
        wall = sum(r['wall_seconds'] for r in selected)
        assert abs(wall-checked['captured_wall_seconds'][mode]) < 1e-8
        timing[mode] = dict(
            updates=5, wall_seconds=wall, transitions=5*24*1024,
            transitions_per_second=5*24*1024/wall,
            median_wall_seconds=statistics.median(r['wall_seconds'] for r in selected),
            collection_seconds=sum(r['collection_seconds'] for r in selected),
            learning_seconds=sum(r['learning_seconds'] for r in selected),
            cuda_graph_state=read(pair/checked['result_paths'][mode])['cuda_graph_state'],
            gpu_before=read(pair/mode/'gpu-before.json'),
            gpu_after=read(pair/mode/'gpu-after.json'))
    summary = read(pair/checked['summary_path'])
    ops = summary['operators']
    scopes = sorted((r for r in ops if r['name'].startswith('xgo/') and r['cpu_total_us'] > 0),
                    key=lambda r: r['cpu_total_us'], reverse=True)
    events = [e for e in read(trace_path)['traceEvents'] if e.get('ph') == 'X']
    gpu = [e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    assert len(gpu) == checked['raw_gpu_activity_events']
    top_host = sorted((r for r in ops if not r['name'].startswith(('xgo/', 'ProfilerStep'))),
                      key=lambda r: r['self_cpu_us'], reverse=True)[:15]
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  captured_iteration_labels=[5022,5026], timing=timing,
                  traced_to_reference_ratio=timing['traced']['wall_seconds']/timing['reference']['wall_seconds'],
                  scopes=scopes, largest_self_cpu_operators=top_host,
                  raw_gpu_event_categories=dict(Counter(e['cat'] for e in gpu)),
                  gpu_interval_union_seconds=union_duration(gpu)/1e6,
                  gpu_duration_sum_seconds=sum(e['dur'] for e in gpu)/1e6,
                  verification_sha256=sha(pair/'verification.json'), trace_sha256=sha(trace_path),
                  controlled_speedup_verified=False, formal_model_changed=False,
                  interpretation='One reference-then-traced pair. CPU durations include waiting; nested scopes overlap. Recorded GPU intervals do not measure hardware utilization.')
    inputs = ['verification.json', checked['summary_path'], 'traced/trace.json', *checked['result_paths'].values()]
    inputs += [f'{mode}/{name}' for mode in ('reference','traced')
               for name in ('iteration-timing.json','gpu-before.json','gpu-after.json')]
    result['input_sha256'] = {name:sha(pair/name) for name in inputs}
    output = pair/'analysis'
    output.mkdir(exist_ok=True)
    (output/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    import shutil
    shutil.copy2(__file__, output/Path(__file__).name)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair', type=Path)
    analyze(parser.parse_args().pair.resolve())
