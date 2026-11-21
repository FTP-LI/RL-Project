"""Recompute saved timing, GPU and process CPU observations without training.

The GPU and CPU samples are separate time windows. This checks arithmetic and
log provenance; it does not identify a bottleneck or measure a speedup.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values))


def same(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            same(actual[key], expected[key])
    else:
        assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10), (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gpu_observation', type=Path)
    parser.add_argument('cpu_observation', type=Path)
    args = parser.parse_args()
    gpu_path, cpu_path = args.gpu_observation.resolve(), args.cpu_observation.resolve()
    assert gpu_path.parent == cpu_path.parent
    gpu, cpu = (json.loads(p.read_text()) for p in (gpu_path, cpu_path))
    assert gpu['run_id'] == cpu['run_id'] == gpu_path.parent.name
    assert digest(gpu_path.with_suffix('.py')) == gpu['script_sha256']
    assert digest(cpu_path.with_suffix('.py')) == cpu['script_sha256']

    raw_log = (gpu_path.parent/'training.log').read_text()
    pieces = re.split(r'(Learning iteration\s+(\d+)/\d+)', raw_log)
    blocks = {int(pieces[i+1]): pieces[i]+pieces[i+2] for i in range(1, len(pieces), 3)}
    labels = [row['iteration'] for row in gpu['recent_iterations']]
    assert labels == list(range(labels[0], labels[-1]+1))
    assert labels[-1] == gpu['latest_completed_iteration']
    names = dict(collection_seconds='Collection time', learning_seconds='Learning time',
                 iteration_seconds='Iteration time', transitions_per_second='Steps per second')
    for row in gpu['recent_iterations']:
        block = blocks[row['iteration']]
        assert re.search(r'ETA:\s+\S+', block)
        for key, label in names.items():
            value = float(re.search(re.escape(label)+r':\s+([\d.]+)', block)[1])
            same(row[key], value)
    ordinary = [row for row in gpu['recent_iterations'] if row['iteration'] >= 20 and row['iteration'] % 250 >= 10][-100:]
    assert len(ordinary) == gpu['ordinary_iteration_count'] == 100
    timing = {key: statistics.median(row[key] for row in ordinary) for key in names}
    same(timing, gpu['ordinary_iteration_medians'])
    telemetry = {key: aggregate([row[key] for row in gpu['samples']]) for key in gpu['gpu_telemetry']}
    same(telemetry, gpu['gpu_telemetry'])
    gpu_times = [datetime.fromisoformat(row['time']) for row in gpu['samples']]
    assert all(a < b for a, b in zip(gpu_times, gpu_times[1:]))

    snapshots = cpu['raw_snapshots']
    assert len(snapshots) == len(cpu['intervals'])+1
    assert len({row['process_start_ticks'] for row in snapshots}) == 1
    assert all(len(row['cpu_counters'])-1 == cpu['guest_logical_cpus'] for row in snapshots)
    recomputed = []
    for before, after, saved in zip(snapshots, snapshots[1:], cpu['intervals']):
        seconds = (after['monotonic_ns']-before['monotonic_ns'])/1e9
        assert seconds > 0
        ticks = (after['user_ticks']-before['user_ticks'])+(after['system_ticks']-before['system_ticks'])
        assert ticks >= 0
        busy = {}
        for name in before['cpu_counters']:
            delta = [b-a for a, b in zip(before['cpu_counters'][name], after['cpu_counters'][name])]
            assert len(delta) == 8 and all(v >= 0 for v in delta) and sum(delta) > 0
            busy[name] = sum(delta[i] for i in (0, 1, 2, 5, 6, 7))/sum(delta)*100
        row = dict(seconds=seconds, process_cpu_percent_one_core_100=ticks/cpu['clock_ticks_per_second']/seconds*100,
                   guest_cpu_busy_percent=busy.pop('cpu'), guest_core_busy_percent=busy,
                   busiest_guest_core_percent=max(busy.values()), threads=after['threads'],
                   process_rss_mib=after['rss_pages']*cpu['page_size']/1048576)
        same(row, saved)
        recomputed.append(row)
    cpu_summary = {key: aggregate([row[key] for row in recomputed]) for key in cpu['summary']}
    same(cpu_summary, cpu['summary'])
    cpu_seconds = (snapshots[-1]['monotonic_ns']-snapshots[0]['monotonic_ns'])/1e9
    cpu_ticks = snapshots[-1]['user_ticks']+snapshots[-1]['system_ticks']-snapshots[0]['user_ticks']-snapshots[0]['system_ticks']
    cpu_times = [datetime.fromisoformat(row['time']) for row in snapshots]
    assert gpu_times[-1] < cpu_times[0], 'These observations should describe separate recorded windows'

    excerpt = gpu_path.with_name(gpu_path.stem+'-verified-log.txt')
    excerpt.write_text('\n'.join(blocks[label] for label in labels))
    output = gpu_path.with_name(gpu_path.stem+'-check.json')
    verifier_copy = gpu_path.with_name(gpu_path.stem+'-verify.py')
    shutil.copyfile(__file__, verifier_copy)
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(), run_id=gpu['run_id'],
                  gpu_observation=gpu_path.name, gpu_sha256=digest(gpu_path), cpu_observation=cpu_path.name,
                  cpu_sha256=digest(cpu_path), verifier_sha256=digest(verifier_copy),
                  log_excerpt=excerpt.name, log_excerpt_sha256=digest(excerpt),
                  log_rows_verified=len(labels), ordinary_first_label=ordinary[0]['iteration'], ordinary_last_label=ordinary[-1]['iteration'],
                  ordinary_iteration_count=len(ordinary), ordinary_iteration_medians=timing,
                  gpu_samples_verified=len(gpu_times), gpu_telemetry=telemetry,
                  cpu_snapshots_verified=len(snapshots), cpu_intervals_verified=len(recomputed),
                  guest_logical_cpus=cpu['guest_logical_cpus'], cpu_summary=cpu_summary,
                  process_cpu_whole_window_percent_one_core_100=cpu_ticks/cpu['clock_ticks_per_second']/cpu_seconds*100,
                  gpu_window=[gpu_times[0].isoformat(), gpu_times[-1].isoformat()],
                  cpu_window=[cpu_times[0].isoformat(), cpu_times[-1].isoformat()],
                  simultaneous_windows=False, controlled_speedup_verified=False,
                  scope='Saved timing/log correspondence and GPU/CPU arithmetic; no training, host-wide CPU measurement or bottleneck conclusion')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
