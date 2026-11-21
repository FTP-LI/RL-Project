"""Observe the selected continuation process and Linux guest CPU counters.

No affinity, priority, clocks or training settings change. A process value of
100% means one logical CPU; guest percentages are not Windows host CPU usage.
Raw /proc counters are retained so interval calculations can be recomputed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--samples', type=int, default=12, help='One-second intervals; one extra initial snapshot')
    args = parser.parse_args()
    if not 1 <= args.samples <= 60:
        parser.error('Use 1 to 60 intervals')
    root = Path(__file__).resolve().parents[2]
    record = root/'records/experiments'/args.run_id
    assert record.parent == root/'records/experiments' and record.is_dir()
    process = Path('/proc')/str(args.pid)
    command = (process/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    assert any(Path(arg).name == 'resume_official.py' for arg in command)
    assert command[command.index('--run-id')+1] == args.run_id
    assert Path(command[command.index('--record-dir')+1]).resolve() == record
    hz, page_size = os.sysconf('SC_CLK_TCK'), os.sysconf('SC_PAGE_SIZE')

    def snapshot():
        fields = (process/'stat').read_text().rsplit(') ', 1)[1].split()
        cpus = {}
        for line in Path('/proc/stat').read_text().splitlines():
            name, *values = line.split()
            if re.fullmatch(r'cpu\d*', name):
                # Exclude guest/guest_nice, which Linux also counts in user/nice.
                cpus[name] = list(map(int, values[:8]))
        return dict(time=datetime.now(timezone.utc).isoformat(), monotonic_ns=time.monotonic_ns(),
                    user_ticks=int(fields[11]), system_ticks=int(fields[12]), threads=int(fields[17]),
                    process_start_ticks=int(fields[19]), rss_pages=int(fields[21]), cpu_counters=cpus)

    def last_label():
        with (record/'training.log').open('rb') as stream:
            stream.seek(max(0, (record/'training.log').stat().st_size-16000))
            labels = re.findall(rb'Learning iteration\s+(\d+)/', stream.read())
        return int(labels[-1]) if labels else None

    first_label = last_label()
    snapshots = [snapshot()]
    for _ in range(args.samples):
        time.sleep(1)
        snapshots.append(snapshot())
    assert (process/'cmdline').read_bytes().decode().rstrip('\0').split('\0') == command
    intervals = []
    for before, after in zip(snapshots, snapshots[1:]):
        assert before['process_start_ticks'] == after['process_start_ticks']
        seconds = (after['monotonic_ns']-before['monotonic_ns'])/1e9
        busy = {}
        assert before['cpu_counters'].keys() == after['cpu_counters'].keys()
        for name, current in after['cpu_counters'].items():
            delta = [a-b for a,b in zip(current, before['cpu_counters'][name])]
            assert sum(delta) > 0
            busy[name] = 100*(sum(delta)-delta[3]-delta[4])/sum(delta)
        consumed = after['user_ticks']+after['system_ticks']-before['user_ticks']-before['system_ticks']
        assert consumed >= 0
        intervals.append(dict(seconds=seconds, process_cpu_percent_one_core_100=100*consumed/hz/seconds,
                              guest_cpu_busy_percent=busy.pop('cpu'), guest_core_busy_percent=busy,
                              busiest_guest_core_percent=max(busy.values()), threads=after['threads'],
                              process_rss_mib=after['rss_pages']*page_size/2**20))
    keys = ('process_cpu_percent_one_core_100','guest_cpu_busy_percent',
            'busiest_guest_core_percent','process_rss_mib','threads')
    summary = {key:dict(min=min(v[key] for v in intervals), median=statistics.median(v[key] for v in intervals),
                        max=max(v[key] for v in intervals)) for key in keys}
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    output = record/f'cpu-observation-{stamp}.json'
    script_copy = output.with_suffix('.py')
    shutil.copyfile(Path(__file__), script_copy)
    result = dict(run_id=args.run_id, pid=args.pid, observed_at=datetime.now(timezone.utc).isoformat(),
                  command=command, first_observed_log_label=first_label, last_observed_log_label=last_label(),
                  clock_ticks_per_second=hz, page_size=page_size, guest_logical_cpus=len(snapshots[0]['cpu_counters'])-1,
                  raw_snapshots=snapshots, intervals=intervals, summary=summary,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Read-only /proc counters for this exact continuation process and WSL guest; no host-wide CPU claim, controlled comparison, or bottleneck diagnosis')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), pid=args.pid, intervals=len(intervals), summary=summary), indent=2))


if __name__ == '__main__':
    main()
