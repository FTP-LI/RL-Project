"""Use a transport-only lock copy, then restore and verify the official lock."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path

ORIGIN = 'https://files.pythonhosted.org/'
MIRROR = 'https://pypi.tuna.tsinghua.edu.cn/'


def transport_copy(original: bytes) -> tuple[bytes, int]:
    data = tomllib.loads(original.decode())
    count = 0
    for package in data['package']:
        for artifact in [package.get('sdist', {}), *package.get('wheels', [])]:
            if artifact.get('url', '').startswith(ORIGIN):
                count += 1
    text = original.decode()
    # Only download URLs change: registry, versions, hashes and markers are retained.
    assert text.count(ORIGIN) == count
    derived = text.replace(ORIGIN, MIRROR).encode()
    assert tomllib.loads(derived.decode().replace(MIRROR, ORIGIN)) == tomllib.loads(original.decode())
    return derived, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    official = (project / 'official_resources/xgoduck_rl/uv.lock').read_bytes()
    derived, count = transport_copy(official)
    report = {
        'started_at': datetime.now().astimezone().isoformat(),
        'official_lock_sha256': hashlib.sha256(official).hexdigest(),
        'transport_lock_sha256': hashlib.sha256(derived).hexdigest(),
        'download_urls_replaced': count,
        'versions_hashes_registries_and_markers_unchanged': True,
        'download_mirror': MIRROR,
        'build_dependency_index': MIRROR + 'simple',
        'official_lock_restored': False,
        'status': 'planned',
    }
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return
    work = Path.home() / 'XGO-Duck/workspace/xgoduck_rl'
    lock = work / 'uv.lock'
    assert lock.read_bytes() == official, 'Working lock differs from official archive; refusing to overwrite.'
    records = project / 'records/deployment' / datetime.now().strftime('%Y-%m-%d')
    records.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime('%H%M%S')
    (records / f'mirror-transport-{run_id}.lock').write_bytes(derived)
    report_path = records / f'mirror-sync-{run_id}.json'
    env = os.environ.copy()
    env['UV_CONCURRENT_DOWNLOADS'] = '4'
    command = ['uv', 'sync', '--frozen', '--default-index', MIRROR + 'simple']
    report['command'] = command
    print(json.dumps(report, indent=2), flush=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        lock.write_bytes(derived)
        report['sync_exit_code'] = subprocess.run(command, cwd=work, env=env).returncode
        report['status'] = 'downloaded' if report['sync_exit_code'] == 0 else 'failed'
    finally:
        lock.write_bytes(official)
        report['official_lock_restored'] = lock.read_bytes() == official
        report['finished_at'] = datetime.now().astimezone().isoformat()
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print('Official lock restored:', report['official_lock_restored'], flush=True)
    if report.get('sync_exit_code') != 0:
        raise SystemExit(report.get('sync_exit_code', 1))
    report['offline_locked_exit_code'] = subprocess.run(['uv', 'sync', '--locked', '--offline'], cwd=work, env=env).returncode
    report['status'] = 'passed' if report['offline_locked_exit_code'] == 0 else 'failed'
    report['finished_at'] = datetime.now().astimezone().isoformat()
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(report['offline_locked_exit_code'])


if __name__ == '__main__':
    main()
