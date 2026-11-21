"""Archive a paused, partly completed comparison without running its candidate.

This entry only checks processes, reads saved evidence and packages files. It
does not import the simulator, call an optimizer, change a frozen protocol, or
delete an original asset. ZIP members and external assets are hash-verified.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'records/reports/pause-archive-20261116-monitored'
PAIR = ROOT / 'records/reports/getup-monitored-authorization-20261116'
PLAN = ROOT / 'configs/experiments/observed_getup_monitor_pair_20261116.json'
OLD = ROOT / 'records/reports/pause-archive-20261116'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def paused():
    p = read(ROOT/'project.json')
    assert p['goal_execution']['status'] == 'paused'
    assert p['goal_execution']['pause_requested_by_user'] is True
    assert p['staged_training_request']['pause_requested'] is True
    assert p['staged_training_request']['current_protocol_frozen'] is False
    return p


def process_check():
    import fcntl
    paused()
    raw = subprocess.check_output(['ps', '-eo', 'pid,ppid,etimes,stat,args'], text=True)
    (OUT/'wsl-processes.txt').write_text(raw, encoding='utf-8')
    excluded, pid = set(), os.getpid()
    while pid > 1 and pid not in excluded:
        excluded.add(pid)
        pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
    matches = []
    for line in raw.splitlines()[1:]:
        fields = line.split(None, 4)
        if len(fields) == 5 and int(fields[0]) not in excluded:
            if re.search(r'XGO-Duck|xgoduck|observed_getup|native_run_observer|supervised_ppo', fields[4], re.I):
                matches.append(line)
    # Read-only open: acquiring/releasing the lock does not start any worker.
    with Path('/home/atlas/XGO-Duck/training.lock').open('r') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f, fcntl.LOCK_UN)
    win = read(OUT/'windows-processes.json')
    assert isinstance(win, list)
    result = dict(status='passed' if not matches and not win else 'review-required',
                  checked_at=now(), wsl_matches=matches, windows_matches=win,
                  training_lock_free=True, excluded_ancestor_pids=sorted(excluded),
                  new_physical_steps=0, new_learning_updates=0,
                  windows_query='Read-only elevated CIM query after sandbox CIM access failed')
    write(OUT/'process-check.json', result)
    print(json.dumps(result))
    assert result['status'] == 'passed'


def collect():
    paused()
    assert read(OUT/'process-check.json')['status'] == 'passed'
    plan = read(PLAN)
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    control = plan['arms']['control']['run_id']
    candidate = plan['arms']['throughput']['run_id']
    run = ROOT/'records/experiments'/control
    assert not (ROOT/'records/experiments'/candidate).exists()
    assert not (PAIR/'training/throughput').exists()
    v = read(PAIR/'training/control/independent-verification.json')
    result = read(run/'online-result.json')
    sup = read(PAIR/'training/control/supervisor-result.json')
    assert v['status'] == 'passed-record-and-export-checks'
    assert v['updates'] == result['actual_updates'] == 17
    assert v['transitions'] == result['actual_transitions'] == 1253376
    assert sha(run/'model-final.pt') == v['checkpoint_sha256']
    assert sha(ROOT/'artifacts'/control/'policy.onnx') == v['onnx_sha256']
    assert sha(PAIR/'training/control/execution.log') == sup['log_sha256']
    assert sup['exit_code'] == 0 and sup['protocol_sha256'] == sha(PLAN)
    z = read(PAIR/'zero-verification.json')
    assert z['protocol_sha256'] == sha(PLAN) and z['arrays_compared_exactly'] == 12
    for role, expected in z['zero_result_sha256'].items():
        assert sha(PAIR/'cuda-zero'/role/'result.json') == expected
    old_manifest = read(OLD/'manifest.json')
    assert sha(OLD/'snapshot.zip') == old_manifest['snapshot_sha256']
    assets = read(OLD/'external-assets.json')['files']
    delivery = []
    for item in assets:
        name = item.get('path', item.get('file'))
        assert name, item
        assert sha(ROOT/name) == item['sha256'], name
        if name.startswith('artifacts/delivery/'):
            delivery.append(item)
    assert delivery
    guard = read(PAIR/'training/control/guard-timing.json')
    timings = read(PAIR/'training/control/component-timing.json')
    summary = dict(status='passed-saved-evidence-pair-incomplete', checked_at=now(),
        protocol_sha256=sha(PLAN), frozen_sources_unchanged=len(plan['source_sha256']),
        control=v, guard=guard, components=timings['components'],
        guard_fraction_of_elapsed=guard['wall_seconds']/v['elapsed_seconds'],
        candidate=dict(run_id=candidate, training_started=False, actual_updates=0, actual_transitions=0,
                       cuda_zero_completed=True),
        performance_pair_complete=False, measured_monitor_speedup=None,
        delivery_files_unchanged=len(delivery), previous_external_files_unchanged=len(assets),
        previous_archive_sha256=sha(OLD/'snapshot.zip'),
        new_physical_steps=0, new_learning_updates=0, model_promotion=False, hardware_io=False)
    write(OUT/'saved-evidence-readback.json', summary)
    print(json.dumps({k:summary[k] for k in ['status','frozen_sources_unchanged','delivery_files_unchanged','previous_external_files_unchanged','candidate']}))


def inventory(path):
    assert path.is_file()
    # Canonical physical paths avoid counting the experiments junction twice.
    path = path.resolve()
    assert path.is_relative_to(ROOT.resolve()), path
    return dict(path=path.relative_to(ROOT.resolve()).as_posix(), bytes=path.stat().st_size, sha256=sha(path))


def documents():
    paused()
    names = ['README.md', 'docs/current_goal.md', 'docs/reports/development_report.md',
             'docs/reports/development_history.md', 'artifacts/delivery/iteration_record.md',
             'records/README.md', 'records/training-current.md',
             'records/reports/getup-monitored-authorization-20261116/README.md',
             'records/reports/pause-archive-20261116-monitored/README.md']
    checked, missing, labels = 0, [], []
    for name in names:
        path = ROOT/name
        body = path.read_text(encoding='utf-8')
        if re.search(r'\bV\d+\b|修订稿|版本号', body):
            labels.append(name)
        targets = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', body)
        targets += re.findall(r'(?:src|href)="([^"]+)"', body)
        for target in targets:
            if '://' in target or target.startswith('#'):
                continue
            resolved = (path.parent/target.split('#')[0]).resolve()
            if not resolved.is_relative_to(ROOT.resolve()) or not resolved.exists():
                missing.append(dict(document=name, target=target))
            checked += 1
    record = dict(status='passed' if not missing and not labels else 'failed',
                  documents=names, references_checked=checked, missing=missing,
                  internal_labels=labels, checked_at=now())
    write(OUT/'document-check.json', record)
    print(json.dumps(record, ensure_ascii=False))
    assert record['status'] == 'passed'


def package():
    p = paused()
    assert p['staged_training_request']['learning_running'] is False
    assert p['staged_training_request']['simulation_running'] is False
    assert read(OUT/'saved-evidence-readback.json')['status'] == 'passed-saved-evidence-pair-incomplete'
    assert read(OUT/'document-check.json')['status'] == 'passed'
    archive = OUT/'snapshot.zip'
    assert not archive.exists(), 'Keep previous snapshot immutable'
    plan = read(PLAN)
    candidates = set()
    for name in plan['source_sha256']:
        candidates.add((ROOT/name).resolve())
    for folder in [PAIR, ROOT/'records/reports/getup-runtime-review-20261116',
                   ROOT/'records/experiments'/plan['arms']['control']['run_id'],
                   ROOT/'artifacts'/plan['arms']['control']['run_id'],
                   ROOT/'artifacts/delivery', OUT]:
        candidates.update(x.resolve() for x in folder.rglob('*') if x.is_file())
    names = ['README.md', 'AGENTS.md', 'project.json', 'docs/current_goal.md',
             'docs/project_structure.md', 'docs/runbooks/pc_training.md',
             'docs/runbooks/hardware_commissioning.md',
             'docs/reports/development_report.md', 'docs/reports/development_report.html',
             'docs/reports/development_history.md', 'docs/reports/development_history.html',
             'records/README.md', 'records/training-current.md', 'records/training-current.json',
             'records/reports/development-report-check.json',
             'scripts/validation/archive_monitored_training_pause.py',
             'scripts/validation/prepare_monitored_getup_pair.py',
             'scripts/validation/review_monitored_getup_pair.py']
    candidates.update((ROOT/n).resolve() for n in names)
    candidates.update(x.resolve() for x in (ROOT/'configs/experiments').glob('observed_getup_monitor_*.json'))
    files, external = [], []
    large_types = {'.mp4', '.pt', '.onnx', '.npz', '.zip', '.npy'}
    for f in sorted(candidates):
        if f.name in ('manifest.json', 'verification.json', 'external-assets.json') and f.parent == OUT.resolve():
            continue
        item = inventory(f)
        if f.suffix.lower() in large_types or f.stat().st_size > 3_000_000:
            external.append(item)
        else:
            files.append(item)
    old = inventory(OLD/'snapshot.zip')
    write(OUT/'external-assets.json', dict(files=external, previous_archives=[old],
        scope='Keep these assets and the full XGO-Duck project. This ZIP is an incremental source/document snapshot, not a standalone environment backup.'))
    files.append(inventory(OUT/'external-assets.json'))
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(ROOT/f['path'], f['path'])
    write(OUT/'manifest.json', dict(created_at=now(), files=files, snapshot_sha256=sha(archive),
        external_assets='external-assets.json', standalone_backup=False, original_files_deleted=0))
    verify()


def verify():
    paused()
    m = read(OUT/'manifest.json'); archive = OUT/'snapshot.zip'
    assert sha(archive) == m['snapshot_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == len(m['files'])
        for f in m['files']:
            data = z.read(f['path'])
            assert len(data) == f['bytes'] and hashlib.sha256(data).hexdigest() == f['sha256']
            assert sha(ROOT/f['path']) == f['sha256'], f['path']
    assets = read(OUT/'external-assets.json')
    for f in assets['files']+assets['previous_archives']:
        assert sha(ROOT/f['path']) == f['sha256'], f['path']
    plan = read(PLAN)
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    result = dict(status='passed', checked_at=now(), zip_files_checked=len(m['files']),
        zip_bytes=archive.stat().st_size, zip_sha256=sha(archive), manifest_sha256=sha(OUT/'manifest.json'),
        external_files_rechecked=len(assets['files']),
        external_bytes=sum(x['bytes'] for x in assets['files']),
        previous_archives_rechecked=len(assets['previous_archives']),
        frozen_sources_unchanged=len(plan['source_sha256']),
        delivery_files_unchanged=read(OUT/'saved-evidence-readback.json')['delivery_files_unchanged'],
        training_status='paused', original_files_deleted=0,
        new_physical_steps=0, new_learning_updates=0, hardware_io=False)
    write(OUT/'verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['process', 'collect', 'documents', 'package', 'verify'])
    mode = parser.parse_args().mode
    {'process': process_check, 'collect': collect, 'documents': documents,
     'package': package, 'verify': verify}[mode]()
