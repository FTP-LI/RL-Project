"""Inventory saved training evidence and prune only reviewed legacy checkpoints."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import stat
import subprocess

from cleanup_training_storage import LOGS, ROOT, sha


OUT = ROOT / 'records/reports/training-curation'
RECOVERY = ROOT / 'artifacts/recovery'
EXPERIMENTS = ROOT / 'records/experiments'
WORK = LOGS.parents[1]
BASELINE = 'baseline_continuation_20261101T101809_s42'
# These closed records have documented initial and terminal checkpoints.
# Newer short runs use protocol-specific files and retain their full inputs.
CLOSED = {
    BASELINE: 'training-finished',
    'baseline_extension_20261101T160203_s42': 'interrupted',
    'gait_repair_control_20261101T174853_s42': 'training-finished',
    'gait_repair_target_envelope_20261101T174853_s42': 'training-finished',
    'head_command_repair_control_20261101T210228_s42': 'failed',
    'head_command_repair_control_20261101T213026_s42': 'training-finished',
    'head_command_repair_reachable_head_20261101T213026_s42': 'training-finished',
}
AREAS = (
    'records/experiments', 'records/evaluations', 'records/benchmarks',
    'records/profiling', 'records/reports', 'artifacts', 'official_resources',
    'workspace/official-reference',
)
TEXT_ROOTS = (
    'docs', 'configs', 'scripts', 'records', 'artifacts', '.vscode', 'project.json',
    str(WORK / 'src'), str(WORK / 'scripts'), str(LOGS),
)
MODEL = re.compile(r'model_(\d+)\.pt')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def named(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def file_info(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def checked_file(path, boundary):
    if path.resolve(strict=True) != path or not path.is_relative_to(boundary) or not path.is_file():
        raise ValueError(f'Unexpected file path: {path}')
    return path


def describe_storage():
    models, areas = [], []
    for base in (*[ROOT / label for label in AREAS], LOGS):
        total, folders = 0, Counter()
        for folder, dirs, names in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in {OUT.name, 'browser-preview-temp', '.git', '__pycache__'})
            for name in sorted(names):
                path = Path(folder, name)
                info = path.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    continue
                suffix = path.suffix.lower() or '(none)'
                total += info.st_size
                folders[suffix] += 1
                if suffix in {'.pt', '.pth', '.onnx'}:
                    models.append(file_info(path))
        areas.append(dict(path=named(base), bytes=total, files=folders.total(), formats=dict(folders)))
    return areas, models


def numbered(folder):
    return {int(m[1]): path for path in folder.glob('model_*.pt') if (m := MODEL.fullmatch(path.name))}


def choices():
    candidates, retained = [], []
    for run, status in CLOSED.items():
        result = read(EXPERIMENTS / run / 'training-result.json')
        if result.get('status') != status or result.get('run_id') != run:
            raise ValueError(f'Changed run status: {run}')
        folder = Path(result['log_dir'])
        if folder.resolve(strict=True) != folder or not folder.is_relative_to(LOGS) or not folder.name.endswith('_' + run):
            raise ValueError(f'Unexpected training log folder: {folder}')
        source, backups = numbered(folder), numbered(RECOVERY / run)
        labels = sorted(source.keys() | backups.keys())
        keep = set(labels[:1] + labels[-2:])
        terminal = source[max(labels)]
        for label in labels:
            for path in filter(None, (source.get(label), backups.get(label))):
                if label in keep or (run == BASELINE and path.is_relative_to(RECOVERY)):
                    reason = ('initial or recent recovery point' if label in keep else
                              'historical completion verifier requires this Windows backup')
                    retained.append(dict(**file_info(path), run=run, reason=reason))
                    continue
                audit = EXPERIMENTS / run / 'checkpoint-checks' / f'{path.stem}.json'
                candidates.append(dict(**file_info(path), run=run, audit=named(audit) if audit.is_file() else None,
                                       terminal=file_info(terminal)))
    return candidates, retained


@lru_cache(maxsize=None)
def history_text(source):
    return (ROOT / source).read_text(encoding='utf-8-sig')


def is_catalog(source, line, row):
    filename = Path(row['path']).name
    check = f"{row['run']}/checkpoint-checks/{Path(filename).stem}.json"
    if source == row['audit']:
        check_data = read(ROOT / source)
        return check_data['status'] == 'passed' and check_data['checkpoint_sha256'] == row['sha256']
    if source.endswith('/training-result.json') and f"records/experiments/{row['run']}/" in source:
        # The historical 'checkpoints' list is a file catalog. Resume and export
        # fields are outside it and remain dependencies.
        data = read(ROOT / source)
        listed_path = str(Path(data['log_dir']) / filename)
        entry = dict(path=listed_path, sha256=row['sha256'])
        if entry in data.get('checkpoints', []):
            for key, value in entry.items():
                field = f'{json.dumps(key)}: {json.dumps(value)}'
                if line.strip().rstrip(',') == field and history_text(source).count(field) == 1:
                    return True
    history = ('docs/reports/development_history.', 'records/reports/project-organization/source-report.md')
    if source.startswith(history) and ('/' + filename) not in line and '--checkpoint' not in line:
        if f'/checkpoint-checks/{Path(filename).stem}.json' in line:
            return True
        if line.strip() == f'<td>{filename}</td>':
            return check in history_text(source)
        if '/head_command_repair_control_20261101T210228_s42/interruption-recovery.json' in line:
            return True  # Descriptive historical interruption note.
    if source == 'records/reports/goal-review/current_goal-before-review.md':
        return ('/' + filename) not in line and '--checkpoint' not in line
    if f"(e.g. '{filename}')" in line or (source.startswith('records/deployment/') and
                                          f"'{filename}'). (default: None)" in line):
        return True  # Help text for the upstream optional W&B argument.
    return source == 'records/reports/storage-cleanup/plan.json' and line.strip().rstrip(',') == json.dumps(filename)


def references(candidates):
    by_token = defaultdict(list)
    for row in candidates:
        by_token[Path(row['path']).name].append(row)
        by_token[row['sha256']].append(row)
    pattern = '|'.join(re.escape(token) for token in by_token)
    command = ['rg', '--json', '--no-ignore', '--hidden', '-g', '*.{md,html,json,jsonl,py,sh,yaml,yml,toml,log,txt,csv}',
               '-g', '!**/.git/**', '-g', '!**/browser-preview-temp/**', '-g', f'!**/{OUT.name}/**',
               '-g', '!**/curate_training_checkpoints.py', '-e', pattern, '--', *TEXT_ROOTS]
    report = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', check=False)
    if report.returncode not in {0, 1}:
        raise RuntimeError(report.stderr)
    hits = defaultdict(lambda: dict(catalog=[], protected=[]))
    for entry in map(json.loads, report.stdout.splitlines()):
        if entry['type'] != 'match':
            continue
        data = entry['data']
        source = data['path']['text']
        line = data['lines'].get('text', '').replace('\\\\', '/').replace('\\', '/')
        for match in data['submatches']:
            token = match['match']['text']
            for row in by_token[token]:
                if (token.endswith('.pt') and re.search(r'\d{8}T\d{6}(?:_s\d+)?/' + re.escape(token), line)
                        and row['run'] not in line and row['run'] not in source and row['sha256'] not in line):
                    continue  # A qualified path for another run with the same save number.
                kind = 'catalog' if is_catalog(source, line, row) else 'protected'
                location = f"{source}:{data['line_number']}"
                if location not in hits[row['path']][kind]:
                    hits[row['path']][kind].append(location)
    return hits


def make_plan():
    if (OUT / 'result.json').exists():
        raise ValueError('Keep the applied plan with its execution receipt')
    areas, models = describe_storage()
    candidates, retained = choices()
    refs = references(candidates)
    delete = []
    for row in candidates:
        row.update(refs[row['path']])
        if row['protected']:
            retained.append(dict(**row, reason='runtime dependency or unresolved reference'))
            continue
        path = Path(row['path'])
        backup = RECOVERY / row['run'] / path.name
        if row['run'] == BASELINE and path.is_relative_to(LOGS):
            copy = file_info(backup)
            if copy['sha256'] != row['sha256']:
                raise ValueError(f'Windows backup does not match: {backup}')
            row['identical_backup'] = copy
        delete.append(dict(**row, reason='superseded intermediate with no active reference'))
    plan = dict(status='reviewed-inventory', created_at=now(), areas=areas, models=models,
                retained_legacy=retained, delete=delete, delete_bytes=sum(row['bytes'] for row in delete))
    OUT.mkdir(exist_ok=True)
    write(OUT / 'plan.json', plan)
    print(json.dumps(dict(models=len(models), retained_legacy=len(retained), deletes=len(delete),
                          bytes=plan['delete_bytes'], plan_sha256=sha(OUT / 'plan.json'))))


def apply(plan_sha256):
    plan_file = OUT / 'plan.json'
    receipt = OUT / 'result.json'
    if receipt.exists() or sha(plan_file) != plan_sha256:
        raise ValueError('Plan changed or was already applied')
    plan = read(plan_file)
    if read(ROOT / 'project.json')['goal_execution']['status'] != 'paused':
        raise ValueError('Pause the staged goal before changing its saved artifacts')
    with (LOGS.parents[3] / 'training.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        permitted, _ = choices()
        permissible = {(row['path'], row['sha256']) for row in permitted}
        rows = plan['delete']
        if len({row['path'] for row in rows}) != len(rows):
            raise ValueError('Duplicate delete path')
        refs = references(rows) if rows else {}
        for row in rows:
            path = Path(row['path'])
            boundary = LOGS if path.is_relative_to(LOGS) else RECOVERY
            checked_file(path, boundary)
            if (row['path'], row['sha256']) not in permissible or refs[row['path']]['protected']:
                raise ValueError(f'Unexpected path, hash or dependency: {path}')
            if path.stat().st_size != row['bytes'] or path.stat().st_nlink != 1:
                raise ValueError(f'Changed file size or link count: {path}')
            for keep in filter(None, (row['terminal'], row.get('identical_backup'))):
                kept = Path(keep['path'])
                checked_file(kept, LOGS if kept.is_relative_to(LOGS) else RECOVERY)
                if file_info(kept) != keep:
                    raise ValueError(f'Changed retained checkpoint: {kept}')
        result = dict(status='in-progress', started_at=now(), plan_sha256=plan_sha256, deleted=[])
        write(receipt, result)
        for row in rows:
            Path(row['path']).unlink()
            result['deleted'].append(row)
            write(receipt, result)
        assert all(not Path(row['path']).exists() for row in rows)
        location = lambda row: 'wsl' if Path(row['path']).is_relative_to(LOGS) else 'windows'
        totals = {site: sum(row['bytes'] for row in rows if location(row) == site) for site in ('wsl', 'windows')}
        result.update(status='completed', finished_at=now(), logical_bytes_by_location=totals,
                      logical_bytes_reclaimed=sum(totals.values()), new_training_or_evaluation_started=False)
        write(receipt, result)
        print(json.dumps(dict(deleted=len(rows), bytes=totals, result_sha256=sha(receipt))))


def verify():
    plan_path, result_path = OUT / 'plan.json', OUT / 'result.json'
    plan, result = read(plan_path), read(result_path)
    assert result['status'] == 'completed' and result['plan_sha256'] == sha(plan_path)
    assert plan['delete'] == result['deleted']
    deleted = {row['path'] for row in result['deleted']}
    assert all(not Path(path).exists() for path in deleted)
    areas, models = describe_storage()
    before = {row['path']: row for row in plan['models']}
    after = {row['path']: row for row in models}
    assert after == {path: row for path, row in before.items() if path not in deleted}
    assert result['logical_bytes_reclaimed'] == sum(before[path]['bytes'] for path in deleted)
    media = {}
    for original, current in zip(plan['areas'], areas, strict=True):
        assert original['path'] == current['path']
        counts = {suffix: current['formats'].get(suffix, 0) for suffix in ('.mp4', '.npz')}
        assert all(original['formats'].get(suffix, 0) == count for suffix, count in counts.items())
        media[current['path']] = counts
    record = dict(status='passed', checked_at=now(), deleted_absent=len(deleted),
                  retained_model_hashes_verified=len(after), media_counts_unchanged=media,
                  plan_sha256=sha(plan_path), result_sha256=sha(result_path))
    write(OUT / 'verification.json', record)
    print(json.dumps({key: value for key, value in record.items() if key != 'media_counts_unchanged'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inventory', 'apply', 'verify'))
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.action == 'inventory':
        make_plan()
    elif args.action == 'verify':
        verify()
    elif args.plan_sha256:
        apply(args.plan_sha256)
    else:
        parser.error('apply requires --plan-sha256')
