"""Build and verify a manifest for the project-local training record archive."""

import argparse
import hashlib
import json
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / 'records'
ARCHIVE = RECORDS / 'training-archive'
AREAS = ('experiments', 'evaluations', 'benchmarks', 'profiling')


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def inventory(directory):
    if directory.resolve(strict=True) != directory or not directory.is_dir():
        raise ValueError(f'Expected a physical archive directory: {directory}')
    entries = []
    for path in sorted(directory.rglob('*')):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f'Unexpected nested link: {path}')
        if stat.S_ISREG(info.st_mode):
            entries.append(dict(path=path.relative_to(directory).as_posix(), bytes=info.st_size, sha256=digest(path)))
        elif not stat.S_ISDIR(info.st_mode):
            raise ValueError(f'Unexpected archive entry: {path}')
    return entries


def write(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('x', encoding='utf-8', newline='\n') as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write('\n')
    temp.replace(path)


def main(action):
    manifest = ARCHIVE / 'manifest.json'
    if action == 'plan':
        if manifest.exists() or any((ARCHIVE / name).exists() for name in AREAS):
            raise ValueError('An existing archive must keep its original manifest')
        ARCHIVE.mkdir(exist_ok=True)
        if ARCHIVE.resolve(strict=True) != ARCHIVE:
            raise ValueError('Expected a physical archive directory')
        groups = {name: inventory(RECORDS / name) for name in AREAS}
        write(manifest, dict(groups=groups, scope='Archive-time files only. New runs may be filed through the compatibility paths.'))
    else:
        groups = json.loads(manifest.read_text(encoding='utf-8'))['groups']
        assert tuple(groups) == AREAS
        for name, archived in groups.items():
            physical, alias = ARCHIVE / name, RECORDS / name
            if not alias.samefile(physical):
                raise ValueError(f'Compatibility path is not bound to the archive: {alias}')
            actual = {entry['path']: entry for entry in inventory(physical)}
            assert all(actual.get(entry['path']) == entry for entry in archived), name
        write(ARCHIVE / 'verification.json', dict(status='passed', manifest_sha256=digest(manifest),
                                                  files_verified=sum(map(len, groups.values())),
                                                  same_file_compatibility_paths=list(groups)))
    print(json.dumps(dict(action=action, manifest_sha256=digest(manifest),
                          groups={name: dict(files=len(rows), bytes=sum(row['bytes'] for row in rows))
                                  for name, rows in groups.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'verify'))
    main(parser.parse_args().action)
