"""Verify the archived training tree before making a dedicated working copy."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def verify(root: Path, files: list[dict]) -> None:
    for entry in files:
        path = root / entry['path']
        if not path.is_file():
            raise RuntimeError(f'Missing archived file: {path}')
        content = path.read_bytes()
        if len(content) != entry['size_bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise RuntimeError(f'Archive mismatch: {path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    archive = project / 'official_resources'
    manifest = json.loads((archive / 'manifests/xgoduck_rl.files.json').read_text(encoding='utf-8'))
    source = archive / 'xgoduck_rl'
    verify(source, manifest['files'])
    print(f'Archive verified: {len(manifest["files"])} files, commit {manifest["commit"]}')
    if args.verify_only:
        return
    if args.destination is None:
        parser.error('--destination is required unless --verify-only is used')
    destination = args.destination.expanduser().resolve()
    if destination == archive or archive in destination.parents:
        parser.error('The destination must be outside official_resources')
    if destination.exists():
        verify(destination, manifest['files'])
        print(f'Existing baseline copy verified; nothing overwritten: {destination}')
    else:
        shutil.copytree(source, destination)
        verify(destination, manifest['files'])
        print(f'Working copy created and verified: {destination}')
    provenance = {
        'upstream_commit': manifest['commit'],
        'source': str(source),
        'working_copy': str(destination),
        'verified_file_count': len(manifest['files']),
        'uv_lock_sha256': hashlib.sha256((source / 'uv.lock').read_bytes()).hexdigest(),
    }
    (destination.parent / 'training-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
