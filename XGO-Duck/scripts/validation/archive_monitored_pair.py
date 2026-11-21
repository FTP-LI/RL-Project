"""Snapshot the completed monitor comparison; never delete or rerun a job.

Large assets stay at canonical physical paths and are listed with SHA-256.
The small ZIP is read back member by member; prior archives are not changed.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'records/reports/getup-monitored-authorization-20261116'
PLAN = ROOT/'configs/experiments/observed_getup_monitor_pair_20261116.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def inventory(path):
    path = path.resolve()
    assert path.is_file() and path.is_relative_to(ROOT.resolve()), path
    return {'path':path.relative_to(ROOT.resolve()).as_posix(), 'bytes':path.stat().st_size, 'sha256':sha(path)}


def main():
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    result=json.loads((REPORT/'comparison.json').read_text(encoding='utf-8'))
    assert result['status']=='completed-monitored-runtime-review' and sha(PLAN)==result['protocol_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in plan['source_sha256'].items())
    archive=REPORT/'stage-snapshot.zip'; manifest=REPORT/'stage-snapshot-manifest.json'
    external_file=REPORT/'stage-external-assets.json'; check=REPORT/'stage-verification.json'
    assert not any(p.exists() for p in (archive,manifest,external_file,check))
    paths={(ROOT/name).resolve() for name in plan['source_sha256']}
    for folder in [REPORT,ROOT/'artifacts/delivery'] + [ROOT/'records/experiments'/spec['run_id'] for spec in plan['arms'].values()] + [ROOT/'artifacts'/spec['run_id'] for spec in plan['arms'].values()]:
        paths.update(p.resolve() for p in folder.rglob('*') if p.is_file())
    for name in ['README.md','project.json','docs/current_goal.md','docs/reports/development_report.md',
                 'docs/reports/development_report.html','docs/reports/development_history.md',
                 'docs/reports/development_history.html','records/training-current.md','records/training-current.json',
                 'scripts/validation/review_monitored_getup_pair.py',
                 'scripts/validation/compose_monitor_pair_contact_sheet.py','scripts/validation/archive_monitored_pair.py',
                 'records/reports/pause-archive-20261116-monitored/snapshot.zip']:
        paths.add((ROOT/name).resolve())
    paths.update(p.resolve() for p in (ROOT/'configs/experiments').glob('observed_getup_monitor_*.json'))
    small,large=[],[]
    for path in sorted(paths):
        item=inventory(path)
        (large if path.suffix.lower() in {'.pt','.onnx','.npz','.npy','.mp4','.zip'} or path.stat().st_size>3_000_000 else small).append(item)
    write(external_file,{'scope':'Original models, trajectories, videos and prior snapshots remain at these paths; preserve the complete XGO-Duck project.','files':large})
    small.append(inventory(external_file))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for item in small:z.write(ROOT/item['path'],item['path'])
    write(manifest,{'created_at':datetime.now(timezone.utc).isoformat(),'snapshot_sha256':sha(archive),
                    'files':small,'external_assets':'stage-external-assets.json',
                    'standalone_backup':False,'original_files_deleted':0})
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(small)
        for item in small:
            data=z.read(item['path'])
            assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    for item in large:
        assert sha(ROOT/item['path'])==item['sha256']
    proof={'status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),
           'zip_files_checked':len(small),'zip_bytes':archive.stat().st_size,'zip_sha256':sha(archive),
           'external_files_checked':len(large),'external_bytes':sum(x['bytes'] for x in large),
           'frozen_sources_checked':len(plan['source_sha256']),'original_files_deleted':0,
           'new_physical_steps':0,'new_learning_updates':0,'hardware_io':False}
    write(check,proof);print(json.dumps(proof))


if __name__=='__main__':
    main()
