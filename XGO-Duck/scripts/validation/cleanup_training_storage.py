"""Inventory first, then delete only listed superseded unreferenced models.

Preserve raw logs, videos, final checkpoints, audit inputs and official trees.
The huge profiler timeline is losslessly compressed and hash-verified before
removing its plain JSON. No recursive deletion or guessed directory removal.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
LOGS = Path.home()/'XGO-Duck/workspace/xgoduck_rl/logs/rsl_rl'
OUT = ROOT/'records/reports/storage-cleanup'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def inventory():
    OUT.mkdir(exist_ok=True)
    # Search complete project text evidence, excluding large binary/media and
    # previous cleanup outputs. Match filenames conservatively across runs.
    refs=set(); text_files=0
    for base in ('docs','scripts','configs','records','artifacts'):
        for p in (ROOT/base).rglob('*'):
            if not p.is_file() or p.suffix not in ('.json','.jsonl','.md','.py','.sh','.yaml','.txt','.csv'):continue
            if p.is_relative_to(OUT) or p.stat().st_size>10_000_000:continue
            text_files+=1
            refs.update(re.findall(r'model_\d+\.(?:pt|onnx)',p.read_text(errors='replace')))
    refs.update(re.findall(r'model_\d+\.(?:pt|onnx)',(ROOT/'project.json').read_text()))
    rows=[]
    for run in sorted(LOGS.glob('*/*')):
        if not run.is_dir():continue
        ckpts=sorted(run.glob('model_*.pt'),key=lambda p:int(p.stem[6:]))
        if not ckpts:continue
        final=ckpts[-1]
        for p in ckpts[:-1]:
            resolved=p.resolve()
            if p.is_symlink() or not resolved.is_relative_to(LOGS.resolve()):raise ValueError(str(p))
            if p.name in refs:continue
            rows.append(dict(path=str(resolved),bytes=p.stat().st_size,sha256=sha(p),
                             reason='Superseded intermediate checkpoint; no project text reference',
                             retained_later_checkpoint=str(final),retained_sha256=sha(final)))
    trace=ROOT/'records/profiling/profile_20261101T151652_s42/traced/trace.json'
    compressed=[]
    if trace.is_file():
        compressed.append(dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace),
                               target=str(trace)+'.gz',reason='Lossless archival of large profiling timeline'))
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),text_files_checked=text_files,
              protected_model_names=sorted(refs),delete=rows,compress=compressed,
              delete_bytes=sum(x['bytes'] for x in rows),status='inventoried')
    write(OUT/'plan.json',plan)
    print(json.dumps({k:v for k,v in plan.items() if k not in ('protected_model_names','delete')},indent=2))


def apply():
    plan=json.loads((OUT/'plan.json').read_text())
    audit=json.loads((ROOT/'records/reports/training-review/history-audit.json').read_text())
    protected=set(audit['input_sha256'])
    result=dict(started_at=datetime.now(timezone.utc).isoformat(),deleted=[],compressed=[])
    # Validate every absolute target and successor before any destructive write.
    for row in plan['delete']:
        p=Path(row['path']);keep=Path(row['retained_later_checkpoint'])
        if (p.is_symlink() or not p.resolve().is_relative_to(LOGS.resolve()) or str(p) in protected
                or not re.fullmatch(r'model_\d+\.pt',p.name)):
            raise ValueError('Deletion outside reviewed intermediate set: '+str(p))
        if sha(p)!=row['sha256'] or sha(keep)!=row['retained_sha256']:raise ValueError('Changed source')
    for row in plan['compress']:
        p=Path(row['path']);out=Path(row['target'])
        if not p.resolve().is_relative_to((ROOT/'records/profiling').resolve()) or p.is_symlink():raise ValueError('Bad trace path')
        if sha(p)!=row['sha256'] or out.exists():raise ValueError('Changed/already compressed source')
    write(OUT/'result.json',result)
    for row in plan['compress']:
        p=Path(row['path']);out=Path(row['target'])
        with p.open('rb') as src,gzip.open(out,'wb',compresslevel=6) as dst:
            for b in iter(lambda:src.read(1024*1024),b''):dst.write(b)
        h=hashlib.sha256();size=0
        with gzip.open(out,'rb') as src:
            for b in iter(lambda:src.read(1024*1024),b''):h.update(b);size+=len(b)
        if h.hexdigest()!=row['sha256'] or size!=row['bytes']:raise ValueError('Archive verification failed')
        detail=dict(**row,compressed_sha256=sha(out),compressed_bytes=out.stat().st_size,
                    decompressed_hash_verified=True)
        result['compressed'].append(detail);write(OUT/'result.json',result)
        p.unlink()
    for row in plan['delete']:
        Path(row['path']).unlink()
        result['deleted'].append(row);write(OUT/'result.json',result)
    assert all(not Path(x['path']).exists() for x in result['deleted']+result['compressed'])
    result.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),
        logical_bytes_reclaimed=sum(x['bytes'] for x in result['deleted'])+
            sum(x['bytes']-x['compressed_bytes'] for x in result['compressed']),
        original_videos_and_evaluation_traces_deleted=0,
        wsl_virtual_disk_compacted=False)
    write(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('deleted','compressed')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inventory','apply']);a=p.parse_args()
    inventory() if a.action=='inventory' else apply()
