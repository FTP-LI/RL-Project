"""Verify delivered offline review media, provenance, links and paused workers.

Read-only with respect to models, training sources and original videos. Writes
only the review verification record. Does not initialize a simulator or policy.
"""
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'records/reports/training-review'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=[];self.text=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        for key in ('src','href'):
            if key in attrs:self.links.append(attrs[key])
        if tag=='article':self.ids.append(attrs.get('id'))
    def handle_data(self,data):self.text.append(data)

def main():
    manifest=read(OUT/'overview-manifest.json');audit=read(OUT/'history-audit.json')
    assert manifest['status']==audit['status']=='passed'
    assert manifest['history_audit_sha256']==sha(OUT/'history-audit.json')
    assert manifest['script_sha256']==sha(ROOT/'scripts/validation/build_gait_video_overview.py')
    assert audit['script_sha256']==sha(ROOT/'scripts/validation/audit_training_history.py')
    for p,h in audit['input_sha256'].items():
        path=Path(p) if p.startswith('/') else ROOT/p
        assert sha(path)==h,p
    original=read(OUT/'decoded-training-videos.json')['clips']+read(OUT/'decoded-evaluation-videos.json')['clips']
    index={c['id']:c for c in original};ids=[p['clip_id'] for p in manifest['panels']]
    assert len(ids)==len(set(ids))==131 and set(ids)==set(index)
    assert sum(len(p['selected_frames']) for p in manifest['panels'])==262
    count=0
    for p in manifest['panels']:
        source=index[p['clip_id']]
        assert sha(Path(p['source']))==p['video_sha256']==source['sha256']
        assert sha(ROOT/p['video'])==source['sha256']
        for frame in p['selected_frames']:
            assert sha(ROOT/frame['file'])==frame['sha256']
            saved=next(f for f in source['frames'] if f['index']==frame['index'])
            assert saved['sha256']==frame['sha256']
            assert abs(frame['time_seconds']-frame['index']/source['fps'])<1e-9
            assert 0<=frame['index']<source['decoded_frames']
            l,t,r,b=frame['crop_source_xyxy'];w,h=source['size']
            assert 0<=l<r<=w and 0<=t<b<=h
            with Image.open(ROOT/frame['file']) as im:im.verify()
            count+=1
    image=ROOT/manifest['image']
    assert sha(image)==manifest['image_sha256']
    with Image.open(image) as im:
        assert list(im.size)==manifest['image_size'];im.load()
    # Verify annotation fit without relying on a scaled preview.
    f=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',22)
    overflow=[p['clip_id'] for p in manifest['panels'] if f.getlength(p['annotation'])>892]
    assert not overflow,overflow
    gallery=ROOT/manifest['gallery'];assert sha(gallery)==manifest['gallery_sha256']
    page=Links();page.feed(gallery.read_text());assert page.ids==ids
    for link in page.links:
        assert '://' not in link
        path=(gallery.parent/link.split('#')[0]).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(),link
    assert not re.search(r'\bV\d+\b|修订稿|版本号',' '.join(page.text))
    paused=read(ROOT/'project.json')['formal_training']['status'].startswith('paused');assert paused
    names={'resume_official.py','resume_extension.py','train.py','train','evaluate_fixed_policy.py',
        'head_command_repair_pilot.py','gait_repair_pilot.py','run_repair_pilot.py',
        'run_head_command_repair.py','profile_saved_run.py'}
    workers=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
        except (OSError,ProcessLookupError):continue
        if any(Path(a).name in names for a in args) and any('XGO-Duck' in a for a in args):
            workers.append(dict(pid=int(p.name),arguments=args))
    assert not workers,workers
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        immutable_audit_inputs_checked=len(audit['input_sha256']),video_contexts=131,selected_frames=count,
        training_contexts=64,evaluation_contexts=67,all_video_contexts_in_image=True,
        source_and_frame_hashes_verified=True,annotation_text_overflow=[],
        gallery_local_references_checked=len(page.links),image_size=manifest['image_size'],
        overview_sha256=sha(image),gallery_sha256=sha(gallery),manifest_sha256=sha(OUT/'overview-manifest.json'),
        main_report_sha256=sha(ROOT/'docs/reports/development_report.md'),training_paused=paused,
        active_training_or_evaluation_workers=workers,training_or_simulation_started=False,
        visual_inspection='All source contact sheets, larger critical frames, header and representative final cards inspected; browser layout not certified')
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
