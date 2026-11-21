"""Check project navigation and original media after report organization.

Reads files only; does not run historical scripts, simulation or a browser.
Immutable byte backups are evidence snapshots, not relocated reading copies.
"""
import ast
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import markdown

from s1_reference_inputs import ROOT, read, sha


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.ids=set(); self.media=[]

    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if 'id' in attrs: self.ids.add(attrs['id'])
        if 'href' in attrs: self.links.append(attrs['href'])
        if 'src' in attrs: self.links.append(attrs['src'])
        if tag in ('img','video'): self.media.append((tag,attrs))


def parsed(path):
    text=path.read_text(encoding='utf-8-sig')
    if path.suffix=='.md': text=markdown.markdown(text,extensions=['tables','fenced_code','toc'])
    out=Links();out.feed(text);return out


def main():
    out=ROOT/'records/reports/project-organization'
    files=[ROOT/n for n in ['README.md','scripts/README.md','configs/README.md','records/README.md',
        'docs/project_structure.md','docs/current_goal.md','docs/runbooks/pc_training.md',
        'docs/runbooks/hardware_commissioning.md',
        'docs/runbooks/skill_switching.md',
        'docs/reports/development_report.md','docs/reports/development_report.html',
        'docs/reports/development_history.md','docs/reports/development_history.html']]
    checks=[];media=[];missing=[];cache={}
    for file in files:
        doc=parsed(file);cache[file.resolve()]=doc
        for link in doc.links:
            if re.match(r'^[a-zA-Z]+:',link) or link.startswith('//'):continue
            relative, _, anchor=link.partition('#')
            target=(file.parent/relative).resolve() if relative else file.resolve()
            if not target.exists():
                missing.append(dict(source=str(file.relative_to(ROOT)),target=link));continue
            if anchor and target.suffix in ('.md','.html'):
                target_doc=cache.setdefault(target,parsed(target))
                if anchor not in target_doc.ids:missing.append(dict(source=str(file.relative_to(ROOT)),target=link,reason='missing-anchor'))
            checks.append(dict(source=str(file.relative_to(ROOT)),target=link))
        if file.suffix=='.html':
            for kind,attrs in doc.media:
                target=(file.parent/attrs['src']).resolve()
                assert target.is_relative_to(ROOT) and target.is_file()
                assert not attrs['src'].startswith('data:')
                if kind=='video':assert 'controls' in attrs and attrs.get('preload')=='none'
                if kind=='img':assert attrs.get('loading')=='lazy'
                media.append(dict(document=str(file.relative_to(ROOT)),kind=kind,
                    path=str(target.relative_to(ROOT)),sha256=sha(target),bytes=target.stat().st_size))
    inventory=[]
    for folder in ('scripts','configs','docs'):
        for file in sorted((ROOT/folder).rglob('*')):
            if not file.is_file() or '__pycache__' in file.parts:continue
            if file.suffix=='.py':ast.parse(file.read_text(encoding='utf-8-sig'),filename=str(file))
            if file.suffix=='.json':read(file)
            inventory.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=sha(file)))
    tasks=read(ROOT/'.vscode/tasks.json')
    assert tasks['presentation']['panel']=='shared' and tasks['presentation']['reveal']=='always'
    assert len({t['label'] for t in tasks['tasks']})==len(tasks['tasks'])
    assert all(t['runOptions']['instanceLimit']==1 for t in tasks['tasks'])
    before=read(out/'organization.json')
    assert sha(out/'source-report.md')==before['original_markdown_sha256']
    result=dict(status='failed' if missing else 'passed',checked_at=datetime.now(timezone.utc).isoformat(),
        files_checked=len(files),local_references=len(checks),missing=missing,
        media=media,inventory=inventory,daily_tasks=len(tasks['tasks']),
        current_main_html_bytes=(ROOT/'docs/reports/development_report.html').stat().st_size,
        current_history_html_bytes=(ROOT/'docs/reports/development_history.html').stat().st_size,
        original_main_html_bytes=before['original_html_bytes'],
        original_report_backup_verified=True,training_evidence_deleted=False,
        browser_visual_verification=False,
        browser_limit='In-app browser rejected file protocol; no bypass or alternative browser used',
        scope='Navigation, original media resolution, Python syntax, JSON and task sharing; no GUI layout or gait claim')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inventory','media')},indent=2))
    if missing:raise SystemExit(1)


if __name__=='__main__':main()
