"""Check saved report hashes and embedded media against the original files.

This validates the standalone HTML artifact, not browser layout or playback.
Simulation videos are decoded separately by verify_fixed_evaluation.py.
"""
import base64
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import mimetypes
from pathlib import Path
import re
import sys


class MediaParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.media = []

    def handle_starttag(self, tag, attrs):
        if tag in ('img', 'video'):
            self.media.append((tag, dict(attrs)))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    source = root/'docs/reports/development_report.md'
    html = source.with_suffix('.html')
    rendered = json.loads((root/'records/reports/development-report-check.json').read_text())
    assert rendered['source_sha256'] == sha(source.read_bytes())
    assert rendered['html_sha256'] == sha(html.read_bytes())
    text = source.read_text()
    pattern = r'!\[[^\]]*\]\(([^)]+)\)|<(video|img)\b[^>]*\bsrc="([^"]+)"'
    expected = []
    for match in re.finditer(pattern, text):
        path = match.group(1) or match.group(3)
        kind = 'img' if match.group(1) else match.group(2)
        file = (source.parent/path).resolve()
        assert file.is_relative_to(root) and file.is_file(), path
        expected.append((kind, file))
    parser = MediaParser()
    parser.feed(html.read_text())
    assert len(parser.media) == len(expected) > 0
    checks = []
    for (kind, file), (actual_kind, attrs) in zip(expected, parser.media):
        assert actual_kind == kind
        original = file.read_bytes()
        if attrs['src'].startswith('data:'):
            prefix, data = attrs['src'].split(',', 1)
            assert prefix == f'data:{mimetypes.guess_type(file.name)[0]};base64'
            assert base64.b64decode(data, validate=True) == original
        else:
            assert (html.parent/attrs['src']).resolve() == file
            assert rendered['media_mode'] == 'relative-files-on-demand'
        if kind == 'video':
            assert 'controls' in attrs and attrs.get('preload') == 'none'
        else:
            assert attrs.get('loading') == 'lazy'
        checks.append(dict(kind=kind, file=str(file.relative_to(root)),
                           bytes=len(original), sha256=sha(original)))
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  command=[sys.executable, *sys.argv], exit_code=0,
                  source_sha256=sha(source.read_bytes()), html_sha256=sha(html.read_bytes()),
                  images=sum(c['kind']=='img' for c in checks),
                  videos=sum(c['kind']=='video' for c in checks), media=checks,
                  browser_layout_and_playback_verified=False,
                  media_mode=rendered['media_mode'],
                  scope='Saved HTML resolves the intended original media with controls and on-demand loading; not browser rendering')
    (root/'records/reports/report-media-check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='media'}, indent=2))


if __name__ == '__main__':
    main()
