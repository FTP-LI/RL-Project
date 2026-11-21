"""Verify the already-openable Windows page for the qualified same-process run."""

from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import verify_walk_live_page as live


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    assert platform.system() == 'Windows'
    raw = args.plan.read_bytes()
    spec = json.loads(raw)
    assert spec['status'] == 'frozen-archived-getup-same-process'
    folder, page = root / spec['record'], root / spec['page']
    state, html = (folder / 'state.json').read_bytes(), page.read_bytes()
    current = json.loads(state)
    assert current['status'] == 'starting' and current['controls'] == 0 and current['run_id'] == folder.name
    assert folder.name.encode() in html and b'http-equiv="refresh"' in html
    proof = dict(status='starting-page-observed', observer_host=platform.system(), run_id=folder.name,
                 observed_at=datetime.now().astimezone().isoformat(), protocol_sha256=sha256(raw).hexdigest(),
                 initial_state_sha256=sha256(state).hexdigest(), page_sha256=sha256(html).hexdigest())
    with (folder / 'windows-observer-ready.json').open('x', encoding='utf-8') as file:
        json.dump(proof, file, indent=2)
    print('ARCHIVED_POLICY_WINDOWS_READY', folder.name, flush=True)
    with patch.object(sys, 'argv', [live.__file__, str(folder), str(page), '--timeout', '1000']):
        live.main()
