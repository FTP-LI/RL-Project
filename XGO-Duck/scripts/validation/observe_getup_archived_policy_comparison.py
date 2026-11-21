"""Verify the Windows observation page before and during archived-policy evaluation."""

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
    parser.add_argument('seed', type=int)
    seed = parser.parse_args().seed
    assert platform.system() == 'Windows'
    root = Path(__file__).resolve().parents[2]
    raw = (root / f'configs/experiments/getup_archived_policy_comparison_s{seed}.json').read_bytes()
    spec = json.loads(raw)
    assert spec['status'] == 'frozen-archived-getup-same-release'
    record, page = root / spec['record'], root / spec['page']
    raw_state, html = (record / 'state.json').read_bytes(), page.read_bytes()
    state = json.loads(raw_state)
    assert state['status'] == 'starting' and state['controls'] == 0 and state['run_id'] == record.name
    assert record.name.encode() in html and b'http-equiv="refresh"' in html
    proof = dict(status='starting-page-observed', observer_host=platform.system(), run_id=record.name,
                 observed_at=datetime.now().astimezone().isoformat(), protocol_sha256=sha256(raw).hexdigest(),
                 initial_state_sha256=sha256(raw_state).hexdigest(), page_sha256=sha256(html).hexdigest())
    with (record / 'windows-observer-ready.json').open('x', encoding='utf-8') as file:
        json.dump(proof, file, indent=2)
    print('ARCHIVED_POLICY_WINDOWS_READY', record.name, flush=True)
    with patch.object(sys, 'argv', [live.__file__, str(record), str(page), '--timeout', '1000']):
        live.main()
