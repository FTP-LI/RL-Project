"""Independently confirm the hard-start page and observe live updates on Windows."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import verify_walk_live_page as live


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/"configs/experiments/getup_supine_hard_start_screen.json"


def main():
    assert platform.system() == "Windows"
    raw = PLAN.read_bytes()
    spec = json.loads(raw)
    assert spec["status"] == "frozen-hard-start-parent-screen"
    folder,page = ROOT/spec["record"],ROOT/spec["page"]
    data,html = (folder/"state.json").read_bytes(),page.read_bytes()
    state = json.loads(data)
    assert state["status"] == "starting" and state["controls"] == 0 and state["run_id"] == folder.name
    assert folder.name.encode() in html and b'http-equiv="refresh"' in html
    path = folder/"windows-observer-ready.json"
    assert not path.exists()
    proof = dict(status="starting-page-observed",observer_host=platform.system(),run_id=folder.name,
        observed_at=datetime.now().astimezone().isoformat(),protocol_sha256=sha256(raw).hexdigest(),
        initial_state_sha256=sha256(data).hexdigest(),page_sha256=sha256(html).hexdigest())
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
    temp.replace(path)
    print("GETUP_HARD_START_WINDOWS_READY",folder.name,flush=True)
    with patch.object(sys,"argv",[live.__file__,str(folder),str(page),"--timeout","620"]):
        live.main()


if __name__ == "__main__":
    main()
