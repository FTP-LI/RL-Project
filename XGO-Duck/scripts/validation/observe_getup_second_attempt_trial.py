"""Independently read the starting and running second-attempt page on Windows."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import verify_walk_live_page as live


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/"configs/experiments/getup_supine_second_attempt_trial.json"


def main():
    assert platform.system() == "Windows"
    raw = PLAN.read_bytes()
    spec = json.loads(raw)
    assert spec["status"] == "frozen-getup-second-attempt-trial"
    record,page = ROOT/spec["record"],ROOT/spec["page"]
    data,html = (record/"state.json").read_bytes(),page.read_bytes()
    state = json.loads(data)
    assert state["status"] == "starting" and state["controls"] == 0 and state["run_id"] == record.name
    assert record.name.encode() in html and b'http-equiv="refresh"' in html
    target = record/"windows-observer-ready.json"
    assert not target.exists()
    proof = dict(status="starting-page-observed",observer_host=platform.system(),run_id=record.name,
        observed_at=datetime.now().astimezone().isoformat(),protocol_sha256=sha256(raw).hexdigest(),
        initial_state_sha256=sha256(data).hexdigest(),page_sha256=sha256(html).hexdigest())
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
    temporary.replace(target)
    print("GETUP_SECOND_ATTEMPT_WINDOWS_READY",record.name,flush=True)
    with patch.object(sys,"argv",[live.__file__,str(record),str(page),"--timeout","660"]):
        live.main()


if __name__ == "__main__":
    main()
