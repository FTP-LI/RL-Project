"""Verify the starting and moving recovery view from Windows."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import verify_walk_live_page as live


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/"configs/experiments/getup_supine_left_hip_load_capture.json"


def main():
    assert platform.system() == "Windows"
    raw = PLAN.read_bytes()
    spec = json.loads(raw)
    assert spec["status"] == "frozen-native-left-hip-load-capture"
    record,page = ROOT/spec["record"],ROOT/spec["page"]
    data,html = (record/"state.json").read_bytes(),page.read_bytes()
    initial = json.loads(data)
    assert initial["status"] == "starting" and initial["controls"] == 0
    assert initial["run_id"] == record.name and record.name.encode() in html and b'http-equiv="refresh"' in html
    target = record/"windows-observer-ready.json"
    assert not target.exists()
    proof = dict(status="starting-page-observed",observer_host=platform.system(),run_id=record.name,
        observed_at=datetime.now().astimezone().isoformat(),protocol_sha256=sha256(raw).hexdigest(),
        initial_state_sha256=sha256(data).hexdigest(),page_sha256=sha256(html).hexdigest())
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
    temp.replace(target)
    print("LEFT_HIP_LOAD_WINDOWS_READY",record.name,flush=True)
    with patch.object(sys,"argv",[live.__file__,str(record),str(page),"--timeout","660"]):
        live.main()


if __name__ == "__main__":
    main()
