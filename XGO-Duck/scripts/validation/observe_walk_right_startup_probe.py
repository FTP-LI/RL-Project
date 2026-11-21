"""Read both right-turn run pages on Windows and witness their real progress."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import verify_walk_live_page as live


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/experiments/common_walk_right_startup_probe.json"


def main():
    assert platform.system() == "Windows"
    protocol = PLAN.read_bytes()
    spec = json.loads(protocol)
    assert spec["status"] == "frozen-native-right-turn-probe"
    for role, item in spec["cases"].items():
        record, page = (ROOT / item[key] for key in ("record", "page"))
        initial, html = (record / "state.json").read_bytes(), page.read_bytes()
        state = json.loads(initial)
        assert state["status"] == "starting" and state["controls"] == 0
        assert state["run_id"] == record.name and record.name.encode() in html and b'http-equiv="refresh"' in html
        target = record / "windows-observer-ready.json"
        assert not target.exists()
        proof = dict(status="starting-page-observed", observer_host=platform.system(),
            observed_at=datetime.now().astimezone().isoformat(), run_id=record.name,
            protocol_sha256=sha256(protocol).hexdigest(), initial_state_sha256=sha256(initial).hexdigest(),
            page_sha256=sha256(html).hexdigest())
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(proof, indent=2)+"\n", encoding="utf-8")
        temporary.replace(target)
        print("RIGHT_STARTUP_WINDOWS_READY", role, flush=True)
        with patch.object(sys, "argv", [live.__file__, str(record), str(page), "--timeout", "230"]):
            try:
                live.main()
            except SystemExit as status:
                assert status.code == 0, role


if __name__ == "__main__":
    main()
