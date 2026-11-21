"""Observe the initial page on Windows before allowing each CUDA case to start."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import verify_walk_live_page as live


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/experiments/common_walk_forward_model_routing.json"


def main():
    assert platform.system() == "Windows"
    protocol = PLAN.read_bytes()
    spec = json.loads(protocol)
    assert spec["status"] == "frozen-native-forward-model-routing"
    for key, item in spec["cases"].items():
        folder, page = (ROOT / item[name] for name in ("record", "page"))
        initial, html = (folder / "state.json").read_bytes(), page.read_bytes()
        view = json.loads(initial)
        assert view["status"] == "starting" and view["controls"] == 0
        assert view["run_id"] == folder.name and folder.name.encode() in html and b"http-equiv=\"refresh\"" in html
        target = folder / "windows-observer-ready.json"
        assert not target.exists()
        proof = dict(status="starting-page-observed", observer_host=platform.system(),
                     observed_at=datetime.now().astimezone().isoformat(), run_id=folder.name,
                     protocol_sha256=sha256(protocol).hexdigest(), initial_state_sha256=sha256(initial).hexdigest(),
                     page_sha256=sha256(html).hexdigest())
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(proof, indent=2)+"\n", encoding="utf-8")
        temporary.replace(target)
        print("WINDOWS_WALK_ROUTING_READY", key, flush=True)
        with patch.object(sys, "argv", [live.__file__, str(folder), str(page), "--timeout", "420"]):
            try:
                live.main()
            except SystemExit as exit_status:
                assert exit_status.code in (0, 1)
        proof = json.loads((folder / "windows-view-verification.json").read_text(encoding="utf-8"))
        assert proof["observations"][0]["status"] == "starting"
        assert proof["observations"][-1]["status"] == "completed", key
        assert proof["running_snapshots"] >= 3 and proof["running_snapshots_with_original_frame"] > 0, key


if __name__ == "__main__":
    main()
