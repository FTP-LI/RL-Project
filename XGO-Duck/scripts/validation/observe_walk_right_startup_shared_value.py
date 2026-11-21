"""Independently witness the CUDA sampling, value metrics and original clip from Windows."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform
import re
from time import monotonic, sleep


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/experiments/common_walk_right_startup_shared_value.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    temp.replace(path)


def main():
    assert platform.system() == "Windows"
    spec = read(PLAN)
    assert spec["status"] == "frozen-native-right-turn-shared-value"
    record, page = (ROOT / spec[key] for key in ("record", "page"))
    start, html = (record / "state.json").read_bytes(), page.read_bytes()
    state = json.loads(start)
    assert state["status"] == "starting" and not state["controls"]
    assert state["run_id"] == record.name and record.name.encode() in html and b'http-equiv="refresh"' in html
    ready = record / "windows-observer-ready.json"
    assert not ready.exists()
    write(ready, dict(status="starting-page-observed", observer_host=platform.system(), run_id=record.name,
        protocol_sha256=sha256(PLAN.read_bytes()).hexdigest(), initial_state_sha256=sha256(start).hexdigest(),
        page_sha256=sha256(html).hexdigest(), observed_at=datetime.now().astimezone().isoformat()))
    print("RIGHT_STARTUP_VALUE_WINDOWS_READY", flush=True)
    seen = {}
    deadline = monotonic()+580
    while monotonic() < deadline:
        try:
            state = read(record / "state.json")
            html = page.read_text(encoding="utf-8")
            assert state["run_id"] in html
            key = state["status"], state["controls"], state["learning_updates"]
            if key[2]:
                curve = re.search(r'id="value-fit" data-update="(\d+)"', html)
                if not curve or int(curve[1]) != key[2]:
                    sleep(.25)
                    continue
            if key not in seen:
                row = dict(status=key[0], controls=key[1], learning_updates=key[2], throughput_points=len(state["updates"]),
                    auto_refresh='http-equiv="refresh"' in html, observed_at=datetime.now().astimezone().isoformat())
                for name in ("frame", "latest_clip"):
                    media = state.get(name)
                    if media and media in html:
                        row[name] = sha256((record / media).read_bytes()).hexdigest()
                seen[key] = row
        except (OSError, ValueError):
            sleep(.25)
            continue
        if key[0] in {"completed", "stopped-review-required"}:
            break
        sleep(.7)
    observations = list(seen.values())
    running = [row for row in observations if row["status"] == "running"]
    fitted = sorted({row["learning_updates"] for row in observations if row["learning_updates"]})
    passed = (observations and observations[0]["status"] == "starting" and observations[-1]["status"] == "completed"
              and observations[-1]["learning_updates"] == spec["critic_updates_authorized"]
              and "latest_clip" in observations[-1] and len(running) >= 3
              and any("frame" in row and row["auto_refresh"] for row in running)
              and any(0 < update < spec["critic_updates_authorized"] for update in fitted))
    write(record / "windows-view-verification.json", dict(status="passed" if passed else "review-required",
        observer_host=platform.system(), running_snapshots=len(running), observed_fit_updates=fitted, observations=observations))
    print("RIGHT_STARTUP_VALUE_WINDOWS_OBSERVED", passed, len(running), fitted, flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
