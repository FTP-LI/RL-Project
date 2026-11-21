"""Test the real collector adapter with archived states and a fake environment."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
from guarded_roll_collector import make_guarded_collector
from roll_stand_walk_sequence import STATE_FIELDS

OUT = ROOT / "records/reports/roll-domain-parameter-random-s20"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ActorObservation:
    def __init__(self, worlds):
        self.worlds = worlds

    def clone(self):
        return np.zeros((self.worlds, 61), dtype=np.float32)


class Robot:
    def find_joints(self, names, preserve_order):
        return list(range(len(names))), list(names)

    def find_sites(self, names, preserve_order):
        return [0, 1], list(names)

    def find_bodies(self, name):
        return [0], [name]


class FakeEnvironment:
    def __init__(self, worlds):
        self.num_envs = worlds
        self.scene = {"robot": Robot()}
        self.common_step_counter = 0
        self._sim_step_counter = 0
        self.observation = {"actor": ActorObservation(worlds)}
        self.term = SimpleNamespace(_processed_actions=None)
        self.action_manager = SimpleNamespace(get_term=lambda name: self.term)
        self.fail_after_step = None

    def step(self, action):
        self.common_step_counter += 1
        self._sim_step_counter += 5
        if self.common_step_counter == self.fail_after_step:
            raise RuntimeError("injected post-step failure")
        self.term._processed_actions = action
        flags = np.zeros(self.num_envs, dtype=bool)
        return self.observation, None, flags, flags.copy(), None


class FakeEndpoint:
    def __init__(self):
        self.rows = 1
        self.first_mismatches = []
        self.saved = False

    def capture(self):
        self.rows += 1

    def save(self, maximum):
        self.saved = True
        self.maximum = maximum


class FakeSnapshot:
    def __init__(self, env, folder):
        self.env = env
        self.closed = False

    def capture(self):
        return {"file": f"frame-step-{self.env.common_step_counter}.jpg"}

    def close(self):
        self.closed = True


class FakeVideo:
    def __init__(self, snapshot, purpose, folder, seconds):
        self.writer = object()
        self.clips = []
        self.closed = False
        self.samples = 0

    def sample(self):
        self.samples += 1

    def _sample_terminal(self):
        self.samples += 1

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self, plan, networks, homes, sim_home, limits):
        self.worlds = plan["worlds"]

    def next(self, step, observation, sequence):
        target = np.zeros((self.worlds, 14), dtype=np.float32)
        return target, {"mock_actions": target.copy()}


def run_case(name, plan, trace, destination):
    env = FakeEnvironment(plan["worlds"])
    if name == "advance-error":
        env.fail_after_step = 9
    endpoint = FakeEndpoint()
    instances = {}
    published = []

    def make_snapshot(e, folder):
        instances["snapshot"] = FakeSnapshot(e, folder)
        return instances["snapshot"]

    def make_video(snapshot, purpose, folder, seconds):
        instances["video"] = FakeVideo(snapshot, purpose, folder, seconds)
        return instances["video"]

    def capture(env, endpoint, alive, joints, head, feet, limits, tolerance):
        step = env.common_step_counter
        frame = {key: np.array(trace[key][step], copy=True) for key in STATE_FIELDS}
        if name == "head-safety" and step == 5:
            frame["head_penetration_m"][0] = 0.006
        if name == "bad-quaternion" and step == 5:
            frame["quaternion"][0] = 0
        if name == "nonfinite" and step == 5:
            frame["height"][0] = np.nan
        return frame

    def write(path, data):
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def publish(folder, state):
        if name == "observer-error" and state["controls"] >= 50:
            raise OSError("injected observer failure")
        published.append(deepcopy(state))

    live = SimpleNamespace(PAGE=None, publish=publish)
    fake_base = SimpleNamespace(
        cpu=lambda value: np.asarray(value), PolicyControl=FakePolicy, Snapshot=make_snapshot,
        ActionVideo=make_video, capture=capture, datetime=datetime, TZ=timezone.utc,
        patch=patch, live=live, PAGE=destination / "unused-page.html", write=write,
        torch=SimpleNamespace(testing=SimpleNamespace(assert_close=lambda a, b, **kwargs: np.testing.assert_allclose(a, b))),
    )
    keep_running = lambda: not (name == "pause" and env.common_step_counter >= 7)
    collector = make_guarded_collector(fake_base, keep_running)
    destination.mkdir()
    result = collector(env, env.observation, plan, destination, endpoint, {}, {},
                       np.zeros((plan["worlds"], 14)), np.zeros((14, 2)))
    assert read(destination / "guarded-evaluation.json") == result
    assert instances["video"].closed and instances["snapshot"].closed and endpoint.saved
    assert (destination / "trace.npz").is_file() and (destination / "commands.npz").is_file()
    with np.load(destination / "trace.npz", allow_pickle=False) as saved:
        assert saved["active"].shape[0] == result["saved_states"]
    return result, env, endpoint, published


def main():
    progress = read(OUT / "progress.json")
    review = read(OUT / "offline-review.json")
    item, source = progress["results"][0], review["arms"][0]
    plan = read(ROOT / item["protocol"])
    trace_path = ROOT / item["run"] / "trace.npz"
    assert sha(trace_path) == source["source_sha256"][trace_path.relative_to(ROOT).as_posix()]
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in STATE_FIELDS}
    results = []
    with tempfile.TemporaryDirectory(prefix="guarded-collector-test-", dir=OUT) as temp:
        root = Path(temp)
        for name, expected_reason, expected_controls in [
            ("unreachable-roll-gate", "roll-gate-impossible", 500),
            ("head-safety", "head-penetration-exceeded", 5),
            ("bad-quaternion", "sequence-or-guard-error", 5),
            ("nonfinite", "invalid-or-nonfinite-state", 5),
            ("pause", "pause-requested", 7),
        ]:
            result, env, endpoint, published = run_case(name, plan, trace, root / name)
            assert result["status"] == "guarded-stop-review-required" and result["stop_reason"] == expected_reason
            assert result["saved_controls"] == env.common_step_counter == expected_controls
            assert result["saved_states"] == endpoint.rows == expected_controls + 1
            assert result["actual_physics_substeps"] == expected_controls * 5
            assert not result["artifact_errors"] and published[-1]["status"] == "guarded-stop-review-required"
            assert not (root / name / "raw-score.json").exists()
            if name == "head-safety":
                assert result["safety_summary"]["head_over_limit_worlds"] == 1
                assert not result["safety_summary"]["trace_valid"]
            if name == "bad-quaternion":
                assert not result["safety_summary"]["trace_valid"]
            if name == "nonfinite":
                assert not result["safety_summary"]["trace_valid"]
                assert result["safety_summary"]["nonfinite_samples"] == 1
            results.append({"case": name, "reason": result["stop_reason"], "controls": result["saved_controls"],
                            "states": result["saved_states"], "no_next_step": True})
        result, env, endpoint, published = run_case("advance-error", plan, trace, root / "advance-error")
        assert result["stop_reason"] == "advance-error" and result["saved_controls"] == 8
        assert env.common_step_counter == result["actual_control_steps"] == 9
        assert result["actual_physics_substeps"] == 45 and result["saved_states"] == 9
        assert result["guard_result"]["actual_physical_steps"] == "unknown-read-caller-counter"
        results.append({"case": "advance-error", "reason": result["stop_reason"], "controls": 9,
                        "recorded_controls": 8, "states": 9, "no_next_step": True})
        result, env, endpoint, published = run_case("observer-error", plan, trace, root / "observer-error")
        assert result["stop_reason"] == "state-record-or-observer-error" and env.common_step_counter == 50
        assert result["saved_controls"] == 50 and result["saved_states"] == 51
        results.append({"case": "observer-error", "reason": result["stop_reason"], "controls": 50,
                        "states": 51, "no_next_step": True})
        short = deepcopy(plan)
        short["maximum_controls"] = 10
        result, env, endpoint, published = run_case("short-complete", short, trace, root / "short-complete")
        assert result["status"] == "collected-review-required" and result["saved_controls"] == 10
        assert (root / "short-complete" / "raw-score.json").is_file()
        results.append({"case": "short-complete-mock", "reason": None, "controls": 10,
                        "states": result["saved_states"], "no_next_step": True})
    report = {
        "status": "passed-fake-environment-collector-runtime-gpu-pending",
        "checked_at": datetime.now(timezone.utc).isoformat(), "cases": results,
        "new_physics_steps": 0, "learning_updates": 0, "hardware_io": False,
        "torch_imported": "torch" in sys.modules,
        "scope": "The real collector adapter used archived states and fake environment, policy, video and observer objects; actual CUDA, model outputs, factor injection and S17 runtime wiring remain unverified.",
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in [
            Path(__file__), ROOT / "scripts/validation/guarded_roll_collector.py",
            ROOT / "scripts/validation/guarded_roll_loop.py", ROOT / "scripts/validation/roll_screen_stop_guard.py",
            trace_path, OUT / "offline-review.json"]},
    }
    assert not report["torch_imported"]
    (OUT / "guarded-collector-offline-verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
