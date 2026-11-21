"""Recalculate the native scale decision from finished trial artifacts."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/experiments/common_walk_backward_native_scale.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    plan = read(PLAN)
    batch = ROOT / plan["record"]
    progress = read(batch / "progress.json")
    summary_path = batch / "summary.json"
    summary = read(summary_path)
    assert progress["status"] == "completed" and progress["trials"] == summary["trials"]
    assert summary["protocol_sha256"] == sha(PLAN) == sha(batch / "protocol.json")
    trials = summary["trials"]
    assert len(trials) == len(plan["trials"]) == 2 * len(plan["sizes"])
    assert not plan["heldout_used"] and not summary["heldout_used"] and not summary["learning_updates"]
    verified, timings = [], {}
    for original, trial in zip(plan["trials"], trials, strict=True):
        assert all(trial[key] == value for key, value in original.items())
        folder = ROOT / trial["record"]
        state = read(folder / "state.json")
        assert trial["protocol_sha256"] == sha(folder / "protocol.json")
        if trial["status"] == "skipped-by-earlier-capacity-stop":
            assert state["controls"] == 0 and not (folder / "execution.log").exists()
            assert any(row["status"] == "capacity-stop" and row["worlds"] <= trial["worlds"]
                       for row in trials[:trial["index"]])
            continue
        assert state["transitions"] == trial["actual_transitions"]
        if trial["status"] == "capacity-stop":
            assert state["status"] == "stopped-review-required"
            assert (folder / "execution-exit.txt").read_text().strip() != "0"
            telemetry = read(folder / "gpu-telemetry.json")
            assert any(row["used_mib"] / row["total_mib"] > plan["memory_cap_fraction"] for row in telemetry)
            continue
        assert trial["status"] == "passed" and state["status"] == "completed"
        assert (folder / "execution-exit.txt").read_text().strip() == "0"
        result = read(folder / "analysis.json")
        assert sha(folder / "analysis.json") == trial["analysis_sha256"]
        assert result["protocol_sha256"] == trial["protocol_sha256"] and result["learning_updates"] == 0
        controls = plan["warmup_controls"] + plan["timed_controls"]
        n = trial["worlds"]
        assert result["actual_transitions"] == controls * n == trial["actual_transitions"]
        assert result["batched_physical_substeps"] == controls * 5
        assert result["measured_transitions"] == n * plan["timed_controls"] == trial["measured_transitions"]
        rate = result["measured_transitions"] / result["timing"]["wall_seconds"]
        assert math.isclose(rate, trial["net_transitions_per_second"], rel_tol=1e-10)
        assert result["memory"]["peak_mib"] == max(row["used_mib"] for row in read(folder / "gpu-telemetry.json"))
        assert result["memory"]["peak_mib"] <= result["memory"]["gpu"]["total_mib"] * plan["memory_cap_fraction"]
        assert sha(folder / "native-rollout.npz") == result["trace_sha256"]
        assert sha(folder / state["latest_clip"]) == result["video_sha256"]
        assert read(folder / "clips.json")[-1]["frames"] == 101
        timings[n, trial["pass_number"]] = rate
        verified.append({"worlds": n, "pass": trial["pass_number"], "analysis_sha256": trial["analysis_sha256"]})

    chosen = None
    for n in plan["sizes"]:
        if not all((n, repeat) in timings for repeat in (1, 2)):
            continue
        if chosen is None or all(timings[n, repeat] >= timings[chosen, repeat] * (1 + plan["minimum_paired_gain"])
                                 for repeat in (1, 2)):
            chosen = n
    assert chosen == summary["selected"]["worlds"]
    assert sum(trial.get("actual_transitions", 0) for trial in trials) == summary["actual_transitions"]
    assert sum(trial.get("measured_transitions", 0) for trial in trials) == summary["measured_transitions"]
    assert summary["actual_transitions"] <= plan["maximum_transitions"]
    witness = ROOT / trials[0]["record"] / "windows-view-verification.json"
    assert read(witness)["status"] == "passed" and read(witness)["observer_host"] == "Windows"
    result = dict(status="passed-independent-scale-verification", selected_worlds=chosen, verified_trials=verified,
                  summary_sha256=sha(summary_path), windows_witness_sha256=sha(witness),
                  verifier_sha256=sha(Path(__file__)), learning_updates=0)
    (batch / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
