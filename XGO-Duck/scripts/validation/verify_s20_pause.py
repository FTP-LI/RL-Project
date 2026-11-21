"""Verify the S20 user-requested pause and retained evidence without simulation."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/roll-domain-parameter-random-s20"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    progress = read(OUT / "progress.json")
    summary = read(OUT / "pause-summary.json")
    stop = read(OUT / "stop-record.json")
    media = read(OUT / "partial-media-audit.json")
    assert progress["status"] == "stopped-by-user"
    assert progress["completed"] == summary["completed_arms"] == stop["completed_arms"] == 6
    assert progress["total"] == summary["total_arms"] == stop["total_arms"] == 8
    assert sha(OUT / "progress.json") == stop["sha256"]["progress.json"]
    scores = []
    for expected, entry in zip(summary["arms"], progress["results"], strict=True):
        analysis = read(ROOT / entry["output"] / "analysis.json")
        assert expected["name"] == entry["name"]
        assert expected["roll_completed"] == analysis["counts"]["roll_completed"]
        assert expected["entered_forward"] == analysis["counts"]["entered_forward"]
        assert expected["continuous_successes"] == analysis["counts"]["continuous_successes"]
        assert expected["passed_development_gate"] == analysis["passed_development_gate"]
        assert analysis["completed_controls"] == 1550
        video = ROOT / analysis["video"]["path"]
        assert video.is_file() and sha(video) == analysis["video"]["sha256"]
        scores.append({"name": expected["name"], "counts": analysis["counts"], "video_sha256": analysis["video"]["sha256"]})

    partial = ROOT / "records/evaluations/duck_roll-domain-parameter_high-mid_s1240_20261114T113334"
    state = read(partial / "state.json")
    assert state["status"] == "stopped-by-user" and state["controls"] == 1400
    assert sha(partial / "state.json") == stop["sha256"]["partial_state.json"]
    assert media["frames"]["count"] == 29 and media["frames"]["last"] == "frame-step-1400.jpg"
    assert not media["video"]["has_complete_moov"]
    assert not (partial / "trace.npz").exists()
    assert not media["resumable_physics_checkpoint_present"]
    result = {
        "status": "passed",
        "completed_arms": len(scores),
        "partial_controls": state["controls"],
        "partial_video_finalized": False,
        "partial_resumable": False,
        "scores": scores,
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
            for path in [OUT / "progress.json", OUT / "pause-summary.json", OUT / "stop-record.json", OUT / "partial-media-audit.json", Path(__file__)]
        },
    }
    (OUT / "pause-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["status", "completed_arms", "partial_controls", "partial_video_finalized", "partial_resumable"]}, indent=2))


if __name__ == "__main__":
    main()
