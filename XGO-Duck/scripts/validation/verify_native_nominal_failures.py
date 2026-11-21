"""Independently recompute failed roll progress from saved quaternion/contact arrays."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from roll_metrics import progress

ROOT = Path(__file__).resolve().parents[2]
RUN = "duck_roll-native_nominal_s1241_20261115T230228"
REPORT = ROOT / "records/reports" / RUN
FOLDER = ROOT / "records/evaluations" / RUN


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = REPORT / "failure-review-check.json"
    if output.exists(): raise FileExistsError("Keep the previous failure check")
    review = json.loads((REPORT / "failure-review.json").read_text(encoding="utf-8"))
    ids = review["failed_worlds"]
    with np.load(FOLDER / "trace.npz", allow_pickle=False) as archive:
        quaternion = archive["quaternion"][50:501, ids]
        ground = archive["ground_contact"][50:501, ids]
    geometric, supported, _ = progress({"quaternion": quaternion, "ground_contact": ground})
    geometric, supported = np.rad2deg(geometric[-1]), np.rad2deg(supported[-1])
    expected_geo = np.array([row["final_accumulated_pitch_degrees"] for row in review["rows"]])
    expected_support = np.array([row["final_supported_pitch_degrees"] for row in review["rows"]])
    np.testing.assert_allclose(geometric, expected_geo, rtol=0, atol=1e-9)
    np.testing.assert_allclose(supported, expected_support, rtol=0, atol=1e-9)
    assert np.all(supported < 330) and len(ids) == 5
    evidence = {"status": "failed-progress-independently-recomputed", "checked_at": datetime.now(timezone.utc).isoformat(),
                "failed_worlds": ids, "geometric_degrees": geometric.tolist(), "supported_degrees": supported.tolist(),
                "source_review_sha256": sha(REPORT / "failure-review.json"), "trace_sha256": sha(FOLDER / "trace.npz"),
                "source_sha256": sha(Path(__file__)), "scope": "Saved geometric and support-weighted progress, no causal inference."}
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__": main()
