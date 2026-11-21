"""Decompose saved control-boundary roll progress loss, without causal claims."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from roll_metrics import progress, rotation

ROOT = Path(__file__).resolve().parents[2]
RUN = "duck_roll-native_nominal_s1241_20261115T230228"
REPORT = ROOT / "records/reports" / RUN
FOLDER = ROOT / "records/evaluations" / RUN


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = REPORT / "support-gap-analysis.json"
    if output.exists(): raise FileExistsError("Keep the existing analysis")
    score = json.loads((FOLDER / "raw-score.json").read_text(encoding="utf-8"))
    review = json.loads((REPORT / "failure-review.json").read_text(encoding="utf-8"))
    with np.load(FOLDER / "trace.npz", allow_pickle=False) as archive:
        quaternion = archive["quaternion"][50:501]
        ground = archive["ground_contact"][50:501]
    geometric, supported, matrices = progress({"quaternion": quaternion, "ground_contact": ground})
    flat = np.clip((.866 - np.abs(matrices[..., 2, 1])) / (.866 - .5), 0., 1.)
    flat = flat * flat * (3. - 2. * flat)
    delta = np.diff(geometric, axis=0, prepend=geometric[:1])
    positive = np.maximum(delta, 0.)
    airborne = np.cumsum(positive * ~ground, axis=0)
    lateral = np.cumsum(positive * ground * (1. - flat), axis=0)
    np.testing.assert_allclose(geometric - supported, airborne + lateral, atol=1e-10, rtol=0)
    rows = []
    for world, ready in enumerate(score["roll_switch_at"]):
        end = (ready if ready >= 0 else 500) - 50
        rows.append({"world": world, "roll_ready": ready >= 0, "end_control": end + 50,
                     "geometric_degrees": float(np.rad2deg(geometric[end, world])),
                     "supported_degrees": float(np.rad2deg(supported[end, world])),
                     "no_ground_contact_loss_degrees": float(np.rad2deg(airborne[end, world])),
                     "lateral_discount_loss_degrees": float(np.rad2deg(lateral[end, world]))})
    failed = [row for row in rows if not row["roll_ready"]]
    passed = [row for row in rows if row["roll_ready"]]
    assert [row["world"] for row in failed] == review["failed_worlds"] and len(passed) == 123
    keys = ("geometric_degrees", "supported_degrees", "no_ground_contact_loss_degrees", "lateral_discount_loss_degrees")
    summarize = lambda group: {key: float(np.median([row[key] for row in group])) for key in keys}
    result = {"status": "saved-control-boundary-decomposition", "checked_at": datetime.now(timezone.utc).isoformat(),
              "run_id": RUN, "failed_count": len(failed), "passed_count": len(passed),
              "failed_medians": summarize(failed), "passed_medians": summarize(passed), "failed_rows": failed,
              "trace_sha256": sha(FOLDER / "trace.npz"), "source_sha256": sha(Path(__file__)),
              "scope": "Control-boundary contact flags and original flatness weighting; contact between controls and causal effects are not measured."}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
