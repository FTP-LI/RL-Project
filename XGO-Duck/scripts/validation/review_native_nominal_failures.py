"""Reconstruct the original roll latch for failed saved baseline episodes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence

RUN = "duck_roll-native_nominal_s1241_20261115T230228"
REPORT = ROOT / "records/reports" / RUN
FOLDER = ROOT / "records/evaluations" / RUN
PLAN = ROOT / "configs/experiments/roll_native_nominal_observed_s1241_20261115.json"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target = REPORT / "failure-review.json"
    if target.exists(): raise FileExistsError("Keep the existing failure review")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    analysis = json.loads((REPORT / "independent-analysis.json").read_text(encoding="utf-8"))
    saved = json.loads((FOLDER / "raw-score.json").read_text(encoding="utf-8"))
    failures = [item["world"] for item in saved["forward"] if not item["success"]]
    assert len(failures) == 5 and analysis["counts"]["continuous_successes"] == 123
    with np.load(FOLDER / "trace.npz", allow_pickle=False) as archive:
        # NPZ members decompress on access; cache them once for the 501-step replay.
        data = {key: archive[key] for key in STATE_FIELDS}
        seq = Sequence(plan)
        rolling_at_start = None
        first_intact_loss = {world: None for world in failures}
        for step in range(plan["initial_stand_controls"] + plan["roll_controls"] + 1):
            before = seq.roll_latch.intact.copy()
            seq.update(step, {key: data[key][step] for key in STATE_FIELDS})
            if step == plan["initial_stand_controls"]:
                rolling_at_start = seq.phase == ROLL
            for world in failures:
                if before[world] and not seq.roll_latch.intact[world] and first_intact_loss[world] is None:
                    first_intact_loss[world] = step
        latch = seq.roll_latch
        rows = []
        for world in failures:
            contacts = np.flatnonzero(data["head_contact"][50:501, world]) + 50
            rows.append({"world": world, "entered_roll": bool(rolling_at_start[world]),
                         "roll_latch_ready_control": int(seq.stand_at[world]),
                         "forward_entry_control": int(saved["forward_at"][world]),
                         "final_accumulated_pitch_degrees": float(np.rad2deg(latch.angle[world])),
                         "final_supported_pitch_degrees": float(np.rad2deg(latch.supported[world])),
                         "head_contact_seen_by_latch": bool(latch.seen_head[world]),
                         "roll_completion_seen_by_latch": bool(latch.finished[world]),
                         "latch_intact_at_end": bool(latch.intact[world]),
                         "first_latch_intact_loss_control": first_intact_loss[world],
                         "head_contact_samples": len(contacts),
                         "maximum_head_penetration_m": float(data["head_penetration_m"][:, world].max()),
                         "last_active": bool(data["active"][-1, world])})
    result = {"status": "saved-failures-reviewed", "checked_at": datetime.now(timezone.utc).isoformat(),
              "run_id": RUN, "failed_worlds": failures, "rows": rows,
              "protocol_sha256": sha(PLAN), "trace_sha256": sha(FOLDER / "trace.npz"),
              "source_sha256": sha(Path(__file__)), "learning_updates": 0,
              "scope": "Descriptive replay of saved latch and contact states; not a causal intervention or new training."}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
