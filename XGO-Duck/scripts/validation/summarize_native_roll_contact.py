"""Read the saved contact signal at the annotated inversion frame."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = "duck_roll-native_nominal_s1241_20261115T230228"
SOURCE = ROOT / "records/evaluations" / RUN
OUT = ROOT / "records/reports" / RUN


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target = OUT / "roll-contact-review.json"
    if target.exists(): raise FileExistsError("Keep the existing contact review")
    frames = json.loads((OUT / "keyframes.json").read_text(encoding="utf-8"))["frames"]
    selected = min((frame for frame in frames if frame["measured_roll_degrees"] is not None),
                   key=lambda frame: abs(frame["measured_roll_degrees"] - 180))
    step = selected["step"]
    with np.load(SOURCE / "trace.npz", allow_pickle=False) as data:
        head = data["head_contact"][:, 0]
        feet = data["foot_contact"][:, 0]
        depth = data["head_penetration_m"]
        assert head[step] and not feet[step].any()
        indices = np.flatnonzero(head[50:217]) + 50
        result = {"status": "saved-contact-reviewed", "run_id": RUN, "environment_index": 0,
                  "inversion_control": step, "inversion_seconds": step * .02,
                  "measured_roll_degrees": selected["measured_roll_degrees"],
                  "head_contact_at_inversion": bool(head[step]), "foot_contact_at_inversion": feet[step].tolist(),
                  "head_penetration_m_at_inversion": float(depth[step, 0]),
                  "head_contact_samples_during_roll": len(indices),
                  "head_contact_first_control": int(indices[0]), "head_contact_last_control": int(indices[-1]),
                  "maximum_head_penetration_m_all_worlds": float(depth.max()),
                  "source_trace_sha256": sha(SOURCE / "trace.npz"), "keyframes_sha256": sha(OUT / "keyframes.json"),
                  "checked_at": datetime.now(timezone.utc).isoformat(),
                  "scope": "Saved simulator contact and geometric penetration; contact force and real hardware are not measured."}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
