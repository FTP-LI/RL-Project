"""Check the saved route-entry curriculum against the official ONNX interface."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "records/reports/route-aware-walk-training/forward-entry-curriculum.json"
DATA = ROOT / "records/reports/route-aware-walk-training/forward-entry-curriculum.npz"
MODEL = ROOT / "artifacts/delivery/official/walk-simulation-reference.onnx"
OUT = ROOT / "records/reports/route-aware-walk-training/curriculum-check.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "passed-offline-state-curriculum"
    with np.load(DATA) as data:
        arrays = {key: data[key] for key in data.files}
    assert arrays["observed"].shape == (manifest["entries"], 61)
    assert arrays["actor_input"].shape == (manifest["entries"], 61)
    assert arrays["raw_policy_action"].shape == (manifest["entries"], 14)
    assert arrays["qpos"].shape[1:] == (21,)
    assert arrays["qvel"].shape[1:] == (20,)
    assert arrays["window_phase_next"].shape[1:] == (51,)
    assert np.all(arrays["window_phase_next"][:, 0] == 3)
    assert np.isfinite(np.concatenate([arrays["actor_input"].ravel(), arrays["raw_policy_action"].ravel()])).all()
    import onnxruntime as ort
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    output = np.concatenate([session.run(None, {"obs": row[None].astype(np.float32)})[0]
                             for row in arrays["actor_input"]], axis=0)
    max_error = float(np.max(np.abs(output - arrays["raw_policy_action"])))
    assert output.shape == arrays["raw_policy_action"].shape and np.isfinite(output).all()
    assert max_error <= 2e-4, max_error
    result = {
        "status": "passed-curriculum-interface-check",
        "entries": int(manifest["entries"]),
        "model": MODEL.relative_to(ROOT).as_posix(),
        "model_sha256": sha(MODEL),
        "max_official_onnx_action_error": max_error,
        "physical_steps_added": 0,
        "learning_updates": 0,
        "hardware_io": False,
        "reset_readback": "Offline qpos/qvel and actor-input contract checked; live physics restore remains the next gate.",
        "data": DATA.relative_to(ROOT).as_posix(),
        "data_sha256": sha(DATA),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
