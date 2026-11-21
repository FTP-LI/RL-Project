"""CPU-only preflight for the separate getup, roll and walk skill plan."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/experiments/multi_skill_training_plan_draft.json"
OUT = ROOT / "records/reports/multi-skill-training"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import onnxruntime as ort

    plan = read(PLAN)
    assert plan["status"] == "reserved-before-preflight"
    regression = read(ROOT / plan["parent_regression_evidence"])
    assert regression["status"] == "completed"
    assert not plan["hardware_io"] and plan["learning_updates"] == 0
    models = {
        "getup-official": "official_resources/xgoduck_runtime_arduino/python/xgoduck_getup.onnx",
        "roll-official": "official_resources/xgoduck_runtime_arduino/python/xgoduck_roulade.onnx",
        "walk-official": "workspace/official-reference/b401d85d20bb5bfd0237242a8a5ab6847fc962a2/python/xgoduck_walk.onnx",
    }
    for arm in regression["arms"]:
        models[f"getup-{arm['arm']}"] = arm["model"]
    rows = []
    for name, relative in models.items():
        path = ROOT / relative
        assert path.exists(), relative
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        assert len(inputs) == 1 and len(outputs) == 1, name
        assert list(inputs[0].shape) == [1, 61], (name, inputs[0].shape)
        assert list(outputs[0].shape) == [1, 14], (name, outputs[0].shape)
        metadata = dict(session.get_modelmeta().custom_metadata_map)
        assert len(metadata.get("joint_names", "").split(",")) == 14, name
        assert float(metadata["action_scale"]) == 1.0, name
        rows.append(dict(name=name, model=relative, model_sha256=sha(path),
                         input_shape=list(inputs[0].shape), output_shape=list(outputs[0].shape),
                         joint_names=metadata["joint_names"].split(","),
                         action_scale=float(metadata["action_scale"])))
    result = dict(status="passed-cpu-model-interface", checked_at=datetime.now(timezone.utc).isoformat(),
                  plan_sha256=sha(PLAN), parent_regression_evidence=plan["parent_regression_evidence"],
                  models=rows, physical_steps=0, learning_updates=0, hardware_io=False,
                  providers=["CPUExecutionProvider"],
                  next_gate="GPU zero-step per independent getup, roll and walk environment")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
