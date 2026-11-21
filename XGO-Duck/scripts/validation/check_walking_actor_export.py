"""Verify the walking exporter on actual anchor inputs and a known tensor change."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import torch

from check_walk_backward_native_rollout import inputs
from common_walk_startup import make_actor
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walk_backward_export import export_walking_actor


def main():
    torch.set_num_threads(4)
    parent, focus = inputs()
    plan = parent["environment"]
    actor = make_actor(plan, "cpu")
    official = ROOT / plan["model"]
    pinned_receipt = ROOT / focus["retention"][0]
    bank = pinned_receipt.with_name("anchors.npz")
    assert sha(pinned_receipt) == focus["retention"][1] and sha(bank) == read(pinned_receipt)["bank_sha256"]
    with np.load(bank, allow_pickle=False) as source:
        observations = source["observations"]
        assert observations.shape == (3072, 61)
    with torch.no_grad():
        layer = list(actor.mlp.modules())[-1]
        assert isinstance(layer, torch.nn.Linear) and layer.out_features == 14
        layer.bias[0] += 0.01
    with TemporaryDirectory() as directory:
        output = Path(directory) / "policy.onnx"
        parity = export_walking_actor(actor, official, plan["model_sha256"], output, observations)
        original, exported = onnx.load(official), onnx.load(output)
        assert list(original.metadata_props) == list(exported.metadata_props)
        assert list(original.graph.node) == list(exported.graph.node)
        expected_changes = []
        for before, after in zip(original.graph.initializer, exported.graph.initializer, strict=True):
            assert before.name == after.name
            if not np.array_equal(onnx.numpy_helper.to_array(before), onnx.numpy_helper.to_array(after)):
                expected_changes.append(before.name)
        assert len(expected_changes) == 1 and expected_changes[0].endswith(".bias")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        old = ort.InferenceSession(str(official), sess_options=options, providers=["CPUExecutionProvider"])
        new = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
        probes = observations[::16]
        shift = np.concatenate([new.run(None, {"obs": obs[None]})[0] - old.run(None, {"obs": obs[None]})[0]
                                for obs in probes])
        expected = np.zeros_like(shift)
        expected[:, 0] = 0.01
        np.testing.assert_allclose(shift, expected, atol=2e-6, rtol=0)
    receipt = dict(status="passed-walking-export-readback", official_sha256=sha(official), anchor_sha256=sha(bank),
                   graph_and_metadata_exact=True, changed_tensors=expected_changes,
                   maximum_known_shift_error=float(np.abs(shift-expected).max()), synthetic_check=True,
                   saved_model=False, physics_steps=0, learning_updates=0, **parity)
    path = ROOT / "records/reports/common-walk-backward-online-focus/export-interface-verification.json"
    assert not path.exists()
    write(path, receipt)
    print("WALKING_EXPORT_INTERFACE_VERIFIED", receipt, flush=True)


if __name__ == "__main__":
    main()
