"""Export learned walking tensors through the pinned official ONNX interface."""

import hashlib

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

from walk_backward_ppo import actor_mean


def export_walking_actor(actor, official_path, official_sha256, output, observations):
    """Retain the graph, coordinate metadata, and frozen input normalization."""
    assert hashlib.sha256(official_path.read_bytes()).hexdigest() == official_sha256
    assert official_path.resolve() != output.resolve() and not output.exists()
    model = onnx.load(official_path)
    graph = model.graph
    assert [node.op_type for node in graph.node] == ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]
    weights = {name: tensor.detach().cpu().numpy() for name, tensor in actor.state_dict().items()
               if name.startswith("mlp.")}
    assert set(weights) == {tensor.name for tensor in graph.initializer if tensor.name.startswith("mlp.")}
    for tensor in graph.initializer:
        if tensor.name in weights:
            array = weights[tensor.name]
            assert tuple(tensor.dims) == array.shape and np.isfinite(array).all()
            tensor.CopyFrom(numpy_helper.from_array(array, tensor.name))
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    assert len(observations) > 0 and observations.shape[1] == 61 and observations.dtype == np.float32
    assert session.get_inputs()[0].shape == [1, 61] and session.get_outputs()[0].shape == [1, 14]
    device = next(actor.parameters()).device
    with torch.no_grad():
        expected = actor_mean(actor, torch.as_tensor(observations, device=device)).cpu().numpy()
    actual = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in observations])
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    onnx.save(model, output)
    return dict(model_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                maximum_action_error=float(np.abs(actual-expected).max()), observations=len(actual))
