"""Read both trained exports through the right-only router on archived seven-command input."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from s1_reference_inputs import ROOT, read, sha
from check_walk_right_startup_native import OUT, load_actor, source_plan
from check_walk_right_startup_routes import DRAFT, RightCommandActor
from common_walk_startup import make_actor, td
from record_io import write
from run_walk_forward_model_routing import FixedCommandActor
from run_walk_right_startup_paired_ppo import PLAN, checked


def main():
    torch.set_num_threads(4)
    plan = read(DRAFT)
    pair = checked()
    proof_path = ROOT / plan["trained_pair_readback"]
    proof = read(proof_path)
    assert proof["status"] == "passed-native-right-turn-paired-ppo-readback" and proof["equal_start"]
    assert proof["protocol_sha256"] == sha(PLAN) and not proof["heldout_used"]
    source_proof = OUT / "right-route-components.json"
    component = read(source_proof)
    assert component["status"] == "passed-right-only-archive-routing" and component["draft_sha256"] == sha(DRAFT)
    draft, parent = source_plan()
    case = parent["cases"][str(plan["known_case_seed"])]
    p, record = case["environment"], ROOT / case["record"]
    trace = record / "trace-shared-prefix.npz"
    assert sha(trace) == component["archive_sha256"] == read(record / "verification.json")["shared_prefix_sha256"]
    desired = np.empty((plan["worlds"], 3), np.float32)
    for name, ids in p["groups"].items():
        desired[ids] = p["commands"][name]
    official, base = make_actor(p, "cpu"), load_actor(draft, p, "cpu")
    nominal = FixedCommandActor(official, base, desired, p["commands"], "cpu")
    with np.load(trace, allow_pickle=False) as saved:
        samples = saved["observations"][[0, 49, 99, 149, 199]]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    results = {}
    for role, item in pair["arms"].items():
        path = ROOT / item["record"] / "policy.onnx"
        digest = proof["per_arm"][role]["onnx_sha256"]
        assert sha(path) == digest
        routed = RightCommandActor(official, base, make_actor(dict(p, model=path.relative_to(ROOT).as_posix(),
                                  model_sha256=digest), "cpu"), desired, p["commands"], "cpu")
        right = routed.right_worlds
        kept = torch.ones(len(desired), dtype=torch.bool)
        kept[right] = False
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        maximum = 0.
        changed = 0
        with torch.no_grad():
            for obs in samples:
                inputs = td(torch.from_numpy(obs))
                old, new = nominal(inputs), routed(inputs)
                torch.testing.assert_close(new[kept], old[kept], atol=0, rtol=0)
                requested = obs[right.numpy()]
                expected = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in requested])
                actual = new[right].numpy()
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
                subset = inputs[routed.forward_ids]
                torch.testing.assert_close(routed(subset), official(subset), atol=0, rtol=0)
                maximum = max(maximum, float(np.abs(actual-expected).max()))
                changed += int(((new[right]-old[right]).abs().max(dim=1).values > 0).sum())
        assert changed > 0
        results[role] = dict(onnx_sha256=digest, right_inputs=len(samples)*len(right), changed_right_inputs=changed,
            exact_non_right_inputs=int(kept.sum())*len(samples), exact_forward_feedback_inputs=len(samples)*32,
            maximum_onnx_difference=maximum)
    assert not torch.cuda.is_initialized()
    output = dict(status="passed-trained-right-only-routing", source_sha256=sha(Path(__file__)),
        pair_readback_sha256=sha(proof_path), fixture_readback_sha256=sha(source_proof),
        archive_sha256=sha(trace), results=results, physical_substeps=0, learning_updates=0, heldout_used=False)
    write(OUT / "trained-right-route-components.json", output)
    print("RIGHT_STARTUP_TRAINED_ROUTES", output, flush=True)


if __name__ == "__main__":
    main()
