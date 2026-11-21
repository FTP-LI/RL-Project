"""Replay the actual first observation under both CUDA batching layouts, without physics."""

import fcntl
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.torch import configure_torch_backends

from run_walk_forward_model_routing import FixedCommandActor, OUT, PLAN, checked, native
from record_io import write
from s1_reference_inputs import ROOT, read, sha


@torch.inference_mode()
def replay(spec, result):
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    models = {name: native.rollout.make_actor(dict(model=path, model_sha256=digest), "cuda:0")
              for name, (path, digest) in spec["models"].items()}
    rows = {}
    for seed in spec["development_seeds"]:
        first = {}
        for role in spec["arms"]:
            item = spec["cases"][f"{seed}-{role}"]
            folder = ROOT / item["record"]
            receipt = folder / "verification.json"
            assert sha(receipt) == result["records"][f"{seed}-{role}"]["verification_sha256"]
            prefix = folder / "trace-shared-prefix.npz"
            assert sha(prefix) == read(receipt)["trace_hashes"][0]
            with np.load(prefix, allow_pickle=False) as trace:
                assert trace["active"][0].all()
                first[role] = (trace["observations"][0], trace["actions"][0])
        obs = first["routed"][0]
        for reference in ("official", "control"):
            np.testing.assert_array_equal(first[reference][0], obs)
        p = item["environment"]
        _, desired = native.rollout.groups_and_commands(p)
        wrapper = FixedCommandActor(models["official"], models["control"], desired, p["commands"], "cuda:0")
        inputs = native.rollout.td(torch.from_numpy(obs).to("cuda:0"))
        split = native.rollout.cpu(wrapper(inputs))
        np.testing.assert_allclose(split, first["routed"][1], atol=2e-5, rtol=2e-5)
        for name, ids in (("official", wrapper.forward_ids), ("control", wrapper.other_ids)):
            indices = native.rollout.cpu(ids)
            full = native.rollout.cpu(models[name](inputs))[indices]
            observed_full, observed_split = first[name][1][indices], first["routed"][1][indices]
            np.testing.assert_allclose(full, observed_full, atol=2e-5, rtol=2e-5)
            actual = split[indices]
            rows[f"{seed}-{name}"] = dict(
                rows=len(indices), full_batch_replays_recorded_exactly=np.array_equal(full, observed_full),
                split_batch_replays_recorded_exactly=np.array_equal(actual, observed_split),
                maximum_layout_difference_rad=float(abs(full-actual).max()),
                maximum_recorded_difference_rad=float(abs(observed_full-observed_split).max()),
                changed_scalar_outputs=int(np.count_nonzero(full != actual)),
                full_replay_maximum_error_rad=float(abs(full-observed_full).max()),
                split_replay_maximum_error_rad=float(abs(actual-observed_split).max()))
    return rows


def main():
    spec = checked()
    result_path = OUT / "analysis.json"
    result = read(result_path)
    assert result["status"] == "passed-independent-fixed-routing-readback" and result["protocol_sha256"] == sha(PLAN)
    with (Path.home() / "XGO-Duck/training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = replay(spec, result)
    write(OUT / "cuda-batch-shape.json", dict(status="passed-first-input-cuda-batch-readback", source_sha256=sha(Path(__file__)),
        fixed_result_sha256=sha(result_path), protocol_sha256=sha(PLAN), results=rows,
        scope="First nominal observations only; confirms arithmetic effects of layout, not causes of downstream gait.",
        physical_substeps=0, learning_updates=0, heldout_used=False))
    print("ROUTING_CUDA_BATCH_READBACK", rows, flush=True)


if __name__ == "__main__":
    main()
