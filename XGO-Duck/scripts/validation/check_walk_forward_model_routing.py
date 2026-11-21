"""Check command-to-model selection and raw-action carry with the saved ONNX models."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from run_walk_backward_paired_eval import checked, rel
from record_io import write
from s1_reference_inputs import ROOT, read, sha


DRAFT = ROOT / "configs/experiments/common_walk_forward_model_routing_draft.json"
OUT = ROOT / "records/reports/common-walk-forward-routing"


def official_rows(external, commands):
    """Use the external command. A feedback-induced yaw never changes the route."""
    names = list(commands)
    vectors = np.asarray([commands[name] for name in names], dtype=np.float32)
    matches = (external[:, None] == vectors[None]).all(axis=-1)
    if not (matches.sum(axis=1) == 1).all():
        raise ValueError("External command falls outside the frozen seven-command set")
    return matches[:, names.index("forward")]


def routed_actions(observations, mask, official, learned):
    out = np.empty((len(observations), 14), dtype=np.float32)
    for model, subset in ((official, mask), (learned, ~mask)):
        if subset.any():
            out[subset] = model(observations[subset])
    return out


def main():
    draft, spec = read(DRAFT), checked()
    assert draft["physical_substeps_authorized"] == draft["learning_updates"] == 0
    for key in ("previous_scores", "reference_protocol"):
        path, expected = draft[key]
        assert sha(ROOT / path) == expected
    case = spec["cases"]["1140-official"]
    p = case["environment"]
    folder = ROOT / case["record"]
    base = ROOT / draft["official_model"]
    equal = ROOT / draft["equal_weight_model"]
    assert sha(base) == p["model_sha256"] and sha(equal) == spec["cases"]["1140-control"]["environment"]["model_sha256"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = {name: ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
                for name, path in (("official", base), ("learned", equal))}

    def direct(model, rows):
        return np.concatenate([sessions[model].run(None, {"obs": row[None]})[0] for row in rows])

    def infer(obs, external):
        mask = official_rows(external, p["commands"])
        observed = obs.copy()
        actual = routed_actions(obs, mask, lambda rows: direct("official", rows),
                                lambda rows: direct("learned", rows))
        expected = np.concatenate([direct("official" if pick else "learned", row[None])
                                   for pick, row in zip(mask, obs, strict=True)])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(obs, observed)
        assert np.isfinite(actual).all()
        return actual

    path = folder / "trace-actuator-feedback.npz"
    assert sha(path) == read(folder / "verification.json")["trace_hashes"][1]
    with np.load(path, allow_pickle=False) as trace:
        rows = trace["observations"][[0, 49, 199, 499]]
        assert trace["active"][[0, 49, 199, 499]].all()
    desired = np.zeros((p["parallel_worlds"], 3), dtype=np.float32)
    for name, worlds in p["groups"].items():
        desired[worlds] = p["commands"][name]
    mask = official_rows(desired, p["commands"])
    assert np.flatnonzero(mask).tolist() == sorted(p["groups"]["forward"])
    nominal = infer(rows.reshape(-1, 61), np.tile(desired, (len(rows), 1)))
    options_path = folder / "feedback-readback.npz"
    assert sha(options_path) == read(folder / "verification.json")["feedback"]["readback_sha256"]
    with np.load(options_path, allow_pickle=False) as feedback:
        first = feedback["step"] == 51
        worlds = feedback["world"][first]
        assert len(worlds) == 32 and np.count_nonzero(feedback["yaw"][first])
        alternate = rows[0, worlds].copy()
        alternate[:, 50] = feedback["yaw"][first]
        adjusted = infer(alternate, desired[worlds])
        np.testing.assert_allclose(adjusted, feedback["alternate_raw"][first], atol=2e-5, rtol=2e-5)
    try:
        official_rows(np.array([[.123, .234, .345]], dtype=np.float32), p["commands"])
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported command was accepted")

    last = np.zeros((1, 14), dtype=np.float32)
    sequence = ("forward", "backward", "stand", "forward")
    carried = []
    for command in sequence:
        world = p["groups"][command][0]
        obs = rows[0, world:world+1].copy()
        obs[:, 34:48] = last
        np.testing.assert_array_equal(obs[:, 34:48], last)
        last = infer(obs, desired[world:world+1])
        home = np.asarray(p["model_home"])
        lo, hi = np.asarray(p["joint_limits"]).T
        target = np.clip(last + home, lo, hi)
        assert ((target >= lo) & (target <= hi)).all()
        carried.append(dict(command=command, raw=last.tolist()[0], projected_target=target.tolist()[0]))

    OUT.mkdir(exist_ok=True)
    write(OUT / "cpu-components.json", dict(
        status="passed-cpu-routing-components", draft_sha256=sha(DRAFT), source_sha256=sha(Path(__file__)),
        official=[rel(base), sha(base)], learned=[rel(equal), sha(equal)], nominal_rows=len(nominal),
        forward_feedback_rows=len(adjusted), unsupported_command_rejected=True, raw_carry_sequence=carried,
        physical_substeps=0, learning_updates=0, heldout_used=False, hardware_io=False))
    print("FORWARD_ROUTING_CPU", len(nominal), len(adjusted), list(sequence), flush=True)


if __name__ == "__main__":
    main()
