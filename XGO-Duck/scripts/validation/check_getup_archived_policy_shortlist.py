"""Recheck selected archived policies, recoverable weights and common inputs."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from record_io import write


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/"configs/experiments/getup_archived_policy_shortlist.json"
OUT = ROOT/"records/reports/getup-archived-policy-shortlist"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def pinned(pair):
    name,expected = pair
    path = ROOT/name
    assert sha(path) == expected, path
    return read(path)


def scores(arm):
    rows = arm["timing"]["episodes"]
    final = arm["end_hold"]["stable_by_environment"]
    complete = [row["hold_start_seconds"] is not None and stable for row,stable in zip(rows,final,strict=True)]
    assert sum(complete) == arm["combined_successes"]
    return dict(total=len(rows),overall=sum(complete),
        actual_supine=sum(ok and row["release_pose"]=="supine" for ok,row in zip(complete,rows,strict=True)),
        actual_supine_n=sum(row["release_pose"]=="supine" for row in rows),
        requested_supine=sum(ok and row["requested_pose"]=="supine" for ok,row in zip(complete,rows,strict=True)),
        requested_supine_n=sum(row["requested_pose"]=="supine" for row in rows))


def infer(model,inputs,interface):
    path = ROOT/model["model"]
    assert sha(path) == model["model_sha256"]
    manifest = read(path.with_name("manifest.json"))
    assert manifest["status"] == "exported-parity-checked" and manifest["model_sha256"] == model["model_sha256"]
    checkpoint = ROOT/model["checkpoint"]
    assert sha(checkpoint) == manifest["checkpoint_sha256"] == model["checkpoint_sha256"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    runtime = ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
    inp, = runtime.get_inputs()
    out, = runtime.get_outputs()
    assert inp.name == "obs" and inp.shape[-1] == 61 and out.shape[-1] == 14
    meta = runtime.get_modelmeta().custom_metadata_map
    result = np.concatenate([runtime.run(None,{inp.name:row[None]})[0] for row in inputs])
    assert result.shape == (len(inputs),14) and np.isfinite(result).all()
    return result,{key:meta[key] for key in interface}


def main():
    p = read(PLAN)
    project = read(ROOT/"project.json")
    assert p["status"] == "frozen-archived-policy-interface-only" and not OUT.exists()
    assert project["current_phase"] == p["stage"] and project["goal_execution"]["status"] == "active"
    assert ROOT/project["goal_execution"]["current_protocol"] == PLAN
    assert not any(p[key] for key in ("physical_substeps","learning_updates","heldout_used","hardware_io"))
    screened = pinned(p["closed_curriculum"])
    assert screened["status"] == "passed-independent-hard-start-screen" and not screened["admitted_to_curriculum"]
    record = ROOT/screened["record"]
    state = read(record/"evaluation.json")
    path = record/"policy-replay.npz"
    assert sha(path) == state["policy_replay_sha256"]
    with np.load(path,allow_pickle=False) as src:
        inputs = src["obs"].reshape(-1,61)
        actual = src["raw"].reshape(-1,14)
    baseline = read(record/"protocol.json")["environment"]
    baseline = {key:baseline[key] for key in ("model","model_sha256")}
    baseline.update(checkpoint=p["baseline_checkpoint"],
        checkpoint_sha256=read((ROOT/baseline["model"]).with_name("manifest.json"))["checkpoint_sha256"])
    base,interface = infer(baseline,inputs,p["interface"])
    np.testing.assert_allclose(base,actual,atol=2e-5,rtol=2e-5)
    results = {}
    for role,item in p["candidates"].items():
        e,t = (pinned(item[key]) for key in ("evaluation","training"))
        assert e["status"] == "passed-independent-verification" and not e["heldout_used"] and not t["heldout_used"]
        ep = ROOT/e["record"]/"protocol.json"
        assert sha(ep) == e["protocol_sha256"]
        protocol = read(ep)
        assert protocol["policies"]["fixed-weighted"]["model_sha256"] == baseline["model_sha256"]
        arm = item["arm"]
        proof = t["arms"][arm]
        model = {key:proof[key] for key in ("model","model_sha256","checkpoint_sha256")}
        assert proof["status"] == "passed" and proof["checkpoint_reloaded"]
        assert model["model_sha256"] == protocol["policies"][arm]["model_sha256"]
        model["checkpoint"] = proof["record"]+"/model-final.pt"
        parent,selected = (scores(e["arms"][key]) for key in ("fixed-weighted",arm))
        assert parent["total"] == selected["total"] and parent["actual_supine_n"] == selected["actual_supine_n"]
        assert selected["overall"]-parent["overall"] >= p["minimum_development_overall_net_gain"]
        assert selected["actual_supine"]-parent["actual_supine"] >= p["minimum_development_actual_supine_net_gain"]
        value,meta = infer(model,inputs,p["interface"])
        assert meta == interface == protocol["model_interface"]
        change = float(np.sqrt(np.square(value-base).mean()))
        assert change > 1e-5
        results[role] = dict(model=model,source_seed=protocol["environment"]["seed"],archived_parent=parent,archived_candidate=selected,
            archived_full_gate=bool(e["arms"][arm]["all_motion_gates_passed"]),shared_input_rms_delta_rad=change,
            checkpoints_verified=True,physical_steps=0)
    OUT.mkdir()
    write(OUT/"analysis.json",dict(status="passed-archived-policy-shortlist",protocol_sha256=sha(PLAN),
        source_sha256=sha(Path(__file__)),shared_input_sha256=sha(path),shared_input_rows=len(inputs),
        fixed_baseline=baseline,interface=interface,baseline_action_replay_max_error=float(abs(actual-base).max()),
        candidates=results,admitted_to_fresh_four_pose_comparison=True,physical_substeps=0,
        learning_updates=0,heldout_used=False,hardware_io=False))
    print("GETUP_ARCHIVED_POLICY_SHORTLIST",len(inputs),results,flush=True)


if __name__ == "__main__":
    main()
