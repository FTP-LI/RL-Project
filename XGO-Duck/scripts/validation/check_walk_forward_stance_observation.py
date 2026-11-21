"""Fit and verify a support-side observer from ordinary policy inputs only."""

from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/"configs/experiments/common_walk_forward_stance_observation.json"
OUT = ROOT/"records/reports/common-walk-right-support-yaw-training"


def load_group(group):
    protocol, result = (ROOT/group[key] for key in ("protocol", "receipt"))
    spec, evidence = read(protocol), read(result)
    assert evidence["protocol_sha256"] == sha(protocol) and not evidence["heldout_used"]
    cases = {}
    for key in group["cases"]:
        item = evidence["records"][key]
        record, p = ROOT/item["path"], spec["cases"][key]["environment"]
        receipt = record/"verification.json"
        assert sha(receipt) == item["verification_sha256"]
        proof = read(receipt)
        ids = p["groups"]["forward"]
        joints = [i for i, name in enumerate(p["joint_names"]) if name.startswith(("left_", "right_"))]
        assert len(joints) == 10
        columns = [0,1,3,4,5]+[6+i for i in joints]+[20+i for i in joints]
        parts = []
        for name, checksum in zip(("shared-prefix", "actuator-feedback"), proof["trace_hashes"], strict=True):
            path = record/f"trace-{name}.npz"
            assert sha(path) == checksum
            with np.load(path, allow_pickle=False) as archive:
                assert archive["active"][:,ids].all()
                parts.append((archive["observations"][:,ids][:,:,columns], archive["contact"][:,ids]))
        features, contacts = (np.concatenate(x, axis=0) for x in zip(*parts, strict=True))
        assert features.shape == (1000,32,len(columns)) and contacts.shape == (1000,32,2)
        before = contacts[:-1]
        phases = before[:,:,0].astype(int)+2*before[:,:,1]
        labels = np.where(phases == 3, 0, phases)
        cases[key] = dict(seed=spec["cases"][key]["seed"], worlds=ids, inputs=features[1:], labels=labels,
                          columns=columns, evidence_sha256=sha(receipt))
    return cases


def report_metrics(actual, prediction, plan):
    matrix = np.bincount((3*actual+prediction).ravel(), minlength=9).reshape(3,3)
    true, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    assert (true > 0).all()
    precision = np.divide(matrix.diagonal(), predicted, out=np.zeros(3), where=predicted > 0)
    recall = matrix.diagonal()/true
    other = float(1.-recall[0])
    admitted = bool((precision[1:] >= plan["minimum_single_side_precision"]).all() and
                    (recall[1:] >= plan["minimum_single_side_recall"]).all() and
                    other <= plan["maximum_other_as_single_fraction"])
    return dict(rows=int(matrix.sum()), confusion=matrix.tolist(), precision=precision.tolist(),
                recall=recall.tolist(), other_as_single_fraction=other, passed=admitted)


def main():
    plan = read(PLAN)
    assert plan["status"] == "frozen-archive-cpu-screen" and plan["class_order"] == ["other","left-only","right-only"]
    assert not any(plan[k] for k in ("heldout_used", "physical_substeps", "policy_updates", "hardware_io"))
    fit = load_group(plan["fit"])
    validation = {key: row for group in plan["verification"] for key,row in load_group(group).items()}
    source, = fit.values()
    assert source["seed"] not in {row["seed"] for row in validation.values()}
    columns = source["columns"]
    x = source["inputs"].reshape(-1,len(columns)).astype(np.float64)
    y = source["labels"].ravel()
    weight = 1./np.bincount(y)[y]
    weight /= weight.sum()
    mean, scale = x.mean(axis=0), x.std(axis=0).clip(1e-6)
    standardized = np.column_stack(((x-mean)/scale, np.ones(len(x))))
    regularization = np.eye(standardized.shape[1])*plan["ridge"]
    regularization[-1,-1] = 0
    gram = (standardized*weight[:,None]).T@standardized
    target = (standardized*weight[:,None]).T@np.eye(3)[y]
    weights = np.linalg.solve(gram+regularization, target)
    results = {}
    for key,row in validation.items():
        assert row["columns"] == columns
        inputs = (row["inputs"]-mean)/scale
        scores = inputs@weights[:-1]+weights[-1]
        winner = scores.argmax(axis=2)
        top = np.sort(scores, axis=2)[:,:,-2:]
        predicted = np.where(top[:,:,1]-top[:,:,0] >= plan["minimum_score_separation"], winner, 0)
        result = report_metrics(row["labels"], predicted, plan)
        result["seed"] = row["seed"]
        result["source_receipt_sha256"] = row["evidence_sha256"]
        result["known_worlds"] = {}
        if row["seed"] == 1152:
            for world in plan["selected_known_worlds"]:
                col = row["worlds"].index(world)
                result["known_worlds"][str(world)] = report_metrics(row["labels"][:,col],predicted[:,col],plan)
        results[key] = result
    admitted = all(row["passed"] and all(v["passed"] for v in row["known_worlds"].values()) for row in results.values())
    write(OUT/"forward-stance-observation-cpu.json", dict(status="checked-archive-stance-observation",
        protocol_sha256=sha(PLAN), source_sha256=sha(Path(__file__)), fit_seed=source["seed"],
        fit_rows=len(x), fit_receipt_sha256=source["evidence_sha256"], verification=results,
        admitted_for_control_design=admitted, model=dict(columns=columns, means=mean.tolist(),
        scales=scale.tolist(), weights=weights.tolist(), class_order=plan["class_order"]),
        physical_substeps=0, policy_updates=0, heldout_used=False, hardware_io=False))
    print("FORWARD_STANCE_CPU", admitted, {key: (row["precision"],row["recall"]) for key,row in results.items()}, flush=True)


if __name__ == "__main__":
    main()
