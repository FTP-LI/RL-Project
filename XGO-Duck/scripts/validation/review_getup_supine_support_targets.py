"""Separate original requests, projected targets and realized joints at support."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnx

from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/"configs/experiments/getup_supine_support_target_readback.json"
OUT = ROOT/"records/reports/getup-supine-support-target-readback"


def summarize(traces,mask,limits,joints,spec):
    request, applied, actual = traces
    keep = mask.any(axis=0)
    if not keep.any():
        return dict(episodes=0,samples=0,joints={})
    for signal in traces:
        assert np.isfinite(signal[mask]).all()
    low,high = limits
    distance = np.minimum(applied-low,high-applied)
    assert distance[mask].min() >= -spec["clipping_tolerance_rad"]
    edge = distance <= spec["edge_tolerance_rad"]
    changed = abs(applied-request) > spec["clipping_tolerance_rad"]
    error = np.rad2deg(applied-actual)
    exposure = mask[:,keep].sum(axis=0)

    def episodes(values):
        values = np.where(mask,values,0.)[:,keep]
        return values.sum(axis=0)/exposure

    return dict(episodes=int(keep.sum()),samples=int(mask.sum()),joints={name:dict(
        projection_changed_fraction=float(changed[...,i][mask].mean()),
        applied_at_limit_fraction=float(edge[...,i][mask].mean()),
        mean_absolute_tracking_gap_median_deg=float(np.median(episodes(abs(error[...,i])))),
        mean_signed_tracking_gap_median_deg=float(np.median(episodes(error[...,i]))))
        for i,name in enumerate(joints)})


def review(seed,spec,archive,labels):
    reference,expected = archive["archives"][str(seed)]
    source = ROOT/reference
    assert sha(source) == expected
    proof = read(source)
    assert not proof["heldout_used"] and proof["status"] == "passed-independent-verification"
    folder = ROOT/proof["record"]
    protocol = folder/"protocol.json"
    assert sha(protocol) == proof["protocol_sha256"]
    env = read(protocol)["environment"]
    assert env["seed"] == seed and env["criteria"]["step_dt"] == .02
    model = ROOT/env["model"]
    assert sha(model) == env["model_sha256"]
    names = {p.key:p.value for p in onnx.load(model,load_external_data=False).metadata_props}["joint_names"].split(",")
    joints = spec["joint_names"]
    cols = [names.index(name) for name in joints]
    limits = np.asarray(env["joint_limits"],np.float64)[cols].T
    role = spec["role"]
    timing = next(row for row in labels["batches"] if row["seed"] == seed)["roles"][role]
    path = folder/f"trace-{role}.npz"
    assert sha(path) == proof["arms"][role]["trace_sha256"] == timing["trace_sha256"]
    supine = [row for row in timing["rows"] if row["actual_release"] == "supine"]
    ids = [row["world"] for row in supine]
    success = np.asarray([row["passed"] for row in supine])
    assert (~success).sum() == timing["actor"]["failed"] and success.sum() == timing["actor"]["successful"]
    output = {}
    with np.load(path,allow_pickle=False) as saved:
        times = saved["time"]
        for stage in ("early","late"):
            lower,upper = spec[f"{stage}_window_seconds"]
            rows = np.flatnonzero((times >= lower-1e-8) & (times <= upper+1e-8))
            assert len(rows) == round((upper-lower)/.02)+1
            block = np.ix_(rows,ids,cols)
            active = saved["active"][np.ix_(rows,ids)]
            double = saved["foot_contact"][rows][:,ids].all(axis=-1)
            head = saved["native_head_leg_contact"][np.ix_(rows,ids)]
            target = [saved[name][block] for name in ("requested_targets","applied_targets","joint_positions")]
            group = {}
            for name,indices in (("passed",success),("failed",~success)):
                mask = active & indices[None,:]
                assert mask.any()
                group[name] = dict(all=summarize(target,mask,limits,joints,spec),
                    simultaneous_support_and_contact=summarize(target,mask&double&head,limits,joints,spec),
                    double_support_fraction=float(double[mask].mean()),head_leg_fraction=float(head[mask].mean()))
            output[stage] = group
    return dict(seed=seed,trace=path.relative_to(ROOT).as_posix(),trace_sha256=sha(path),windows=output)


def main():
    spec = read(PLAN)
    assert spec["status"] == "frozen-archive-diagnostic-only" and not OUT.exists()
    assert not any(spec[key] for key in ("physical_substeps","ppo_updates","heldout_used","hardware_io"))
    project = read(ROOT/"project.json")
    assert project["goal_execution"]["status"] == "active" and project["current_phase"] == spec["stage"]
    assert ROOT/project["staged_training_request"]["current_protocol"] == PLAN
    archive_path,label_path = (ROOT/spec[key] for key in ("archive_protocol","timing_labels"))
    archive,labels = read(archive_path),read(label_path)
    assert not labels["heldout_used"] and labels["status"] == "passed-saved-signal-readback"
    with ThreadPoolExecutor(max_workers=len(spec["seeds"])) as pool:
        batches = list(pool.map(lambda seed:review(seed,spec,archive,labels),spec["seeds"]))
    OUT.mkdir()
    write(OUT/"analysis.json",dict(status="completed-support-target-archive-readback",protocol_sha256=sha(PLAN),
        source_sha256=sha(Path(__file__)),archive_sha256=sha(archive_path),labels_sha256=sha(label_path),batches=batches,
        physical_substeps=0,ppo_updates=0,heldout_used=False,hardware_io=False,dynamic_trial_authorized=False))
    for batch in batches:
        print("SUPINE_SUPPORT_TARGET_READBACK",batch["seed"],batch["windows"]["late"]["failed"]["all"],flush=True)


if __name__ == "__main__":
    main()
