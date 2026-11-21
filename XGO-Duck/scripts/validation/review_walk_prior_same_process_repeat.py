"""Check whether an earlier same-process reference replay was bitwise deterministic."""

from pathlib import Path

import numpy as np

from run_walk_routed_right_onset import OUT
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def main():
    evidence = ROOT / "records/reports/common-walk-forward-phase/analysis.json"
    plan = ROOT / "configs/experiments/common_walk_forward_phase_screen.json"
    proof, spec = read(evidence), read(plan)
    assert proof["status"] == "passed-integrity" and proof["protocol_sha256"] == sha(plan)
    record = ROOT / proof["record"]
    names = ("reference-a", "reference-b")
    paths = [record / f"trace-{name}.npz" for name in names]
    assert [sha(path) for path in paths] == [proof["trace_sha256"][name] for name in names]
    fork = read(record / "fork-readback.json")
    assert fork[names[0]] == fork[names[1]] and fork[names[0]]["physical"]["counters"] == [50, 250]
    stats = {}
    with np.load(paths[0], allow_pickle=False) as left, np.load(paths[1], allow_pickle=False) as right:
        for field in ("observations", "actions", "requested_joint_targets", "applied_joint_targets",
                      "qpos", "velocity", "yaw_rate", "contact"):
            a, b = left[field][:50], right[field][:50]
            hits = np.flatnonzero((a != b).reshape(len(a), -1).any(axis=1))
            delta = abs(a.astype(np.float64)-b.astype(np.float64))
            stats[field] = dict(first_different_suffix_control=int(hits[0])+1 if len(hits) else None,
                first_control_maximum_difference=float(delta[0].max()), maximum_difference=float(delta.max()))
    result = dict(status="verified-archived-same-process-reference-noise", source_sha256=sha(Path(__file__)),
        source_report_sha256=sha(evidence), source_protocol_sha256=sha(plan), seed=spec["environment"]["seed"],
        compared_suffix_controls=50, fork_state_sha256=fork[names[0]]["physical"]["state_sha256"],
        shared_boundary_exact=True, fields=stats, physical_substeps=0, learning_updates=0,
        scope="Earlier official model and control-50 branch only. Current routed policy and control-200 boundary remain untested.")
    write(OUT / "prior-same-process-repeat.json", result)
    print("PRIOR_SAME_PROCESS_FIRST_DIFF",
          {key: row["first_different_suffix_control"] for key, row in stats.items()}, flush=True)


if __name__ == "__main__":
    main()
