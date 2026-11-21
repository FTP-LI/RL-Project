"""Extract paired original supine reset coordinates from development archives."""

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity


PLAN = ROOT/"configs/experiments/getup_supine_hard_start_bank.json"
OUT = ROOT/"records/reports/getup-hard-start-bank"


def pinned(path,checksum):
    target = ROOT/path
    assert sha(target) == checksum, target
    return read(target)


def outcome(early,final):
    episodes = early["episodes"]
    requested = np.array([row["requested_pose"] for row in episodes])
    supine = (requested == "supine")&np.array([row["release_pose"] == "supine" for row in episodes])
    passed = np.array([row["hold_start_seconds"] is not None for row in episodes]) & np.asarray(final["stable_by_environment"])
    return requested,supine,passed


def pair(p,seed,record,requested,hard,good):
    archive = ROOT/record/"initial.npz"
    expected = p["initial_file_sha256"][str(seed)]
    assert sha(archive) == expected
    with np.load(archive,allow_pickle=False) as data:
        np.testing.assert_array_equal(data["requested_pose"],requested)
        state = data["qpos"][:,2:7]
        if "qvel" in data:
            assert not data["qvel"][:,:6].any()
    a,b = np.flatnonzero(hard),np.flatnonzero(good)
    assert len(b) >= len(a) and not np.intersect1d(a,b).size
    g = projected_gravity(state[:,1:])
    both = np.r_[a,b]
    lo,hi = p["reset_height_bounds_m"]
    assert ((state[both,0] >= lo-1e-7)&(state[both,0] <= hi+1e-7)).all()
    assert (g[both,0] <= -np.cos(np.deg2rad(p["reset_orientation_abs_max_deviation_deg"]))).all()
    cost = np.square((state[a,None,0]-state[b,0])/.001)
    cost += np.square((g[a,None,1:]-g[b,1:])/.02).sum(axis=-1)
    left,right = linear_sum_assignment(cost)
    assert np.array_equal(left,np.arange(len(a)))
    matched = b[right]
    rows = [dict(source_seed=seed,hard_world=int(i),matched_world=int(j),
        reset_height_difference_mm=float(abs(state[i,0]-state[j,0])*1000.),
        gravity_direction_separation_deg=float(np.rad2deg(np.arccos(np.clip(g[i]@g[j],-1.,1.)))))
        for i,j in zip(a,matched,strict=True)]
    return state[a],state[matched],rows,dict(seed=seed,record=record,initial_sha256=expected,
        hard=len(a),matched=len(matched),available_successes=len(b))


def main():
    p = read(PLAN)
    active = read(ROOT/"project.json")
    assert p["status"] == "frozen-saved-supine-initial-bank" and not OUT.exists()
    assert active["goal_execution"]["status"] == "active" and active["current_phase"] == p["stage"]
    assert ROOT/active["goal_execution"]["current_protocol"] == PLAN
    assert not any(p[key] for key in ("physical_substeps","learning_updates","heldout_used","hardware_io"))
    late,primary,cohort,catalogue = (pinned(*p[key]) for key in ("late_motion","primary_trial","primary_cohort","older_catalogue"))
    assert not late["timing_followup_gate_passed"] and not primary["heldout_used"] and not primary["admitted"]
    assert cohort["analysis_sha256"] == p["primary_trial"][1]
    a,b = (primary["branches"][key] for key in ("reference-a","reference-b"))
    req,eligible,sa = outcome(a["timely"],a["final"])
    req_b,eligible_b,sb = outcome(b["timely"],b["final"])
    np.testing.assert_array_equal(req,req_b)
    np.testing.assert_array_equal(eligible,eligible_b)
    shared = np.isin(np.arange(len(req)),[row["world"] for row in cohort["worlds"]])
    assert not (shared&(sa|sb)).any()
    sources = [pair(p,1157,primary["record"],req,eligible&shared,eligible&sa&sb)]
    for key,(path,checksum,role) in catalogue["archives"].items():
        old = pinned(path,checksum)
        assert old["status"] == "passed-independent-verification" and not old["heldout_used"]
        arm = old["arms"][role]
        protocol = read(ROOT/old["record"]/"protocol.json")
        assert protocol["environment"]["seed"] == int(key)
        req,eligible,ok = outcome(arm["timing"],arm["end_hold"])
        sources.append(pair(p,int(key),old["record"],req,eligible&~ok,eligible&ok))
    hard,matched,rows,receipts = zip(*sources,strict=True)
    h,m = np.concatenate(hard),np.concatenate(matched)
    assert h.shape == m.shape and h.shape[1] == 5 and np.isfinite(h).all() and np.isfinite(m).all()
    # Quaternion sign does not create an additional physical start.
    canonical = h.copy()
    canonical[:,1:] *= np.where(canonical[:,1:2] < 0.,-1.,1.)
    assert len(np.unique(canonical,axis=0)) == len(h)
    admission = (len(h) >= p["minimum_unique_hard_starts"]
        and sum(row["hard"]>0 for row in receipts) >= p["minimum_contributing_seeds"]
        and sum(row["hard"] for row in receipts[1:]) >= p["minimum_nonprimary_starts"])
    OUT.mkdir()
    target = OUT/"starts.npz"
    combined = [row for group in rows for row in group]
    np.savez_compressed(target,challenge=h,matched=m,seed=np.array([r["source_seed"] for r in combined]))
    write(OUT/"analysis.json",dict(status="completed-saved-supine-initial-bank",protocol_sha256=sha(PLAN),
        source_sha256=sha(Path(__file__)),archive_sha256=sha(target),coordinates=["height","qw","qx","qy","qz"],
        unique_hard_starts=len(h),matched_starts=len(m),sources=list(receipts),pairs=combined,
        admitted_to_fresh_parent_screen=admission,physical_substeps=0,learning_updates=0,heldout_used=False,hardware_io=False))
    print("GETUP_HARD_START_BANK",admission,len(h),list(receipts),flush=True)


if __name__ == "__main__":
    main()
