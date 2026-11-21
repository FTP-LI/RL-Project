"""Independently score original holds and matched-pair initial-state difficulty."""

import numpy as np
import onnxruntime as ort

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_getup_anchor_same_release import verify_video
from verify_getup_left_hip_load_capture import arrays
from verify_getup_left_hip_pulse_trial import summarize
import run_getup_hard_start_screen as run


def verify_reset(record,fixture,spec):
    zero = read(run.OUT/"native-zero.json")
    assert zero == read(record/"native-zero-readback.json") and zero["status"] == "passed-native-bank-zero"
    assert zero["physical_substeps"] == 0 and zero["startup"]["protocol_sha256"] == sha(run.PLAN)
    path = run.OUT/"reset-readback.npz"
    assert sha(path) == zero["reset_sha256"]
    data = arrays(path)
    mask = fixture["group"] != 2
    np.testing.assert_array_equal(data["mask"],mask)
    np.testing.assert_array_equal(data["after"][~mask],data["before"][~mask])
    np.testing.assert_array_equal(data["after"][mask,2:7],fixture["planned"][mask])
    np.testing.assert_array_equal(data["before"][:,:2],data["after"][:,:2])
    np.testing.assert_array_equal(data["before"][:,7:],data["after"][:,7:])
    assert not data["qvel"][:,:6].any()
    return dict(actual_reset_changes=int(np.any(data["after"]!=data["before"],axis=1).sum()),
        nominal_exact=int((~mask).sum()),unchanged_joint_coordinates=True,root_velocity_zero=True)


def contrasts(g,pairs,fixture,passed):
    group,pair,slot = (fixture[key] for key in ("group","pair","slot"))
    n = len(pairs)
    assert set(pair) == set(range(n))
    counts = np.empty((3,n),int)
    failures = np.empty((3,n),int)
    order = np.argsort(slot[group==0])
    for role in range(3):
        mask = group == role
        indices = np.argsort(slot[mask])
        np.testing.assert_array_equal(slot[mask][indices],slot[group==0][order])
        np.testing.assert_array_equal(pair[mask][indices],pair[group==0][order])
        counts[role] = np.bincount(pair[mask],minlength=n)
        failures[role] = np.bincount(pair[mask&~passed],minlength=n)
    assert np.all(counts > 0) and np.array_equal(counts[0],counts[1]) and np.array_equal(counts[0],counts[2])
    rng = np.random.default_rng(g["bootstrap_seed"])
    indices = rng.integers(n,size=(g["bootstrap_draws"],n))
    result = {}
    for i,role in enumerate(run.ROLES[1:],start=1):
        delta = failures[0]-failures[i]
        gap = float(delta.sum()/counts[0].sum())
        interval = np.quantile(delta[indices].sum(axis=1)/counts[0][indices].sum(axis=1),[.025,.975]).tolist()
        result[role] = dict(failure_rate_excess=gap,block_interval95=interval,
            positive_pairs=int((delta>0).sum()),negative_pairs=int((delta<0).sum()),tied_pairs=int((delta==0).sum()),
            admitted=gap >= g["minimum_failure_rate_excess_each_reference"] and interval[0] > g["minimum_lower_interval"])
    primary = np.array([row["source_seed"] == 1157 for row in pairs])
    differences = failures[0]-failures[1]
    by_source = dict(primary_failure_rate_excess=float(differences[primary].sum()/counts[0][primary].sum()),
        nonprimary_failure_rate_excess=float(differences[~primary].sum()/counts[0][~primary].sum()))
    return result,by_source,dict(count=counts[0].tolist(),failures={key:failures[i].tolist() for i,key in enumerate(run.ROLES)})


def main():
    spec,record,fixture = run.checked()
    p,d = spec["environment"],spec["draft"]
    state = read(record/"evaluation.json")
    assert state["status"] == "completed" and state["actor_weights_unchanged"] and state["protocol_sha256"] == sha(run.PLAN)
    assert state["controls"] == state["attempted_controls"] == d["maximum_controls"]
    assert state["substeps"] == state["attempted_substeps"] == d["maximum_batched_substeps"]
    assert state["transitions"] == d["maximum_transitions"] and (record/"execution-exit.txt").read_text().strip() == "0"
    reset = verify_reset(record,fixture,spec)
    files = {filename:state[field] for filename,field in (("trace.npz","trace_sha256"),
        ("initial-video-states.npz","initial_video_sha256"),("policy-replay.npz","policy_replay_sha256"))}
    for filename,expected in files.items():
        assert sha(record/filename) == expected
    trace,initial,replay = (arrays(record/filename) for filename in files)
    n = p["num_envs"]
    assert trace["active"].shape == (550,n) and trace["active"].all()
    trace["requested_pose"] = np.full(n,"supine")
    timely,final,poses,passed = summarize(trace,p)
    passed &= poses == "supine"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    runtime = ort.InferenceSession(str(ROOT/p["model"]),sess_options=options,providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(replay["control"],[101,150,200,300,550])
    errors = []
    for x,y in zip(replay["obs"],replay["raw"],strict=True):
        expected = np.concatenate([runtime.run(None,{"obs":row[None]})[0] for row in x])
        np.testing.assert_allclose(y,expected,atol=2e-5,rtol=2e-5)
        errors.append(float(abs(y-expected).max()))
    arms,videos = {},{}
    for i,role in enumerate(run.ROLES):
        mask = fixture["group"] == i
        assert mask.sum() == d["worlds_per_condition"]
        actual = mask&(poses == "supine")
        world = spec["cameras"][role]
        info = state["videos"][role]
        assert world == info["environment_index"] == int(initial["camera_worlds"][i]) and mask[world]
        # qpos was saved only for the three verified camera worlds.
        video = verify_video(record,dict(info,environment_index=i),trace,initial,i)
        videos[role] = dict(video,world=world,release_pose=poses[world])
        arms[role] = dict(worlds=int(mask.sum()),actual_supine=int(actual.sum()),combined_holds=int((passed&mask).sum()),
            failures=int((mask&~passed).sum()),failure_rate=float((mask&~passed).sum()/mask.sum()))
    g = d["admission"]
    comparison,sources,paired = contrasts(g,spec["bank_pairs"],fixture,passed)
    gates = dict(actual_pose_coverage=all(row["actual_supine"]/row["worlds"] >= g["minimum_actual_supine_fraction_each"] for row in arms.values()),
        contrasts=all(row["admitted"] for row in comparison.values()),
        widespread_difficulty=comparison["matched"]["positive_pairs"]/len(spec["bank_pairs"]) >= g["minimum_positive_matched_pair_fraction"],
        nonprimary_sign=sources["nonprimary_failure_rate_excess"]>=0.)
    observer = read(record/"windows-view-verification.json")
    assert observer["status"] == "passed" and observer["running_snapshots_with_original_frame"] >= 3
    assert len({row["frame"]["sha256"] for row in observer["observations"] if "frame" in row}) >= 3
    assert observer["observations"][-1]["latest_clip"]["sha256"] == videos["challenge"]["sha256"]
    gpu = read(record/"sample-gpu-telemetry.json")["samples"]
    assert gpu and all(row["used_mib"] <= d["gpu_memory_fraction_cap"]*row["total_mib"] for row in gpu)
    result = run.OUT/"analysis.json"
    assert not result.exists()
    write(result,dict(status="passed-independent-hard-start-screen",record=spec["record"],protocol_sha256=sha(run.PLAN),
        transitions=state["transitions"],physical_substeps=state["substeps"],sampling_seconds=state["sampling_seconds"],
        peak_gpu_mib=max(row["used_mib"] for row in gpu),gpu_percent_mean=float(np.mean([row["gpu_percent"] for row in gpu])),
        observer_snapshots=observer["running_snapshots"],policy_import_max_error=max(errors),native_reset=reset,
        arms=arms,comparison=comparison,source_split=sources,pairs=paired,decision=gates,admitted_to_curriculum=all(gates.values()),
        videos=videos,learning_updates=0,heldout_used=False,hardware_io=False,
        scope="Fresh all-supine development screen under one fixed model; four-pose and heldout gates remain separate."))
    print("GETUP_HARD_START_VERIFIED",arms,comparison,sources,gates,flush=True)


if __name__ == "__main__":
    main()
