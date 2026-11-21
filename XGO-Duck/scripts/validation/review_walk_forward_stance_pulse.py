"""Reconstruct signed actuator-only responses for observed single-support states."""

from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_stance_feedback import StanceObserver


OUT = ROOT/"records/reports/common-walk-right-support-yaw-training"
PLAN = ROOT/"configs/experiments/common_walk_forward_stance_observation.json"
PULSE = ROOT/"configs/experiments/common_walk_yaw_history_probe.json"
PULSE_RESULT = ROOT/"records/reports/common-walk-yaw-history-probe/analysis.json"


def side_response(delta, signs, mask, amplitude, rng):
    groups = [delta[:, mask & (signs == sign)]*sign/amplitude for sign in (-1, 1)]
    counts = [len(row.T) for row in groups]
    enough = min(counts) >= 3 and sum(counts) >= 8
    if min(counts) == 0:
        return dict(sign_counts=counts, enough=False, admitted=False)
    response = sum(row.mean(axis=1) for row in groups)/2
    draws = sum(row[:, rng.integers(len(row.T), size=(2000, len(row.T)))].mean(axis=2) for row in groups)/2
    first, short = (np.quantile(values, [.025, .975]).tolist() for values in (draws[0], draws[:3].mean(axis=0)))
    return dict(sign_counts=counts, enough=enough, mean_slope_per_step=response.tolist(),
        first_step_ci95=first, first_60ms_ci95=short, admitted=bool(enough and first[0] > 0. and short[0] > 0.))


def main():
    frozen, spec, prior = read(PLAN), read(PULSE), read(PULSE_RESULT)
    observation_file = OUT/"forward-stance-observation-cpu.json"
    observer_record = read(observation_file)
    assert observer_record["protocol_sha256"] == sha(PLAN) and observer_record["admitted_for_control_design"]
    assert prior["protocol_sha256"] == sha(PULSE) and not prior["heldout_used"]
    assert prior["matched_applied_targets"] and prior["actuator_raw_and_requested_unchanged"]
    record = ROOT/prior["record"]
    forward = spec["environment"]["groups"]["forward"]
    signs = np.asarray(spec["phase"]["pulse_signs"])[forward]
    source = record/"actuator-pulse-readback.npz"
    assert sha(source) == prior["readback_sha256"]
    with np.load(source, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["enabled"].nonzero()[0], np.sort(forward))
    prefix = record/"trace-shared-prefix.npz"
    with np.load(prefix, allow_pickle=False) as archive:
        feet = archive["contact"][-1, forward]
        assert archive["active"][-1, forward].all()
    phase = feet[:,0].astype(int)+2*feet[:,1]
    phase = np.where(phase == 3, 0, phase)
    history = {}
    for role in ("reference", "actuator-pulse"):
        path = record/f"trace-{role}.npz"
        assert sha(path) == prior["trace_sha256"][role]
        with np.load(path, allow_pickle=False) as archive:
            assert archive["active"][:6, forward].all()
            history[role] = (archive["observations"][0,forward], archive["yaw_rate"][:6,forward].astype(np.float64))
    np.testing.assert_array_equal(history["reference"][0], history["actuator-pulse"][0])
    observer = StanceObserver(observer_record["model"], frozen["minimum_score_separation"])
    inferred = observer.predict(history["reference"][0])
    delta = history["actuator-pulse"][1]-history["reference"][1]
    amplitude = spec["phase"]["peak_rad_s"]
    np.testing.assert_allclose((delta[0]*signs).mean(), prior["response"]["actuator_20ms_rad_s"], atol=1e-12)
    rng = np.random.default_rng(1155)
    sides = {}
    for label, side in (("left-only", 1), ("right-only", 2)):
        sides[label] = dict(actual=side_response(delta, signs, phase == side, amplitude, rng),
            confident=side_response(delta, signs, (phase == side) & (inferred == side), amplitude, rng),
            inferred_incorrect=int(((inferred == side) & (phase != side)).sum()))
    admit = all(row["confident"]["admitted"] and row["inferred_incorrect"] == 0 for row in sides.values())
    write(OUT/"forward-stance-pulse-readback.json", dict(status="checked-archived-stance-pulse-response",
        source_sha256=sha(Path(__file__)), observer_sha256=sha(observation_file), pulse_protocol_sha256=sha(PULSE),
        pulse_result_sha256=sha(PULSE_RESULT), prefix_trace_sha256=sha(prefix), world_count=len(forward),
        pulse_amplitude_rad_s=amplitude, signed_amplitude_normalized_response=sides,
        archived_sign_admitted=admit, physical_substeps=0, policy_updates=0, heldout_used=False))
    print("FORWARD_STANCE_PULSE", admit, sides, flush=True)


if __name__ == "__main__":
    main()
