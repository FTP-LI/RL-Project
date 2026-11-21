"""Choose a bounded reward scale from verified fixed-policy right-turn data."""

from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT / "records/reports/common-walk-right-startup-ppo"
SOURCE = OUT / "probe-analysis.json"
PLAN = ROOT / "configs/experiments/common_walk_right_startup_probe.json"
NEXT = ROOT / "configs/experiments/common_walk_right_startup_shared_value_draft.json"


def read_case(plan, proof, role):
    record = ROOT / plan["cases"][role]["record"]
    assert proof["record"] == plan["cases"][role]["record"]
    path = record / "native-rollout.npz"
    assert sha(path) == proof["trace_sha256"]
    with np.load(path, allow_pickle=False) as saved:
        fields = {key: saved[key][:200] for key in ("age", "yaw", "native_reward", "native_yaw_reward", "deficit")}
        done = saved["terminated"][:200] | saved["timed_out"][:200]
    valid = (fields["age"] == np.arange(1, 201)[:, None]).all(axis=0) & ~done.any(axis=0)
    yaw = fields["yaw"]
    early_blocks = yaw[50:150].reshape(2, 50, -1).mean(axis=1, dtype=np.float64)
    delayed = valid & (np.abs(early_blocks) <= .05).all(axis=0)
    other = valid & ~delayed
    assert other.any()
    return fields, delayed, other


def summary(fields, delayed, other, weight):
    def mean_deficit(ids):
        return float(-fields["deficit"][:200, ids].mean()) if ids.any() else None

    yaw_term = fields["native_yaw_reward"][50:150, delayed]
    cost = -fields["deficit"][50:150, delayed] * weight
    native = fields["native_reward"][50:150, delayed]
    return dict(valid_first_starts=int((delayed | other).sum()), delayed_worlds=int(delayed.sum()),
        delayed_mean_deficit=mean_deficit(delayed), other_mean_deficit=mean_deficit(other),
        median_native_yaw_term_unscaled=float(np.median(yaw_term)) if len(yaw_term.ravel()) else None,
        median_added_cost_per_control=float(.02*np.median(cost)) if len(cost.ravel()) else None,
        median_native_reward_per_control=float(np.median(native)) if len(native.ravel()) else None)


def main():
    plan, verified = read(PLAN), read(SOURCE)
    assert verified["status"] == "passed-independent-native-right-turn-probe" and verified["protocol_sha256"] == sha(PLAN)
    cases = {role: read_case(plan, proof, role) for role, proof in verified["cases"].items()}
    delayed = [data[1] for data in cases.values()]
    eligible = min(int(x.sum()) for x in delayed) >= 16
    native = np.concatenate([fields["native_yaw_reward"][50:150, slow].ravel() for fields, slow, _ in cases.values()])
    # Keep the added term at or below one fifth of the typical native yaw term
    # during delayed starts, rounded down to a reproducible 0.05 increment.
    increments = int(np.floor(.2 * np.median(native) / .05)) if len(native) else 0
    coefficient = min(.5, round(.05 * increments, 2))
    eligible = eligible and coefficient > 0
    result = dict(status="verified-native-right-turn-signal", source_result_sha256=sha(SOURCE),
        source_sha256=sha(Path(__file__)), early_delay_definition="Absolute mean body yaw <= 0.05 rad/s in both "
            "controls 51-100 and 101-150, uninterrupted through control 200. Exclude the two reset witnesses.",
        delayed_world_overlap=int(np.logical_and(*delayed).sum()), new_reward_coefficient=coefficient,
        selection_rule="0.20 times the pooled median native yaw reward term among delayed controls 51-150, "
                       "rounded down to 0.05 increments, maximum 0.50. At least 16 delayed starts in each pass.",
        cases={role: summary(*values, coefficient) for role, values in cases.items()}, value_draft_admitted=eligible,
        physical_substeps=0, learning_updates=0, heldout_used=False)
    output = OUT / "probe-reward-signal.json"
    write(output, result)
    if eligible:
        draft = dict(status="draft-right-turn-shared-value", source=[SOURCE.relative_to(ROOT).as_posix(), sha(SOURCE)],
            reward_scale_evidence=[output.relative_to(ROOT).as_posix(), sha(output)], reward_coefficient=coefficient,
            worlds=plan["worlds"], first_episode_maximum_controls=1000, critic_observation_stride=5,
            discounted_value_target="Average native and candidate rewards for the same complete first episodes "
                "from the fixed right-turn parent. Fit one common critic before either PPO branch. Never read "
                "a value or reward from after the first terminal transition when assembling the targets.",
            split="Split complete worlds within observed early delay groups. Fit normalization on training worlds only.",
            min_delayed_worlds=16, min_delayed_validation_worlds=4, validation_fraction=.25,
            value_learning_plan={"updates": 100, "minibatch": 8192, "learning_rate": .001},
            validation_gate={"explained_variance_min": .5, "normalized_mse_max": .25},
            evaluation="Keep original seven-command criteria and choose a fresh development seed at freeze.",
            physical_substeps_authorized=0, critic_updates_authorized=0, actor_updates_authorized=0,
            heldout_used=False, hardware_io=False)
        write(NEXT, draft)
    print("RIGHT_STARTUP_SIGNAL", result, flush=True)


if __name__ == "__main__":
    main()
