#!/usr/bin/env bash
# Experiment: linear-velocity tracking reward weight 3.0 -> 3.3 (+10%).
# What changes: only track_linear_velocity in _xgoduck_velocity_cfg; apply the
# recorded velocity_tracking.patch to the Linux working copy before this entry.
# Why: test whether stronger tracking incentive reduces fixed-command error.
# Hypothesis, not a result: tracking may improve, but smoothness/stability may fall.
# Controls: same robot/physics/PPO, seeds and budgets as the official baseline.
# This entry starts fresh. Primary comparisons also require the same host and
# restart procedure in both groups. Do not pair it directly with the resumed
# baseline as a reward-only effect; prepare a matched start/restart plan first.
# The Python entry enforces both the gait gate and the reviewed start-condition
# flag before creating a run; setting the flag alone is not comparison evidence.
# Validate: source hash must match candidate; inspect saved reward=3.3; compare
# physical metrics and video across seeds, not the differently weighted reward.
# Do not run until the baseline gait gate passes. This entry never edits source
# automatically and never resumes a baseline checkpoint as a candidate run.
set -euo pipefail
export XGODUCK_EXPERIMENT_ENTRY="$(realpath "${BASH_SOURCE[0]}")"
training_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$training_dir/run_visible_training.sh" --mode candidate "$@"
