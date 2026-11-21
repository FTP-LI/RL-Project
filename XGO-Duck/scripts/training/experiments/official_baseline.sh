#!/usr/bin/env bash
# Experiment: official XGO-Duck baseline, trained from scratch.
# What changes: no reward, robot, actuator or PPO algorithm change.
# Why: establish the reference before attributing any effect to reward changes.
# Controlled overrides: environment count, iteration budget, seed, logging/video
# are set by configs/experiments/formal_velocity_tracking.json and the runner.
# Expected result: a learning curve and checkpoints, NOT an automatic gait pass.
# Validate: saved source/config hashes, fixed-command survival and tracking error.
# This entry does NOT resume model_250.pt. Existing runs are never overwritten.
set -euo pipefail
export XGODUCK_EXPERIMENT_ENTRY="$(realpath "${BASH_SOURCE[0]}")"
training_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$training_dir/run_visible_training.sh" --mode baseline "$@"
