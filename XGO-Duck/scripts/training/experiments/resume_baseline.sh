#!/usr/bin/env bash
# Continue the saved official baseline; do not start a fresh reward experiment.
# Changes: resume label/budget and PPO adaptive rate are restored consistently.
# Reason: the upstream loader leaves the saved label and configured PPO rate.
# Expected: 4749 additional updates from the currently saved 251 updates, with
# fresh environment/RNG histories. This is not a bitwise trajectory restart.
# Validate CPU load/config first, then first reset/rollout and fixed-command gait.
# User pause is enforced by both the supervisor and the actual child entry.
set -euo pipefail
export XGODUCK_EXPERIMENT_ENTRY="$(realpath "${BASH_SOURCE[0]}")"
training_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$training_dir/run_visible_training.sh" --mode baseline --resume "$@"
