#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

python -m pendulum_lab.scripts.check_model --config configs/double_rotary_pendulum.json
python -m pendulum_lab.scripts.random_rollout --config configs/double_rotary_pendulum.json

