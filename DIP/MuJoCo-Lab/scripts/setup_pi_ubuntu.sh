#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y python3 python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pendulum_lab.scripts.check_model --help

echo "Setup complete. Activate with: source .venv/bin/activate"

