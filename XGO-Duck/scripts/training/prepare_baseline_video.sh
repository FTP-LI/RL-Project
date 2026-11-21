#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HOME/XGO-Duck/workspace/xgoduck_rl"
source_file=src/mjlab_microduck/tasks/__init__.py
actual="$(sha256sum "$source_file" | cut -d ' ' -f 1)"
if [[ "$actual" == cbc5112795e5612e546638998ecff2b65c53c97315afa94103f77da6fd8720fd ]]; then
  echo 'Restoring the one recorded reward change; retaining all previous logs and models.'
  git apply -R --check "$project_root/configs/experiments/velocity_tracking.patch"
  git apply -R "$project_root/configs/experiments/velocity_tracking.patch"
elif [[ "$actual" != c7295505c6876f7b46a0595912189b7dd8a5e30adf149d8e44172717061b035e ]]; then
  echo 'Unexpected source; refusing to overwrite.'; exit 2
fi
sha256sum "$source_file" uv.lock
bash "$project_root/scripts/training/run_visible_training.sh" --mode baseline --seed 42 --preflight
