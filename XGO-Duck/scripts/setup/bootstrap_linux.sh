#!/usr/bin/env bash
# Run in a visible terminal. Do not launch with nohup, &, or a scheduled task.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
linux_root="$HOME/XGO-Duck"
mode="${1:---mirror}"
[[ "$mode" == '--mirror' || "$mode" == '--official-index' || "$mode" == '--plan-only' ]] || {
    echo 'Usage: bootstrap_linux.sh [--mirror|--official-index|--plan-only]'; exit 2;
}
if [[ "$mode" == '--plan-only' ]]; then
    printf 'Project: %s\nDedicated Linux work/cache: %s\n' "$project_root" "$linux_root"
    printf '%s\n' 'Plan: inspect GPU/Python -> verify archive -> ensure uv -> sync official lock via mirror -> check CUDA/tasks.'
    printf '%s\n' 'No shared-cache relocation, driver installation, training, or reboot. No changes made in plan mode.'
    exit 0
fi
record_dir="$project_root/records/deployment/$(TZ=Asia/Shanghai date +%F)"
mkdir -p "$record_dir" "$linux_root/tools" "$linux_root/workspace"
log_file="$record_dir/linux-bootstrap-$(date +%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1
trap 'result=$?; printf "\nBOOTSTRAP_EXIT_CODE=%s\n" "$result"' EXIT
set -x
date -Iseconds
cat /etc/os-release
python3 --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h "$linux_root"
# Fresh/native hosts may have neither uv nor a project cache. Do not invoke
# the historical one-time WSL cache relocation or move another project's cache.
source "$project_root/scripts/setup/project_environment.sh"
mkdir -p "$UV_CACHE_DIR" "$CUDA_CACHE_PATH" "$TORCH_HOME" "$TRITON_CACHE_DIR"
python3 "$project_root/scripts/setup/prepare_training_copy.py" --destination "$linux_root/workspace/xgoduck_rl"

export PATH="$linux_root/tools:$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    # Vendor-owned installer; keep it as evidence and do not change shell profiles.
    curl --fail --location --connect-timeout 20 --max-time 120 \
      https://astral.sh/uv/0.13.6/install.sh -o "$linux_root/tools/uv-install.sh"
    sha256sum "$linux_root/tools/uv-install.sh"
    UV_INSTALL_DIR="$linux_root/tools" UV_NO_MODIFY_PATH=1 sh "$linux_root/tools/uv-install.sh"
fi
uv --version
cd "$linux_root/workspace/xgoduck_rl"
export UV_HTTP_TIMEOUT=600
export UV_HTTP_CONNECT_TIMEOUT=60
export PYTHONUNBUFFERED=1
if [[ "$mode" == '--mirror' ]]; then
    # Use the project's required Python even if native Ubuntu's system Python
    # is older. uv may provision it; this remains part of the visible install.
    uv run --no-project --python 3.12 python "$project_root/scripts/setup/sync_locked_mirror.py"
else
    uv sync --locked
fi
uv run --locked python -c 'import torch, importlib.metadata as m; print({k:m.version(k) for k in ["torch","mjlab","warp-lang","mujoco","onnxruntime"]}); print("CUDA", torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(256,256,device="cuda"); print((x@x.T).mean().item()); torch.cuda.synchronize()'
uv run --locked --offline python -c 'from mjlab.scripts.list_envs import list_environments; from mjlab.tasks.registry import list_tasks; list_environments(); assert "Mjlab-Velocity-Flat-XgoDuck" in list_tasks()'
uv run --locked train Mjlab-Velocity-Flat-XgoDuck --help > "$record_dir/train-help.txt"
uv run --locked play Mjlab-Velocity-Flat-XgoDuck --help > "$record_dir/play-help.txt"
uv pip freeze > "$record_dir/python-packages.txt"
sha256sum pyproject.toml uv.lock
printf '\nEnvironment checks passed. Training has NOT been run by this script.\n'
