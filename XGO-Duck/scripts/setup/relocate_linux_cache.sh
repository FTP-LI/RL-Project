#!/usr/bin/env bash
# Run visibly after the initial bootstrap has finished, never during uv sync.
set -euo pipefail
mode="${1:---verify-offline}"
[[ "$mode" == '--move-only' || "$mode" == '--verify-offline' ]] || exit 2
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
record_dir="$project_root/records/deployment/$(TZ=Asia/Shanghai date +%F)"
mkdir -p "$record_dir"
exec > >(tee -a "$record_dir/cache-relocation-$(date +%H%M%S).log") 2>&1
trap 'result=$?; printf "CACHE_RELOCATION_EXIT_CODE=%s\n" "$result"' EXIT
if pgrep -u "$(id -u)" -x uv >/dev/null; then
    echo 'A uv process is still active; leave its cache in place and retry after it finishes.'
    exit 1
fi
source_dir="$HOME/.cache/uv"
destination_dir="$HOME/XGO-Duck/cache/uv"
if [[ -d "$source_dir" ]]; then
    [[ "$(realpath -e "$source_dir")" == "$source_dir" ]]
    [[ "$(realpath -m "$destination_dir")" == "$HOME/XGO-Duck/cache/uv" ]]
    [[ ! -e "$destination_dir" ]]
    mkdir -p "$HOME/XGO-Duck/cache"
    printf 'Moving this setup cache: %s -> %s\n' "$source_dir" "$destination_dir"
    mv -- "$source_dir" "$destination_dir"
fi
source "$project_root/scripts/setup/project_environment.sh"
mkdir -p "$CUDA_CACHE_PATH" "$TORCH_HOME" "$TRITON_CACHE_DIR"
printf 'Project uv cache: %s\n' "$(uv cache dir)"
du -sh "$UV_CACHE_DIR"
cd "$HOME/XGO-Duck/workspace/xgoduck_rl"
if [[ "$mode" == '--verify-offline' ]]; then
    uv sync --locked --offline
else
    echo 'Cache relocation only; dependency completeness has NOT been verified.'
fi
