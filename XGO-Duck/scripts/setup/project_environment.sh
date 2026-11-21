#!/usr/bin/env bash
# Source from setup, training and validation commands for this project.
export XDG_CACHE_HOME="$HOME/XGO-Duck/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$HOME/XGO-Duck/tools/python"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export PATH="$HOME/XGO-Duck/tools:$HOME/.local/bin:$PATH"
