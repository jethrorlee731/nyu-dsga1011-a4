#!/usr/bin/env bash
# Invoke from your part2 directory: bash t5_baseline/run.sh [arguments]
set -eo pipefail
T5_SCRATCH_ROOT="${T5_SCRATCH_ROOT:-/scratch/jnl9728}"
export T5_SCRATCH_ROOT
export CONDA_ENVS_PATH="$T5_SCRATCH_ROOT/conda-envs"
export CONDA_PKGS_DIRS="$T5_SCRATCH_ROOT/conda-pkgs"
export XDG_CACHE_HOME="$T5_SCRATCH_ROOT/cache"
export PIP_CACHE_DIR="$T5_SCRATCH_ROOT/cache/pip"
export HF_HOME="$T5_SCRATCH_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export NLTK_DATA="$T5_SCRATCH_ROOT/nltk_data"
export TMPDIR="$T5_SCRATCH_ROOT/tmp"
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HUB_CACHE" "$TMPDIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$T5_SCRATCH_ROOT/conda-envs/dsga1011-hw4-part2"
T5_BASELINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$T5_BASELINE_DIR/train_baseline.py" "$@"
