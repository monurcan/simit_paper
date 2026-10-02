# Source this (after simit_env.sh) before running anything in Other_UMMs: puts the
# dedicated Other_UMMs venv (vllm + vllm-omni, see README) on PATH.
: "${SIMIT_ROOT:?source simit_env.sh first}"
export OTHER_UMMS_ROOT="${SIMIT_ROOT}/Other_UMMs"
export PATH="$OTHER_UMMS_ROOT/.venv/bin:$PATH"
export VIRTUAL_ENV="$OTHER_UMMS_ROOT/.venv"

# lmms-eval's Task.download() redirects its datasets cache away from HF_HOME when it
# detects a network filesystem; pin it to the regular HF datasets cache instead.
export LMMS_EVAL_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/datasets}"

# vllm==0.24.0's compiled extension needs CUDA 13 runtime libs that its pinned torch
# (cu128 wheel) does not put on the loader path ("libcudart.so.13: cannot open shared
# object file" otherwise).
export LD_LIBRARY_PATH="$OTHER_UMMS_ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
