# Source this file once per shell before running any SIMIT script:
#     source simit_env.sh
#
# SIMIT_ROOT  this repository (computed from this file's location).
# SIMIT_DATA  where large artifacts go: validation datasets (val50/), synthetic
#             sample pools (generated_samples_*), LoRA checkpoints, LFRL checkpoints.
#             Defaults to $SIMIT_ROOT/workdir; point it at a big scratch disk if needed.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _simit_env_file="${BASH_SOURCE[0]}"
else
  _simit_env_file="$0"  # zsh
fi
export SIMIT_ROOT="$(cd "$(dirname "$_simit_env_file")" && pwd)"
export SIMIT_DATA="${SIMIT_DATA:-$SIMIT_ROOT/workdir}"
unset _simit_env_file
mkdir -p "$SIMIT_DATA"

# lmms-eval imports BAGEL through lmms-eval/pylibs (a symlink to Bagel/).
export PYTHONPATH="$SIMIT_ROOT/lmms-eval/pylibs:$SIMIT_ROOT/Bagel${PYTHONPATH:+:$PYTHONPATH}"
# No OpenAI judge is used: MMBench answers are matched by lmms-eval's rule-based extractor.
export API_TYPE="${API_TYPE:-dummy}"
