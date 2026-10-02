import os

SIMIT_ROOT = os.environ.get("SIMIT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIMIT_DATA = os.environ.get("SIMIT_DATA") or os.path.join(SIMIT_ROOT, "workdir")

BAGEL_REPO_ROOT = os.path.join(SIMIT_ROOT, "Bagel")
BAGEL_MODEL_PATH = os.path.join(BAGEL_REPO_ROOT, "models", "BAGEL-7B-MoT")
BAGEL_VENV_PYTHON = os.path.join(BAGEL_REPO_ROOT, ".venv", "bin", "python")

# Symlinked as pylibs/bagel -> Bagel/, which is what makes `import bagel.xxx`
# resolve (see lmms_eval/models/simple/bagel.py, and run_bagel_eval.sh's
# PYTHONPATH). Reusing this import path keeps our checkpoint format and model
# construction identical to the existing eval path with zero glue code.
PYLIBS_ROOT = os.path.join(SIMIT_ROOT, "lmms-eval", "pylibs")

# The unlabeled eval queries (original_sample.png / _prompt.txt), as saved by
# similar_samples_llms_eval_lite/run_synthesis.sh; the synthesized samples next to
# them are never read.
REAL_SAMPLES_ROOT = os.path.join(SIMIT_DATA, "generated_samples_discard_unverified_no_diff_filtering_4_samples")

# mathvista is not one of the paper's 17 benchmarks.
EXCLUDED_BENCHMARKS = {"mathvista"}

CHECKPOINT_ROOT = os.path.join(SIMIT_DATA, "checkpoints", "lfrl")
