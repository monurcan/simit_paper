"""Shared paths/constants for the label-noise / model-noise audit (Sec 4.5).

Read-only with respect to the rest of the codebase: every path here points
into existing project outputs, nothing here is written to by other code.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[3])
SIMIT_DATA = _os.environ.get("SIMIT_DATA") or _os.path.join(SIMIT_ROOT, "workdir")
import json
import os

ANALYSIS_ROOT = SIMIT_ROOT + "/analyses/label_noise_model_noise"
CACHE_DIR = os.path.join(ANALYSIS_ROOT, "cache")
EVAL_PACKETS_DIR = os.path.join(ANALYSIS_ROOT, "eval_packets")
RESULTS_DIR = os.path.join(ANALYSIS_ROOT, "results")
PROVENANCE_DIR = os.path.join(ANALYSIS_ROOT, "provenance")

# The 17 benchmarks used project-wide (mathvista excluded, confirmed in
# LFRL_Baselines/common/paths.py EXCLUDED_BENCHMARKS and Table 5 of the paper).
BENCHMARKS = [
    "ok_vqa_val2014", "seedbench", "flickr30k_test", "coco2017_cap_val",
    "textvqa_val", "docvqa_val", "gqa", "vqav2_val", "ai2d", "nocaps_val",
    "mmbench_en_dev", "mmbench_cn_dev", "chartqa", "refcoco_bbox_val",
    "infovqa_val", "textcaps_val", "vizwiz_vqa_val",
]
assert len(BENCHMARKS) == 17

# lmms-eval "lite" task name per benchmark (used to locate the baseline eval jsonl).
LITE_TASK = {b: f"{b}_lite" for b in BENCHMARKS}

BASELINE_EVAL_DIR = SIMIT_ROOT + "/lmms-eval/benchmark_eval_outputs/pretrained_baseline/models__BAGEL-7B-MoT"

# Real-query (image, question, GT) triplets + up to 6 synthesized practice
# candidates per query, keyed by sampleN == doc_id (positional join, verified).
GENERATED_SAMPLES_ROOT = SIMIT_DATA + "/generated_samples_discard_unverified_v2"

# LFRL_Baselines project (TTRL/EMPO/TTRV/SRLM) -- separate sibling tree, not under Bagel/.
LFRL_ROOT = SIMIT_ROOT + "/LFRL_Baselines"
LFRL_JOB_OUTPUTS = os.path.join(LFRL_ROOT, "job_outputs")
LFRL_CHECKPOINTS_ROOT = SIMIT_DATA + "/checkpoints/lfrl"

# Canonical run_name per (method, benchmark): the "best per benchmark" run
# the user identified directly by comparing final full-benchmark eval scores
# across every hyperparameter variant that was tried (NOT a naive
# f"{method}_{bench}" formula -- that is wrong for ttrv/ai2d and 3 srlm
# benchmarks, where a non-default hyperparameter variant actually won).
# Single source of truth: provenance/canonical_lfrl_runs.json.
_CANONICAL_RUNS = json.load(open(os.path.join(ANALYSIS_ROOT, "provenance", "canonical_lfrl_runs.json")))

METHODS = ["ttrl", "empo", "ttrv", "srlm"]
CHECKPOINT_SUBDIR = {"ttrl": "ttrl", "empo": "ttrl", "ttrv": "ttrv", "srlm": "srlm"}


def run_name(method: str, bench: str) -> str:
    return _CANONICAL_RUNS[method][bench]["run_name"]


def eval_jsonl(method: str, bench: str) -> str:
    return os.path.join(LFRL_ROOT, _CANONICAL_RUNS[method][bench]["eval_jsonl"])

# SIMIT DF+ASC selection script (imported read-only, never modified).
BAGEL_ROOT = SIMIT_ROOT + "/Bagel"
DFADAPTIVE_SCRIPT = os.path.join(BAGEL_ROOT, "prepare_vqa_data_v2_dfadaptive.py")

DEFAULT_SEED = 42
DEFAULT_N_PER_BENCH = 10

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EVAL_PACKETS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROVENANCE_DIR, exist_ok=True)
