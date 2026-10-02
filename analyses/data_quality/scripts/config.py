"""Shared paths/constants for the Table 2 data-quality audit.

Read-only with respect to the rest of the codebase: every path here points
into existing project outputs; nothing here is written to by other code.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[3])
SIMIT_DATA = _os.environ.get("SIMIT_DATA") or _os.path.join(SIMIT_ROOT, "workdir")
import json
import os

ANALYSIS_ROOT = SIMIT_ROOT + "/analyses/data_quality"
CACHE_DIR = os.path.join(ANALYSIS_ROOT, "cache")
EVAL_PACKETS_DIR = os.path.join(ANALYSIS_ROOT, "eval_packets")
RESULTS_DIR = os.path.join(ANALYSIS_ROOT, "results")
PROVENANCE_DIR = os.path.join(ANALYSIS_ROOT, "provenance")

# Same 17-benchmark set used project-wide (mathvista excluded).
BENCHMARKS = [
    "ok_vqa_val2014", "seedbench", "flickr30k_test", "coco2017_cap_val",
    "textvqa_val", "docvqa_val", "gqa", "vqav2_val", "ai2d", "nocaps_val",
    "mmbench_en_dev", "mmbench_cn_dev", "chartqa", "refcoco_bbox_val",
    "infovqa_val", "textcaps_val", "vizwiz_vqa_val",
]
assert len(BENCHMARKS) == 17

# Same 170 (10/benchmark) queries already drawn for the label-noise/model-noise
# audit -- reused verbatim (per user decision) so all 6 Table 2 settings are
# compared on the exact same queries (paired comparison, no new randomness).
SELECTED_QUERIES_PATH = SIMIT_ROOT + "/analyses/label_noise_model_noise/cache/selected_queries.jsonl"

# Bagel/prepare_vqa_data_v2_dfadaptive.py's build_records() is imported
# read-only and never modified; its SRC_ROOT is monkey-patched per setting
# (see 01_extract_settings.py) since it's hardcoded to one specific folder.
BAGEL_ROOT = SIMIT_ROOT + "/Bagel"

# Real per-benchmark DF+ASC hyperparameters, reused verbatim from the
# label-noise audit's provenance (same "best available config for all 17"
# the user already approved there).
SIMIT_DF_ASC_CONFIGS_PATH = SIMIT_ROOT + "/analyses/label_noise_model_noise/provenance/simit_df_asc_configs.json"

# Dummy DF+ASC hyperparameters for settings where disable_df=True and
# disable_adaptive_k=True: build_records() still computes compute_tau(...)
# unconditionally, but its result (and T_low/T_high) is never read on that
# code path, so any valid-looking floats are safe here.
_DUMMY_DF_ASC = {"epsilon": 0.1, "A0": 0.35, "A1": 0.6, "B": 0.2, "T_low": 0.0, "T_high": 1.0}

# Table 2 rows 2-7 (row 1 is the BAGEL-7B baseline, which has no synthesized
# data and is excluded here). Row -> (generated_samples root, disable_df,
# disable_adaptive_k, use_real_df_asc_config). Mapping confirmed with the
# user 2026-09-15:
#   - full / no_aba read from the production folder (generated_samples_
#     discard_unverified_v2, the hardcoded SRC_ROOT in prepare_vqa_data_v2_
#     dfadaptive.py) using the real per-benchmark DF+ASC config; only the
#     disable_adaptive_k flag differs between them.
#   - the remaining 4 rows all fix the sample budget (disable_df=True,
#     disable_adaptive_k=True, matching the paper's "This design fixes the
#     sample budget for the synthesis-component comparisons" text) and vary
#     only which raw-generation folder (i.e. which component was ablated at
#     generation time) they read from.
SETTINGS = {
    "full": {
        "root": SIMIT_DATA + "/generated_samples_discard_unverified_v2",
        "disable_df": None,  # per-benchmark, from SIMIT_DF_ASC_CONFIGS_PATH
        "disable_adaptive_k": False,
        "use_real_config": True,
        "table_row": "Full pipeline (ABA+DP+SL+V+DF)",
    },
    "no_aba": {
        "root": SIMIT_DATA + "/generated_samples_discard_unverified_v2",
        "disable_df": None,  # per-benchmark, from SIMIT_DF_ASC_CONFIGS_PATH
        "disable_adaptive_k": True,
        "use_real_config": True,
        "table_row": "w/o ABA",
    },
    "no_aba_no_df": {
        "root": SIMIT_DATA + "/generated_samples_discard_unverified_no_diff_filtering_4_samples",
        "disable_df": True,
        "disable_adaptive_k": True,
        "use_real_config": False,
        "table_row": "w/o ABA, DF",
    },
    "no_aba_no_df_no_dp": {
        "root": SIMIT_DATA + "/generated_samples_no_diversity_prompt_no_diff_filtering_4_samples",
        "disable_df": True,
        "disable_adaptive_k": True,
        "use_real_config": False,
        "table_row": "w/o ABA, DP, DF",
    },
    "no_aba_no_df_no_sl": {
        "root": SIMIT_DATA + "/generated_samples_force_natural_no_diff_filtering_4_samples",
        "disable_df": True,
        "disable_adaptive_k": True,
        "use_real_config": False,
        "table_row": "w/o ABA, SL, DF",
    },
    "no_aba_no_df_no_v": {
        "root": SIMIT_DATA + "/generated_samples_keep_unverified_no_diff_filtering_4_samples",
        "disable_df": True,
        "disable_adaptive_k": True,
        "use_real_config": False,
        "table_row": "w/o ABA, V, DF",
    },
}


def load_selected_queries():
    """{benchmark: [sample_id, ...]} for the shared 170-query draw."""
    by_bench = {}
    for line in open(SELECTED_QUERIES_PATH):
        row = json.loads(line)
        by_bench.setdefault(row["benchmark"], []).append(row["sample_id"])
    return by_bench


def load_simit_df_asc_configs():
    return json.load(open(SIMIT_DF_ASC_CONFIGS_PATH))


def df_asc_kwargs(setting_name: str, bench: str) -> dict:
    """(epsilon, A0, A1, B, T_low, T_high, disable_df) for build_records()."""
    cfg = SETTINGS[setting_name]
    if cfg["use_real_config"]:
        real = load_simit_df_asc_configs()[bench]
        return {
            "epsilon": real["epsilon"], "A0": real["A0"], "A1": real["A1"], "B": real["B"],
            "T_low": real["T_low"] if real["T_low"] is not None else 0.0,
            "T_high": real["T_high"] if real["T_high"] is not None else 1.0,
            "disable_df": real["disable_df"],
        }
    return {**_DUMMY_DF_ASC, "disable_df": cfg["disable_df"]}


os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EVAL_PACKETS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROVENANCE_DIR, exist_ok=True)
