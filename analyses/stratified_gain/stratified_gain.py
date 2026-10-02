#!/usr/bin/env python3
"""
Stratified gain analysis: split samples by whether the zero-shot maj@32
baseline got them right, then for each method compute
    A_after - A_before = P(wrong->correct) - P(correct->wrong)
using per-sample correctness matched to the baseline by row position
(verified to be id-aligned across all sources for every benchmark below).

Covers all 17 benchmarks:
  - 12 bounded/binary metrics (ACC_FIELD + MMBENCH from eval_ci.py):
    plain >=0.5 threshold.
  - 5 CIDEr benchmarks (nocaps_val, flickr30k_test, textcaps_val,
    refcoco_bbox_val, coco2017_cap_val): unbounded/continuous per-image
    CIDEr, so a single threshold doesn't make sense. Uses a two-threshold
    scheme instead (see CIDER_HIGH/CIDER_LOW below) -- confidently
    correct/wrong buckets only, ambiguous middle dropped.

Run with the Bagel venv (has pycocoevalcap + pycocotools for CIDEr):
    $SIMIT_ROOT/Bagel/.venv/bin/python3 stratified_gain.py
Needs `java` on PATH (PTBTokenizer) -- confirmed available on the login node.

Requires $SIMIT_ROOT/eval_ci_tools/eval_ci.py to be importable.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[2])
import sys
import os
import json
import glob

sys.path.insert(0, SIMIT_ROOT + "/eval_ci_tools")
from eval_ci import ACC_FIELD, MMBENCH, CIDER_FIELD, acc_scores, mmbench_scores, can_infer_option, cider_per_image_scores  # noqa

# Two-threshold scheme for the unbounded CIDEr benchmarks (per-image CIDEr,
# not corpus-level). CIDEr is TF-IDF-weighted n-gram overlap against human
# references, calibrated so that ~1.0 is roughly "on par with a typical human
# reference caption"; empirically (checked on the maj@32 baseline's own
# per-image score distributions across all 5 CIDEr benches) HIGH=1.0 keeps a
# meaningful "confidently correct" bucket everywhere (26-69% of samples) while
# LOW=0.3 (near-zero meaningful n-gram overlap with any reference) keeps a
# non-trivial "confidently wrong" bucket even on the easiest benches
# (textcaps/coco2017: 6%, n=29). Samples strictly between the two thresholds
# are ambiguous and dropped entirely (not counted as either a starting bucket
# or, on the "after" side, as a transition) -- conservative by construction:
# a method only gets credit for a flip if it moves a sample all the way
# across the gap, not just into the ambiguous middle.
CIDER_HIGH = 1.0
CIDER_LOW = 0.3

ROOT = SIMIT_ROOT

BASELINE_DIR = f"{ROOT}/Bagel/benchmark_eval_outputs/selfconsistency_full_32rollouts"

ICL_TTSI_COMBO_BENCHES = {
    "nocaps_val", "seedbench", "mmbench_cn_dev", "infovqa_val",
    "coco2017_cap_val", "ai2d", "vqav2_val", "textvqa_val", "textcaps_val",
}

SIMIT_SPECIAL = {
    "ok_vqa_val2014": f"{ROOT}/Bagel/benchmark_eval_outputs/ok_vqa_val2014_dfadaptive_ok_vqa_val2014_t0012_0000150/models__BAGEL-7B-MoT/20260821_122308_samples_ok_vqa_val2014_lite.jsonl",
    "refcoco_bbox_val": f"{ROOT}/Bagel/benchmark_eval_outputs/refcoco_bbox_val_v2_roundrobin_genstudy_0000050/models__BAGEL-7B-MoT/20260817_102745_samples_refcoco_bbox_val_lite.jsonl",
    "vizwiz_vqa_val": f"{ROOT}/Bagel/benchmark_eval_outputs/vizwiz_vqa_val_v2_roundrobin_genstudy_0000125/models__BAGEL-7B-MoT/20260817_123540_samples_vizwiz_vqa_val_lite.jsonl",
    "chartqa": f"{ROOT}/Bagel/benchmark_eval_outputs/chartqa_v2_roundrobin_genstudy_0000100/models__BAGEL-7B-MoT/20260817_114157_samples_chartqa_lite.jsonl",
    "flickr30k_test": f"{ROOT}/Bagel/benchmark_eval_outputs/flickr30k_test_v2_roundrobin_genstudy_0000025/models__BAGEL-7B-MoT/20260817_135720_samples_flickr30k_test_lite.jsonl",
    "mmbench_en_dev": f"{ROOT}/lmms-eval/benchmark_eval_outputs/tta_dfasc_sweep/mmbench_en_dev_t0000/models__BAGEL-7B-MoT/20260901_022345_samples_mmbench_en_dev_lite.jsonl",
    "gqa": f"{ROOT}/Bagel/benchmark_eval_outputs/gqa_manual_v2_0000044/models__BAGEL-7B-MoT/20260822_095909_samples_gqa_lite.jsonl",
    "docvqa_val": f"{ROOT}/lmms-eval/benchmark_eval_outputs/pretrained_baseline/models__BAGEL-7B-MoT/20260727_182851_samples_docvqa_val_lite.jsonl",
}

TTRL_PATHS = {
    "ai2d": "ttrl_full_ai2d/after_ttrl_maj32/models__BAGEL-7B-MoT/20260905_075719_samples_ai2d_lite_sc32.jsonl",
    "chartqa": "ttrl_full_chartqa/after_ttrl_maj32/models__BAGEL-7B-MoT/20260905_092323_samples_chartqa_lite_sc32.jsonl",
    "coco2017_cap_val": "ttrl_altvote_coco2017_cap_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260905_225501_samples_coco2017_cap_val_lite_sc32.jsonl",
    "docvqa_val": "ttrl_full_docvqa_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260905_083812_samples_docvqa_val_lite_sc32.jsonl",
    "flickr30k_test": "ttrl_altvote_flickr30k_test/after_ttrl_maj32/models__BAGEL-7B-MoT/20260908_124815_samples_flickr30k_test_lite_sc32.jsonl",
    "gqa": "ttrl_full_gqa/after_ttrl_maj32/models__BAGEL-7B-MoT/20260905_075444_samples_gqa_lite_sc32.jsonl",
    "infovqa_val": "ttrl_full_infovqa_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260905_091749_samples_infovqa_val_lite_sc32.jsonl",
    "mmbench_cn_dev": "ttrl_full_mmbench_cn_dev/after_ttrl_maj32/models__BAGEL-7B-MoT/20260905_101824_samples_mmbench_cn_dev_lite_sc32.jsonl",
    "mmbench_en_dev": "ttrl_full_mmbench_en_dev/after_ttrl_maj32/models__BAGEL-7B-MoT/20260905_100731_samples_mmbench_en_dev_lite_sc32.jsonl",
    "nocaps_val": "ttrl_altvote_nocaps_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260908_152054_samples_nocaps_val_lite_sc32.jsonl",
    "ok_vqa_val2014": "ttrl_full_ok_vqa_val2014/after_ttrl_maj32/models__BAGEL-7B-MoT/20260905_084337_samples_ok_vqa_val2014_lite_sc32.jsonl",
    "refcoco_bbox_val": "ttrl_temp15_naive_refcoco_bbox_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_175946_samples_refcoco_bbox_val_lite_sc32.jsonl",
    "seedbench": "ttrl_full_seedbench/after_ttrl_maj32/models__BAGEL-7B-MoT/20260905_075631_samples_seedbench_lite_sc32.jsonl",
    "textcaps_val": "ttrl_altvote_textcaps_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260907_022624_samples_textcaps_val_lite_sc32.jsonl",
    "textvqa_val": "ttrl_full_textvqa_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260905_070648_samples_textvqa_val_lite_sc32.jsonl",
    "vizwiz_vqa_val": "ttrl_temp15_naive_vizwiz_vqa_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_154841_samples_vizwiz_vqa_val_lite_sc32.jsonl",
    "vqav2_val": "ttrl_full_vqav2_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260905_080537_samples_vqav2_val_lite_sc32.jsonl",
}

EMPO_PATHS = {
    "ai2d": "ttrl_altvote_ai2d/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_063149_samples_ai2d_lite_sc32.jsonl",
    "chartqa": "ttrl_lora32_lexical_chartqa/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_122636_samples_chartqa_lite_sc32.jsonl",
    "coco2017_cap_val": "ttrl_lr2e5_lexical_coco2017_cap_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_195740_samples_coco2017_cap_val_lite_sc32.jsonl",
    "docvqa_val": "ttrl_altvote_docvqa_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_034810_samples_docvqa_val_lite_sc32.jsonl",
    "flickr30k_test": "ttrl_full_flickr30k_test/after_ttrl_maj32/models__BAGEL-7B-MoT/20260909_120729_samples_flickr30k_test_lite_sc32.jsonl",
    "gqa": "ttrl_altvote_gqa/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_085524_samples_gqa_lite_sc32.jsonl",
    "infovqa_val": "ttrl_altvote_infovqa_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_035951_samples_infovqa_val_lite_sc32.jsonl",
    "mmbench_cn_dev": "ttrl_altvote_mmbench_cn_dev/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_052636_samples_mmbench_cn_dev_lite_sc32.jsonl",
    "mmbench_en_dev": "ttrl_altvote_mmbench_en_dev/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_052022_samples_mmbench_en_dev_lite_sc32.jsonl",
    "nocaps_val": "ttrl_full_nocaps_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260908_125818_samples_nocaps_val_lite_sc32.jsonl",
    "ok_vqa_val2014": "ttrl_altvote_ok_vqa_val2014/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_045831_samples_ok_vqa_val2014_lite_sc32.jsonl",
    "refcoco_bbox_val": "ttrl_temp13_refcoco_bbox_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260905_223358_samples_refcoco_bbox_val_lite_sc32.jsonl",
    "seedbench": "ttrl_altvote_seedbench/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_063126_samples_seedbench_lite_sc32.jsonl",
    "textcaps_val": "ttrl_full_textcaps_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_120035_samples_textcaps_val_lite_sc32.jsonl",
    "textvqa_val": "ttrl_altvote_textvqa_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_072653_samples_textvqa_val_lite_sc32.jsonl",
    "vizwiz_vqa_val": "ttrl_temp15_lexical_vizwiz_vqa_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_195719_samples_vizwiz_vqa_val_lite_sc32.jsonl",
    "vqav2_val": "ttrl_altvote_vqav2_val/after_ttrl_maj32/models__BAGEL-7B-MoT/20260906_080256_samples_vqav2_val_lite_sc32.jsonl",
}

TTRV_PATHS = {
    "ai2d": "ttrv_alpha025lora32_ai2d/after_ttrv_maj32/models__BAGEL-7B-MoT/20260907_212157_samples_ai2d_lite_sc32.jsonl",
    "chartqa": "ttrv_chartqa/after_ttrv_maj32/models__BAGEL-7B-MoT/20260907_074547_samples_chartqa_lite_sc32.jsonl",
    "coco2017_cap_val": "ttrv_coco2017_cap_val/after_ttrv_maj32/models__BAGEL-7B-MoT/20260907_055233_samples_coco2017_cap_val_lite_sc32.jsonl",
    "docvqa_val": "ttrv_alpha025_docvqa_val/after_ttrv_maj32/models__BAGEL-7B-MoT/20260908_015253_samples_docvqa_val_lite_sc32.jsonl",
    "flickr30k_test": "ttrv_flickr30k_test/after_ttrv_maj32/models__BAGEL-7B-MoT/20260908_173751_samples_flickr30k_test_lite_sc32.jsonl",
    "gqa": "ttrv_alpha00_gqa/after_ttrv_maj32/models__BAGEL-7B-MoT/20260907_162448_samples_gqa_lite_sc32.jsonl",
    "infovqa_val": "ttrv_alpha025_infovqa_val/after_ttrv_maj32/models__BAGEL-7B-MoT/20260907_123152_samples_infovqa_val_lite_sc32.jsonl",
    "mmbench_cn_dev": "ttrv_mmbench_cn_dev/after_ttrv_maj32/models__BAGEL-7B-MoT/20260907_024158_samples_mmbench_cn_dev_lite_sc32.jsonl",
    "mmbench_en_dev": "ttrv_mmbench_en_dev/after_ttrv_maj32/models__BAGEL-7B-MoT/20260907_093942_samples_mmbench_en_dev_lite_sc32.jsonl",
    "nocaps_val": "ttrv_nocaps_val/after_ttrv_maj32/models__BAGEL-7B-MoT/20260907_142745_samples_nocaps_val_lite_sc32.jsonl",
    "ok_vqa_val2014": "ttrv_alpha025_ok_vqa_val2014/after_ttrv_maj32/models__BAGEL-7B-MoT/20260907_151529_samples_ok_vqa_val2014_lite_sc32.jsonl",
    "refcoco_bbox_val": "ttrv_refcoco_bbox_val/after_ttrv_maj32/models__BAGEL-7B-MoT/20260907_101932_samples_refcoco_bbox_val_lite_sc32.jsonl",
    "seedbench": "ttrv_seedbench/after_ttrv_maj32/models__BAGEL-7B-MoT/20260907_075554_samples_seedbench_lite_sc32.jsonl",
    "textcaps_val": "ttrv_textcaps_val/after_ttrv_maj32/models__BAGEL-7B-MoT/20260908_153946_samples_textcaps_val_lite_sc32.jsonl",
    "textvqa_val": "ttrv_alpha00_textvqa_val/after_ttrv_maj32/models__BAGEL-7B-MoT/20260907_170151_samples_textvqa_val_lite_sc32.jsonl",
    "vizwiz_vqa_val": "ttrv_vizwiz_vqa_val/after_ttrv_maj32/models__BAGEL-7B-MoT/20260907_031030_samples_vizwiz_vqa_val_lite_sc32.jsonl",
    "vqav2_val": "ttrv_alpha025lora32_vqav2_val/after_ttrv_maj32/models__BAGEL-7B-MoT/20260908_020922_samples_vqav2_val_lite_sc32.jsonl",
}

SRLM_PATHS = {
    "ai2d": "srlm_ai2d/after_srlm_greedy/models__BAGEL-7B-MoT/20260909_120541_samples_ai2d_lite.jsonl",
    "chartqa": "srlm_chartqa/after_srlm_greedy/models__BAGEL-7B-MoT/20260911_062000_samples_chartqa_lite.jsonl",
    "coco2017_cap_val": "srlm_coco2017_cap_val/after_srlm_greedy/models__BAGEL-7B-MoT/20260910_103859_samples_coco2017_cap_val_lite.jsonl",
    "docvqa_val": "srlm_docvqa_val/after_srlm_greedy/models__BAGEL-7B-MoT/20260909_111523_samples_docvqa_val_lite.jsonl",
    "flickr30k_test": "srlm_flickr30k_test/after_srlm_greedy/models__BAGEL-7B-MoT/20260911_015049_samples_flickr30k_test_lite.jsonl",
    "gqa": "srlm_k8_gqa/after_srlm_greedy/models__BAGEL-7B-MoT/20260909_221641_samples_gqa_lite.jsonl",
    "infovqa_val": "srlm_infovqa_val/after_srlm_greedy/models__BAGEL-7B-MoT/20260910_064316_samples_infovqa_val_lite.jsonl",
    "mmbench_cn_dev": "srlm_mmbench_cn_dev/after_srlm_greedy/models__BAGEL-7B-MoT/20260911_225302_samples_mmbench_cn_dev_lite.jsonl",
    "mmbench_en_dev": "srlm_mmbench_en_dev/after_srlm_greedy/models__BAGEL-7B-MoT/20260910_123557_samples_mmbench_en_dev_lite.jsonl",
    "nocaps_val": "srlm_nocaps_val/after_srlm_greedy/models__BAGEL-7B-MoT/20260909_162235_samples_nocaps_val_lite.jsonl",
    "ok_vqa_val2014": "srlm_lora32_ok_vqa_val2014/after_srlm_greedy/models__BAGEL-7B-MoT/20260909_000836_samples_ok_vqa_val2014_lite.jsonl",
    "refcoco_bbox_val": "srlm_refcoco_bbox_val/after_srlm_greedy/models__BAGEL-7B-MoT/20260910_062819_samples_refcoco_bbox_val_lite.jsonl",
    "seedbench": "srlm_seedbench/after_srlm_greedy/models__BAGEL-7B-MoT/20260911_083304_samples_seedbench_lite.jsonl",
    "textcaps_val": "srlm_textcaps_val/after_srlm_greedy/models__BAGEL-7B-MoT/20260911_050650_samples_textcaps_val_lite.jsonl",
    "textvqa_val": "srlm_textvqa_val/after_srlm_greedy/models__BAGEL-7B-MoT/20260910_001233_samples_textvqa_val_lite.jsonl",
    "vizwiz_vqa_val": "srlm_temp13_vizwiz_vqa_val/after_srlm_greedy/models__BAGEL-7B-MoT/20260910_100133_samples_vizwiz_vqa_val_lite.jsonl",
    "vqav2_val": "srlm_vqav2_val/after_srlm_greedy/models__BAGEL-7B-MoT/20260911_052827_samples_vqav2_val_lite.jsonl",
}

LFRL_ROOT = f"{ROOT}/LFRL_Baselines"

BINARY_BENCHES = list(ACC_FIELD) + MMBENCH
ALL_BENCHES = BINARY_BENCHES + list(CIDER_FIELD)


def simit_path(bench):
    if bench in SIMIT_SPECIAL:
        return SIMIT_SPECIAL[bench]
    assert bench in ICL_TTSI_COMBO_BENCHES
    matches = glob.glob(f"{ROOT}/lmms-eval/benchmark_eval_outputs/icl_ttsi_combo/{bench}/models__BAGEL-7B-MoT/*_samples_{bench}_lite.jsonl")
    assert len(matches) == 1, (bench, matches)
    return matches[0]


def baseline_path(bench):
    matches = glob.glob(f"{BASELINE_DIR}/{bench}_lite_sc32/models__BAGEL-7B-MoT/*_samples_{bench}_lite_sc32.jsonl")
    assert len(matches) == 1, (bench, matches)
    return matches[0]


METHOD_PATH_FNS = {
    "SIMIT": simit_path,
    "TTRL": lambda b: f"{LFRL_ROOT}/eval_outputs/{TTRL_PATHS[b]}",
    "EMPO": lambda b: f"{LFRL_ROOT}/eval_outputs/{EMPO_PATHS[b]}",
    "TTRV": lambda b: f"{LFRL_ROOT}/eval_outputs/{TTRV_PATHS[b]}",
    "SRLM": lambda b: f"{LFRL_ROOT}/eval_outputs/{SRLM_PATHS[b]}",
}


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def acc_scores_fixed(bench, rows):
    """Like eval_ci.py's acc_scores, but case-insensitive for seedbench.

    Found during this analysis: self-consistency/majority-vote pipelines
    (selfconsistency_full_32rollouts, LFRL_Baselines' after_ttrl_maj32 /
    after_ttrv_maj32 runs) emit lowercase seedbench letters (e.g. "b") while
    the target 'answer' field is uppercase ("B") and SIMIT/SRLM/EMPO outputs
    also happen to be uppercase. eval_ci.py's plain `pred == answer` silently
    scores 0/500 correct for every lowercase-emitting run (confirmed on the
    raw baseline and TTRL jsonls). This does not affect any other benchmark
    (mmbench's answer/prediction are uppercase in all sources checked).
    """
    if bench == "seedbench":
        return [1.0 if r["seed_all"]["pred"].upper() == r["seed_all"]["answer"].upper() else 0.0 for r in rows]
    return acc_scores(rows, ACC_FIELD[bench])


def get_key_sequence(bench, rows):
    """A sequence usable to verify row alignment across sources."""
    if bench in MMBENCH:
        return [r["gpt_eval_score"]["index"] for r in rows]
    if bench in CIDER_FIELD:
        field = CIDER_FIELD[bench]
        return [r[field].get("image_id", r[field].get("ann_id", r[field].get("id"))) for r in rows]
    return [r.get("doc_id") for r in rows]


def main():
    results = {}  # method -> bench -> (p_w2c, p_c2w, gain, n_wrong, n_correct)
    per_bench_debug = {}

    for bench in ALL_BENCHES:
        is_cider = bench in CIDER_FIELD
        bpath = baseline_path(bench)
        brows = load_rows(bpath)
        bkeys = get_key_sequence(bench, brows)

        if is_cider:
            b_raw = cider_per_image_scores(brows, CIDER_FIELD[bench])
        elif bench in MMBENCH:
            bscores_full = mmbench_scores(brows, include_rotations=False)
            bmask = [r["gpt_eval_score"]["index"] < 1_000_000 for r in brows]
        else:
            bscores_full = acc_scores_fixed(bench, brows)
            bmask = [True] * len(brows)

        for method, path_fn in METHOD_PATH_FNS.items():
            mpath = path_fn(bench)
            mrows = load_rows(mpath)
            mkeys = get_key_sequence(bench, mrows)

            assert len(mrows) == len(brows), f"{method}/{bench}: n mismatch {len(mrows)} vs {len(brows)}"
            assert mkeys == bkeys, f"{method}/{bench}: key order mismatch"

            if is_cider:
                m_raw = cider_per_image_scores(mrows, CIDER_FIELD[bench])
                assert len(m_raw) == len(b_raw)

                # Two-threshold scheme: samples with baseline score strictly
                # between LOW and HIGH are ambiguous and dropped entirely.
                # Among the retained "confident" baseline samples, a flip only
                # counts if the after-score crosses all the way into the
                # opposite confident zone (landing back in the ambiguous
                # middle does not count as a flip either way).
                n_wrong = sum(1 for s in b_raw if s < CIDER_LOW)
                n_correct = sum(1 for s in b_raw if s >= CIDER_HIGH)
                w2c = sum(1 for bs, ms in zip(b_raw, m_raw) if bs < CIDER_LOW and ms >= CIDER_HIGH)
                c2w = sum(1 for bs, ms in zip(b_raw, m_raw) if bs >= CIDER_HIGH and ms < CIDER_LOW)
                n = n_wrong + n_correct
            else:
                if bench in MMBENCH:
                    mscores_full = mmbench_scores(mrows, include_rotations=False)
                    mmask = [r["gpt_eval_score"]["index"] < 1_000_000 for r in mrows]
                    assert mmask == bmask
                else:
                    mscores_full = acc_scores_fixed(bench, mrows)

                # mmbench_scores() already filters internally to index<1_000_000 and
                # returns scores in that filtered subset's row order; since bmask==mmask
                # (asserted above), that filtered order is identical across sources, so
                # no further slicing is needed (re-slicing with the full-length mask via
                # zip would silently truncate to min(len(scores), len(mask)) and misalign).
                b_scores = bscores_full
                m_scores = mscores_full
                assert len(b_scores) == len(m_scores)

                n = len(b_scores)
                n_wrong = sum(1 for s in b_scores if s < 0.5)
                n_correct = n - n_wrong
                w2c = sum(1 for bs, ms in zip(b_scores, m_scores) if bs < 0.5 and ms >= 0.5)
                c2w = sum(1 for bs, ms in zip(b_scores, m_scores) if bs >= 0.5 and ms < 0.5)

            p_w2c = w2c / n_wrong if n_wrong else float("nan")
            p_c2w = c2w / n_correct if n_correct else float("nan")
            gain = (w2c - c2w) / n  # = P(w->c)*P(wrong) - P(c->w)*P(correct), matches (A_after-A_before)

            results.setdefault(method, {})[bench] = {
                "n": n, "n_wrong": n_wrong, "n_correct": n_correct,
                "w2c": w2c, "c2w": c2w,
                "p_w2c": p_w2c, "p_c2w": p_c2w,
                "gain": gain,
            }

    return results


if __name__ == "__main__":
    results = main()
    methods = ["SIMIT", "TTRL", "EMPO", "TTRV", "SRLM"]

    # per-benchmark table
    for bench in ALL_BENCHES:
        tag = " (CIDEr, two-threshold)" if bench in CIDER_FIELD else ""
        print(f"\n=== {bench}{tag} ===")
        print(f"  {'method':8s} {'n':>4s} {'n_wrong':>8s} {'n_correct':>10s} {'w2c':>5s} {'c2w':>5s} {'P(w2c)':>8s} {'P(c2w)':>8s} {'gain(A_after-A_before)':>24s}")
        for method in methods:
            r = results[method][bench]
            print(f"  {method:8s} {r['n']:>4d} {r['n_wrong']:>8d} {r['n_correct']:>10d} {r['w2c']:>5d} {r['c2w']:>5d} {r['p_w2c']:>8.3f} {r['p_c2w']:>8.3f} {r['gain']*100:>22.2f}%")

    print(f"\n=== Macro-average gain (A_after - A_before), BINARY_BENCHES only (n={len(BINARY_BENCHES)}) ===")
    for method in methods:
        gains = [results[method][b]["gain"] for b in BINARY_BENCHES]
        avg = sum(gains) / len(gains)
        print(f"  {method:8s} {avg*100:+6.2f} pp")

    print(f"\n=== Macro-average gain (A_after - A_before), CIDER_FIELD only (n={len(CIDER_FIELD)}, HIGH={CIDER_HIGH}, LOW={CIDER_LOW}) ===")
    for method in methods:
        gains = [results[method][b]["gain"] for b in CIDER_FIELD]
        avg = sum(gains) / len(gains)
        print(f"  {method:8s} {avg*100:+6.2f} pp")

    print(f"\n=== Macro-average gain (A_after - A_before), ALL {len(ALL_BENCHES)} benchmarks ===")
    for method in methods:
        gains = [results[method][b]["gain"] for b in ALL_BENCHES]
        avg = sum(gains) / len(gains)
        print(f"  {method:8s} {avg*100:+6.2f} pp")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stratified_gain_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
