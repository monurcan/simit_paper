"""Replays the real ASC adaptive_k formula against real per-sample zero-shot
confidence data (avg_answer_prob, already computed and saved on disk by
compute_confidence.py) to get the TRUE realized average K for SIMIT-ICL's
winning config per benchmark -- not the swept hyperparameter ceiling used
earlier (which was an explicitly-flagged upper-bound approximation).
Read-only data analysis, no GPU/model involved, no codebase changes.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
SIMIT_DATA = _os.environ.get("SIMIT_DATA") or _os.path.join(SIMIT_ROOT, "workdir")
import json
import math
from pathlib import Path

NUM_GENERATIONS = 6  # Kmax for the v2 pool; nodf4 pool caps at 4 regardless


def compute_tau(epsilon, A0, A1, B):
    O_0 = A0 / (1 - A0)
    tau = {0: 1 - epsilon}
    for K in range(1, NUM_GENERATIONS + 1):
        A_K = min(max(A1 + B * math.log(K), 1e-3), 1 - 1e-3)
        O_K = A_K / (1 - A_K)
        tau[K] = (1 - epsilon) * O_0 / (epsilon * O_K + (1 - epsilon) * O_0)
    return tau


def adaptive_k(p0, tau):
    for K in range(0, NUM_GENERATIONS + 1):
        if p0 >= tau[K]:
            return K
    return NUM_GENERATIONS


V2_ROOT = Path(SIMIT_DATA + "/generated_samples_discard_unverified_v2")
NODF4_ROOT = Path(SIMIT_DATA + "/generated_samples_discard_unverified_no_diff_filtering_4_samples")

# SG-ICL DF+ASC winning configuration per benchmark, from the LoRA-vs-FullFT
# artifact's "SG-ICL DF+ASC — best result per benchmark" table.
# mode: "dfasc" (replay real formula), "no_mechanism" (ratio=1.0, all Kmax
# used), "degenerate" (ratio=0, empties the pool).
BENCHMARKS = {
    "ok_vqa_val2014":   {"mode": "no_mechanism", "pool_max": 6},
    "vizwiz_vqa_val":   {"mode": "no_mechanism", "pool_max": 6},
    "nocaps_val":       {"mode": "no_mechanism", "pool_max": 6},
    "textcaps_val":     {"mode": "degenerate", "pool_max": 4},
    "gqa":              {"mode": "dfasc", "pool": "v2", "pool_max": 6, "eps": 0.392, "A0": 0.556, "A1": 0.847, "B": 0.137},
    "vqav2_val":        {"mode": "dfasc", "pool": "nodf4", "pool_max": 4, "eps": 0.260, "A0": 0.128, "A1": 0.629, "B": 0.161},
    "textvqa_val":      {"mode": "dfasc", "pool": "v2", "pool_max": 6, "eps": 0.164, "A0": 0.514, "A1": 0.795, "B": 0.195},
    "chartqa":          {"mode": "dfasc", "pool": "nodf4", "pool_max": 4, "eps": 0.107, "A0": 0.176, "A1": 0.520, "B": 0.142},
    "docvqa_val":       {"mode": "dfasc", "pool": "v2", "pool_max": 6, "eps": 0.321, "A0": 0.155, "A1": 0.196, "B": 0.450},
    "infovqa_val":      {"mode": "dfasc", "pool": "nodf4", "pool_max": 4, "eps": 0.242, "A0": 0.440, "A1": 0.613, "B": 0.310},
    "flickr30k_test":   {"mode": "dfasc", "pool": "v2", "pool_max": 6, "eps": 0.324, "A0": 0.432, "A1": 0.799, "B": 0.365},
    "refcoco_bbox_val": {"mode": "dfasc", "pool": "v2", "pool_max": 6, "eps": 0.379, "A0": 0.561, "A1": 0.803, "B": 0.001},
    "coco2017_cap_val": {"mode": "dfasc", "pool": "v2", "pool_max": 6, "eps": 0.391, "A0": 0.357, "A1": 0.488, "B": 0.316},
    "mmbench_en_dev":   {"mode": "dfasc", "pool": "v2", "pool_max": 6, "eps": 0.141, "A0": 0.323, "A1": 0.852, "B": 0.189},
    "mmbench_cn_dev":   {"mode": "dfasc", "pool": "v2", "pool_max": 6, "eps": 0.449, "A0": 0.264, "A1": 0.585, "B": 0.442},
    "seedbench":        {"mode": "dfasc", "pool": "v2", "pool_max": 6, "eps": 0.411, "A0": 0.368, "A1": 0.833, "B": 0.190},
    "ai2d":             {"mode": "dfasc", "pool": "nodf4", "pool_max": 4, "eps": 0.184, "A0": 0.050, "A1": 0.085, "B": 0.092},
}


def load_p0_values(bench, pool):
    root = V2_ROOT if pool == "v2" else NODF4_ROOT
    bench_dir = root / bench
    p0s = []
    missing = 0
    for sample_dir in sorted(bench_dir.glob("sample*")):
        conf_path = sample_dir / "original_sample_confidence.json"
        if not conf_path.exists():
            missing += 1
            continue
        try:
            rec = json.loads(conf_path.read_text())
            p0 = rec.get("avg_answer_prob")
            if p0 is not None:
                p0s.append(p0)
            else:
                missing += 1
        except Exception:
            missing += 1
    return p0s, missing


def main():
    results = {}
    for bench, cfg in BENCHMARKS.items():
        if cfg["mode"] == "no_mechanism":
            results[bench] = {"avg_k": cfg["pool_max"], "pool_max": cfg["pool_max"], "ratio": 1.0, "n": None, "mode": "no_mechanism"}
            print(f"{bench:20s} no_mechanism -> ratio=1.000 (Kmax={cfg['pool_max']})")
            continue
        if cfg["mode"] == "degenerate":
            results[bench] = {"avg_k": 0.0, "pool_max": cfg["pool_max"], "ratio": 0.0, "n": None, "mode": "degenerate"}
            print(f"{bench:20s} degenerate   -> ratio=0.000 (pool emptied)")
            continue

        tau = compute_tau(cfg["eps"], cfg["A0"], cfg["A1"], cfg["B"])
        p0s, missing = load_p0_values(bench, cfg["pool"])
        if not p0s:
            print(f"{bench:20s} ERROR: no p0 data found in pool={cfg['pool']}")
            continue
        ks = [min(adaptive_k(p0, tau), cfg["pool_max"]) for p0 in p0s]
        avg_k = sum(ks) / len(ks)
        ratio = avg_k / cfg["pool_max"]
        results[bench] = {"avg_k": avg_k, "pool_max": cfg["pool_max"], "ratio": ratio, "n": len(p0s), "missing": missing, "mode": "dfasc"}
        print(f"{bench:20s} dfasc        -> avg_K={avg_k:.3f}/{cfg['pool_max']} = ratio={ratio:.3f}  (n={len(p0s)}, missing={missing})")

    ratios = [r["ratio"] for r in results.values()]
    blended = sum(ratios) / len(ratios)
    print(f"\n=== BLENDED AVERAGE RATIO across {len(ratios)} benchmarks: {blended:.4f} ===")
    print(f"(previous ceiling-based estimate used for SIMIT-ICL: 0.647)")

    with open(SIMIT_ROOT + "/speed_experiments/replay_adaptive_k_result.json", "w") as f:
        json.dump({"per_benchmark": results, "blended_ratio": blended}, f, indent=2)


if __name__ == "__main__":
    main()
