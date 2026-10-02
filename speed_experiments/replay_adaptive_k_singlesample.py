"""Same replay as replay_adaptive_k.py, but for SIMIT-FT Single-Sample's
winning TTA/DF+ASC configs (tta_dfasc_sweep_results.jsonl), which also used
a ceiling-based (ASC-off exact, ASC-on ceiling) approximation earlier.
asc_enabled=False rows already have an EXACT ratio (k capped by pool_max,
no per-sample adaptivity to replay); only asc_enabled=True rows get replayed
against real per-sample p0 data.
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[1])
SIMIT_DATA = _os.environ.get("SIMIT_DATA") or _os.path.join(SIMIT_ROOT, "workdir")
import json
import math
from pathlib import Path

NUM_GENERATIONS = 6


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

# bench: (k_ceiling, pool, asc_enabled, eps, A0, A1, B) from tta_dfasc_sweep_results.jsonl best trial
BENCHMARKS = {
    "ai2d":             {"k": 6, "pool": "nodf4", "asc": False},
    "chartqa":          {"k": 8, "pool": "v2", "asc": True, "eps": 0.26081193334071273, "A0": 0.3689922785841091, "A1": 0.6153822021619452, "B": 0.23422540741215858},
    "coco2017_cap_val": {"k": 2, "pool": "v2", "asc": True, "eps": 0.391, "A0": 0.357, "A1": 0.488, "B": 0.316},
    "docvqa_val":       {"k": 6, "pool": "nodf4", "asc": True, "eps": 0.3622564777885302, "A0": 0.2583047091044539, "A1": 0.8077364522895752, "B": 0.23845418861111195},
    "flickr30k_test":   {"k": 4, "pool": "v2", "asc": False},
    "gqa":              {"k": 2, "pool": "v2", "asc": True, "eps": 0.392, "A0": 0.556, "A1": 0.847, "B": 0.137},
    "infovqa_val":      {"k": 8, "pool": "nodf4", "asc": False},
    "mmbench_cn_dev":   {"k": 6, "pool": "v2", "asc": True, "eps": 0.449, "A0": 0.264, "A1": 0.585, "B": 0.442},
    "mmbench_en_dev":   {"k": 2, "pool": "v2", "asc": True, "eps": 0.141, "A0": 0.323, "A1": 0.852, "B": 0.189},
    "nocaps_val":       {"k": 6, "pool": "v2", "asc": False},
    "ok_vqa_val2014":   {"k": 2, "pool": "nodf4", "asc": False},
    "refcoco_bbox_val": {"k": 4, "pool": "nodf4", "asc": False},
    "seedbench":        {"k": 2, "pool": "v2", "asc": False},
    "textcaps_val":     {"k": 6, "pool": "nodf4", "asc": False},
    "textvqa_val":      {"k": 2, "pool": "nodf4", "asc": True, "eps": 0.4037735037859928, "A0": 0.27125379812321077, "A1": 0.7888704826628967, "B": 0.29959922714956927},
    "vizwiz_vqa_val":   {"k": 4, "pool": "nodf4", "asc": True, "eps": 0.14, "A0": 0.35, "A1": 0.6, "B": 0.2},
    "vqav2_val":        {"k": 8, "pool": "nodf4", "asc": True, "eps": 0.26, "A0": 0.128, "A1": 0.629, "B": 0.161},
}


def load_p0_values(bench, pool):
    root = V2_ROOT if pool == "v2" else NODF4_ROOT
    bench_dir = root / bench
    p0s = []
    for sample_dir in sorted(bench_dir.glob("sample*")):
        conf_path = sample_dir / "original_sample_confidence.json"
        if conf_path.exists():
            try:
                rec = json.loads(conf_path.read_text())
                if rec.get("avg_answer_prob") is not None:
                    p0s.append(rec["avg_answer_prob"])
            except Exception:
                pass
    return p0s


def main():
    results = {}
    for bench, cfg in BENCHMARKS.items():
        pool_max = 6 if cfg["pool"] == "v2" else 4
        k_capped = min(cfg["k"], pool_max)
        if not cfg["asc"]:
            ratio = k_capped / pool_max
            results[bench] = {"ratio": ratio, "mode": "exact"}
            print(f"{bench:20s} asc=False    -> ratio={ratio:.3f} (exact, k={k_capped}/{pool_max})")
            continue
        tau = compute_tau(cfg["eps"], cfg["A0"], cfg["A1"], cfg["B"])
        p0s = load_p0_values(bench, cfg["pool"])
        if not p0s:
            print(f"{bench:20s} ERROR: no p0 data")
            continue
        ks = [min(adaptive_k(p0, tau), pool_max) for p0 in p0s]
        avg_k = sum(ks) / len(ks)
        ratio = avg_k / pool_max
        results[bench] = {"ratio": ratio, "avg_k": avg_k, "pool_max": pool_max, "n": len(p0s), "mode": "dfasc"}
        print(f"{bench:20s} dfasc        -> avg_K={avg_k:.3f}/{pool_max} = ratio={ratio:.3f} (n={len(p0s)})")

    ratios = [r["ratio"] for r in results.values()]
    blended = sum(ratios) / len(ratios)
    print(f"\n=== BLENDED AVERAGE RATIO across {len(ratios)} benchmarks: {blended:.4f} ===")
    print(f"(previous ceiling-based estimate used for Single-Sample: 0.765)")

    with open(SIMIT_ROOT + "/speed_experiments/replay_adaptive_k_ss_result.json", "w") as f:
        json.dump({"per_benchmark": results, "blended_ratio": blended}, f, indent=2)


if __name__ == "__main__":
    main()
