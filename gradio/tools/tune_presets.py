"""Tune ABA + DF for the demo on the mixed general-QA set, separately for each
K_max: one synthesis cache (K_max = 4 candidates per query), one study over
K_max in {1, 2, 3, 4}, and the best trial for each K_max is kept.

python tools/tune_presets.py bagel OUT_DIR
python tools/tune_presets.py lance OUT_DIR
python tools/tune_presets.py qwen OUT_DIR      # vLLM engine for tuning speed (same FP8 weights as the demo)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from general_qa import general_qa  # noqa: E402

from simit import SIMIT, SIMITConfig  # noqa: E402

MODELS = {
    "bagel": dict(model="ByteDance-Seed/BAGEL-7B-MoT"),
    "lance": dict(model="bytedance-research/Lance"),
    "qwen": dict(model="Qwen/Qwen3.8-27B-FP8", image_generator="black-forest-labs/FLUX.2-klein-4B", engine="vllm"),
}

name, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
val = general_qa()
sim = SIMIT.from_pretrained(**MODELS[name], config=SIMITConfig(k_max=4))
t0 = time.time()
res = sim.tune(val, n_trials=800, k_max_choices=[1, 2, 3, 4], cache_dir=out / f"cache_{name}", max_new_tokens=16,
               apply=False)
print(f"tuned in {time.time() - t0:.0f}s: {res}", flush=True)

best = {}
for params, value, cost in res.history:
    k = params.get("k_max")
    if value >= 0 and (k not in best or value > best[k]["score"]):
        best[k] = {"score": value, "mean_candidates": cost, "params": params}
# Regularized pick per K_max: within one validation query's worth of the best score, the widest
# difficulty-filter band (small validation sets overfit narrow bands; a wide band also rejects fewer
# demos, which matters under the demo's time limit).
tol = 1.0 / len(val)
robust = {}
for k, b in best.items():
    near = [(p, v, c) for p, v, c in res.history if p.get("k_max") == k and v >= b["score"] - tol]
    p, v, c = max(near, key=lambda t: (t[0]["t_high"] - t[0]["t_low"], t[1]))
    robust[k] = {"score": v, "mean_candidates": c, "params": p}
report = {"model": name, "n_val": len(val), "zero_shot": res.zero_shot_score, "overall_best": res.best_params,
          "best_per_k_max": {str(k): best[k] for k in sorted(best)},
          "robust_per_k_max": {str(k): robust[k] for k in sorted(robust)},
          "history": [{"params": p, "score": v, "mean_candidates": c} for p, v, c in res.history]}
(out / f"presets_{name}.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
sim.close()
