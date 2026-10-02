"""Combine accuracy (tune_presets.py) and latency (latency.py) into presets.json:
for every model and time budget, the largest K_max whose demos arrive in time
for most requests, at the best-quality speed setting that achieves it, with the
ABA/DF thresholds tuned for that K_max.

python tools/build_presets.py RUN_DIR
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))
from demo_models import ANSWER_RESERVE, BUDGETS, MODELS  # noqa: E402

run = Path(sys.argv[1])
# ZeroGPU vs. our H100 (spaces' own duration_factor: 1.5 on the half GPU, none on the full one)
SLOWDOWN = {"large": 1.5, "xlarge": 1.0}
QUALITY = {  # speed settings, best quality first (as timed by latency.py)
    "bagel": ["think50", "nothink50", "nothink25", "nocritic25"],
    "lance": ["steps30", "steps20"],
    "qwen": ["critic", "skills", "natural"],
}
SETTING = {
    "think50": dict(verify=True, verify_think=True, image_steps=50),
    "nothink50": dict(verify=True, verify_think=False, image_steps=50),
    "nothink25": dict(verify=True, verify_think=False, image_steps=25),
    "nocritic25": dict(verify=False, image_steps=25),
    "steps30": dict(image_steps=30), "steps20": dict(image_steps=20),
    "critic": dict(verify=True, use_skills=True), "skills": dict(verify=False, use_skills=True),
    "natural": dict(verify=False, use_skills=False),
}
SHARE = 0.8            # fraction of requests that must get all K demos in time

out_file = HERE / "presets.json"
presets = json.loads(out_file.read_text()) if out_file.exists() else {}   # models without data keep theirs
presets["_note"] = "built by tools/build_presets.py from tune_presets.py + latency.py runs"
report = []
for key in MODELS:
    tune_file, lat_file = run / f"presets_{key}.json", run / f"latency_{key}.json"
    if not (tune_file.exists() and lat_file.exists()):
        print(f"skip {key}: missing {tune_file.name if not tune_file.exists() else lat_file.name}")
        continue
    tuning = json.loads(tune_file.read_text())
    tuned, zero_shot = tuning["robust_per_k_max"], tuning["zero_shot"]
    timelines = json.loads(lat_file.read_text())
    f = SLOWDOWN[MODELS[key].gpu_size]
    scores = {int(k): v["score"] for k, v in tuned.items()}
    presets[key] = {}
    n_val = tuning["n_val"]
    band = {int(k): v["params"]["t_high"] - v["params"]["t_low"] for k, v in tuned.items()}

    def fits(setting, k, deadline):
        """Most requests have k demos before the deadline. Pooled over every run that asked for at least
        k demos (when its k-th demo arrived): 6 requests per cell alone are too noisy."""
        runs = [r for r in timelines if r["setting"] == setting and r["k"] >= k]
        if not runs:
            return False
        return sum(len(r.get("demos", [])) >= k and r["demos"][k - 1] <= deadline for r in runs) >= SHARE * len(runs)

    for budget in BUDGETS:
        deadline = (budget - ANSWER_RESERVE) / f      # the demo's imagination deadline, in H100 seconds
        choice = None
        # The tuned accuracies hold for the setting the tuning cache was made with (the first, best-quality
        # one), so use it whenever any K fits; faster settings only when it cannot fit at all.
        for setting in QUALITY[key]:
            ks = [k for k in sorted(scores) if fits(setting, k, deadline)]
            if not ks:
                continue
            top = max(scores[k] for k in ks)
            near = [k for k in ks if scores[k] >= top - 1.0 / n_val]   # within one validation query
            k = min(near, key=lambda k: (-band[k], k))                  # widest band, then fewest demos
            choice = (k, setting)
            break
        if choice is None:  # nothing fits: the fastest setting, one demo; the deadline decides
            choice = (1, QUALITY[key][-1])
        k, setting = choice
        # more time: more attempts and checking rounds (the paper's 4 / 2 / 5 at the largest budget)
        effort = {20: (2, 0, 1), 40: (2, 1, 1), 60: (3, 1, 2), 90: (4, 2, 3), 180: (4, 2, 5)}[budget]
        params = {p: v for p, v in tuned[str(k)]["params"].items() if p != "k_max"}
        helps = scores[k] > zero_shot + 0.5 / n_val
        if not helps:  # imagination does not beat zero-shot here: ABA never imagines (the UI's checkbox still can)
            params["epsilon"] = 1.0
        presets[key][str(budget)] = dict(params, k_max=k, attempts_per_slot=effort[0], repair_retries=effort[1],
                                         verify_rounds=effort[2], imagine_by_default=helps, **SETTING[setting])
        report.append((key, budget, k, setting, round(scores[k], 3), round(zero_shot, 3), round(band[k], 2),
                       "" if helps else "  -> no gain: zero-shot by default"))

out_file.write_text(json.dumps(presets, indent=2))
for row in report:
    print("%-6s %4ss  K_max=%d  %-11s val %.3f (zero-shot %.3f)  DF band %.2f%s" % row)
