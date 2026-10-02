"""Per-request timelines on the demo's own code path, as ZeroGPU runs it (a fresh
forked process per request, weights moved to the GPU each time): when the greedy
answer and each imagined demo arrive, for forced K and a given speed setting.

CUDA_VISIBLE_DEVICES=1 python tools/latency.py lance OUT.json
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from spaces.zero.torch import patching  # noqa: E402
import torch  # noqa: E402

patching.patch()
from demo_models import MODELS, run  # noqa: E402
from general_qa import general_qa  # noqa: E402

ABA = dict(epsilon=0.06, A0=0.5, A1=0.85, B=0.4, t_low=0.0, t_high=1.0)
SETTINGS = {  # speed settings to time (accuracy comes from tune_presets.py)
    "bagel": {"think50": dict(verify=True, verify_think=True, image_steps=50),
              "nothink50": dict(verify=True, verify_think=False, image_steps=50),
              "nothink25": dict(verify=True, verify_think=False, image_steps=25),
              "nocritic25": dict(verify=False, image_steps=25)},
    "lance": {"steps30": dict(image_steps=30), "steps20": dict(image_steps=20)},
    "qwen": {"natural": dict(verify=False, use_skills=False), "skills": dict(verify=False, use_skills=True),
             "critic": dict(verify=True, use_skills=True)},
}
KS = {"bagel": [1, 2, 4], "lance": [1, 2, 4], "qwen": [1, 2]}

key, out = sys.argv[1], Path(sys.argv[2])
n_q = int(sys.argv[3]) if len(sys.argv) > 3 else 6
spec = MODELS[key]
spec.load()
assert not torch.cuda.is_initialized()
queries = general_qa(per_subset=1, offset=440)[:n_q]


def child(conn, q, preset):
    patching.unpatch()
    t0 = time.time()
    rec = {"demos": []}
    spec.preset = lambda budget: dict(preset)   # this process only
    for ev in run(spec, q["image"], q["question"], 1000, True):
        t = time.time() - t0
        if ev[0] == "greedy":
            rec["greedy"] = t
        elif ev[0] == "demo":
            rec["demos"].append(t)
        elif ev[0] == "final":
            rec["total"] = t
    conn.send(rec)


results = []
for sname, setting in SETTINGS[key].items():
    for k in KS[key]:
        preset = dict(ABA, k_max=k, attempts_per_slot=4, repair_retries=1, verify_rounds=2, **setting)
        for qi, q in enumerate(queries):
            a, b = mp.get_context("fork").Pipe()
            p = mp.get_context("fork").Process(target=child, args=(b, q, preset))
            p.start()
            rec = a.recv() if a.poll(900) else {"error": "timeout"}
            p.join(timeout=30)
            rec.update(setting=sname, k=k, query=qi, subset=q["subset"])
            results.append(rec)
            print(json.dumps(rec), flush=True)
            out.write_text(json.dumps(results, indent=1))
