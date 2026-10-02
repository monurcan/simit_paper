"""Fixed-sample-count comparisons of self-generated vs. real data (paper Sec. 4.2, App. I, Fig. 4/9/10).

ABA and DF are disabled so every query gets exactly K examples (K in {1, 2, 3, 4}):
  icl        SIMIT-ICL demos from a synthetic pool (run_bagel_icl_eval.sh)
  rices      RICES: K nearest real neighbors from the 2000-sample retrieval pool as demos
  ft_sq      SIMIT-FT (Single-Query) on K synthetic examples (run_bagel_tta_eval.sh)
  ttt_nn     TTT-NN: per-query LoRA on the K nearest real neighbors
The single-query LoRA settings (rank, alpha, lr, steps) of ft_sq and ttt_nn are the
validation-selected ones from run_tta_sweep.py --finalize. Nothing is tuned here, so
results are reported on the 500-sample eval split. K = 0 is the zero-shot baseline.

    python run_scaling_k.py --setting icl --pool nodf4 [--benches ...]
    python run_scaling_k.py --setting ttt_nn
"""
import argparse
import json
from pathlib import Path

from simit_sweep import ALL_BENCHES, BAGEL_ROOT, BENCHMARKS, MODEL, SIMIT_DATA, Job, _slots, load_hparams, read_score, report

INERT_DFASC = "0.14 0.35 0.60 0.20 0.2 0.9"  # ignored when DF and ABA are both disabled


def command(setting, bench, k, pool, out_dir):
    task = BENCHMARKS[bench]["task"]
    if setting == "icl":
        return f"./run_bagel_icl_eval.sh {MODEL} {task} {out_dir} {k} 0 0 {INERT_DFASC} {pool}"
    if setting == "rices":
        return f"./run_bagel_icl_rag_eval.sh {MODEL} {task} {out_dir} {k}"
    p = load_hparams("simit_ft_sq", bench)["params"]
    lora = f"{p['lora_rank']} {p['lora_alpha']} 0.0 {p['adapt_steps']} {p['adapt_lr']}"
    if setting == "ft_sq":
        return f"./run_bagel_tta_eval.sh {MODEL} {task} {out_dir} {k} {pool} 0 0 {INERT_DFASC} {lora}"
    return f"./run_bagel_tta_rag_eval.sh {MODEL} {task} {out_dir} {k} {lora}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=["icl", "rices", "ft_sq", "ttt_nn"])
    ap.add_argument("--pool", default="nodf4", choices=["v2", "nodf4"], help="synthetic pool for icl / ft_sq")
    ap.add_argument("--ks", default="1,2,3,4")
    ap.add_argument("--benches", default=",".join(ALL_BENCHES))
    args = ap.parse_args()
    name = args.setting + (f"_{args.pool}" if args.setting in ("icl", "ft_sq") else "")
    root = Path(SIMIT_DATA) / "scaling_k" / name
    slots = _slots()
    jobs = []
    for i, (bench, k) in enumerate((b, int(k)) for k in args.ks.split(",") for b in args.benches.split(",")):
        out_dir = root / f"{bench}_k{k}"
        if read_score(out_dir, BENCHMARKS[bench]["task"], BENCHMARKS[bench]["metric"]) is None:
            cmd = f"cd {BAGEL_ROOT} && " + command(args.setting, bench, k, args.pool, out_dir)
            job = Job(f"{name}_{bench}_k{k}", cmd, root / "jobs", slots[i % len(slots)], 32)
            if job.proc is not None:
                job.wait(poll=5)
            jobs.append(job)
    for job in jobs:
        job.wait()
    rows = []
    for k in args.ks.split(","):
        print(f"\n== {name}, K={k} ==")
        scores = {b: read_score(root / f"{b}_k{k}", BENCHMARKS[b]["task"], BENCHMARKS[b]["metric"])
                  for b in args.benches.split(",")}
        report(scores)
        rows += [{"bench": b, "k": int(k), "score": s} for b, s in scores.items()]
    (root / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
