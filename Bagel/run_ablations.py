"""SIMIT-FT (Test-Set) ablations and data-scaling runs (paper Tab. 2, Tab. 3, Fig. 9 / App. I).

Every run trains one LoRA adapter per benchmark for 150 steps with that benchmark's
validation-selected SIMIT-FT (Test-Set) LoRA rank / learning rate
(run_ft_sweep.py --finalize) and reports the eval split; nothing is tuned here.

  --variant no_aba          w/o ABA: the tuned DF band applied to all 4 candidates (nodf4 pool)
  --variant no_aba_df       w/o ABA & DF: all candidates of the nodf4 pool (also Tab. 3 row 3)
  --variant no_dp           w/o diversity prompting      (pool no_diversity)
  --variant no_sl           w/o skill library             (pool force_natural)
  --variant no_v            w/o verification              (pool keep_unverified)
  --variant fixed_question  Tab. 3 row 1                  (pool fixed_question)
  --variant fixed_image     Tab. 3 row 2                  (pool fixed_image)
  --variant ntrain --pool nodf4 --ntrain 500,1000,1500,2000   synthetic N_train curve
  --variant real_sft --ntrain 500,1000,1500,2000              SFT on the real retrieval pool
                                                                 (run prepare_vqa_data_real_rag.py and
                                                                 register_real_rag_datasets.py first)
The synthetic pools must exist (similar_samples_llms_eval_lite/run_synthesis.sh <pool> lite),
with confidences for no_aba (compute_confidence.py).
"""
import argparse
import subprocess
from pathlib import Path

from prepare_vqa_data_v2_dfadaptive import CONFIG_DIR, write_yaml_config
from simit_sweep import ALL_BENCHES, BAGEL_ROOT, BENCHMARKS, SIMIT_DATA, Job, _slots, load_hparams, read_score, report

POOL_OF = {"no_aba": "nodf4", "no_aba_df": "nodf4", "no_dp": "no_diversity", "no_sl": "force_natural",
           "no_v": "keep_unverified", "fixed_question": "fixed_question", "fixed_image": "fixed_image"}
STEPS = 150
PY = str(BAGEL_ROOT / ".venv" / "bin" / "python")


def prepare(variant, bench, pool, ntrain):
    """Build the training set; returns {run_name: dataset_config_yaml}."""
    if variant == "no_aba":
        p = load_hparams("simit_ft_ts", bench)["params"]
        tag = f"ablation_no_aba_{bench}"
        subprocess.run([PY, "prepare_vqa_data_v2_dfadaptive.py", "--tag", tag, "--benches", bench, "--pool", "nodf4",
                        "--disable-adaptive-k", "--epsilon", str(p["epsilon"]), "--A0", str(p["A0"]), "--A1", str(p["A1"]),
                        "--B", str(p["B"]), "--T-low", str(p["T_low"]), "--T-high", str(p["T_high"])],
                       cwd=BAGEL_ROOT, check=True)
        return {variant: CONFIG_DIR / f"vqa_{bench}_v2_dfadaptive_{tag}.yaml"}
    if variant in POOL_OF:
        subprocess.run([PY, "prepare_vqa_data_pool.py", "--pool", POOL_OF[variant], "--benches", bench],
                       cwd=BAGEL_ROOT, check=True)
        return {variant: CONFIG_DIR / f"vqa_{bench}_{POOL_OF[variant]}.yaml"}
    if variant == "ntrain":
        subprocess.run([PY, "prepare_vqa_data_pool.py", "--pool", pool, "--benches", bench], cwd=BAGEL_ROOT, check=True)
        dataset = f"vqa_{bench}_{pool}_roundrobin"
    else:  # real_sft
        dataset = f"vqa_{bench}_real_rag"
    configs = {}
    for n in ntrain:
        cfg = CONFIG_DIR / f"{dataset}_n{n}.yaml"
        write_yaml_config(dataset, n, cfg)
        configs[f"{variant}{'_' + pool if variant == 'ntrain' else ''}_n{n}"] = cfg
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(POOL_OF) + ["ntrain", "real_sft"])
    ap.add_argument("--pool", default="nodf4", help="synthetic pool for --variant ntrain")
    ap.add_argument("--ntrain", default="500,1000,1500,2000")
    ap.add_argument("--benches", default=",".join(ALL_BENCHES))
    args = ap.parse_args()
    benches = args.benches.split(",")
    ntrain = [int(n) for n in args.ntrain.split(",")]
    root = Path(SIMIT_DATA) / "ablations"
    slots = _slots()
    runs = []
    for i, bench in enumerate(benches):
        p = load_hparams("simit_ft_ts", bench)["params"]
        for run_name, cfg in prepare(args.variant, bench, args.pool, ntrain).items():
            out_dir = root / run_name / bench
            runs.append((run_name, bench, out_dir))
            if read_score(out_dir, BENCHMARKS[bench]["task"], BENCHMARKS[bench]["metric"]) is not None:
                continue
            ckpt = Path(SIMIT_DATA) / "checkpoints" / "ablations" / run_name / bench
            cmd = (f"cd {BAGEL_ROOT} && ./train_lora.sh {cfg} {ckpt} {STEPS} {p['lr']} {p['lora_rank']} "
                   f"{2 * p['lora_rank']} 0.95 && ./eval_lora_ckpt.sh {ckpt} {BENCHMARKS[bench]['task']} {out_dir}")
            job = Job(f"{run_name}_{bench}", cmd, root / "jobs", slots[i % len(slots)], 170)
            if job.proc is not None:
                job.wait(poll=10)
            else:
                runs[-1] += (job,)
    for run in runs:
        if len(run) == 4:
            run[3].wait()
    for run_name in dict.fromkeys(r[0] for r in runs):
        print(f"\n== {run_name} ==")
        report({b: read_score(o, BENCHMARKS[b]["task"], BENCHMARKS[b]["metric"]) for n, b, o, *_ in runs if n == run_name})


if __name__ == "__main__":
    main()
