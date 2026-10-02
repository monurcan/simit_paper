"""Hyperparameter search for the label-free RL baselines (paper App. G.3, Tab. 7).

Every method trains on the benchmark's 500 unlabeled eval queries (test-set access)
and is tuned on the 50-sample validation split, like SIMIT: a trial trains one
configuration for 10 epochs and its score is the best validation score over the
epoch-{3,4,5,6,10} checkpoints (run_lfrl_trial.sh), so the number of epochs is
selected on validation too. --finalize evaluates the best (trial, epoch) once on the
eval split and writes $SIMIT_DATA/hparams/lfrl_<method>/<bench>.json.

    python sweep_lfrl.py --method ttrv [--benches gqa,ai2d] [--n-trials 12]
    python sweep_lfrl.py --method ttrv --finalize
"""
import argparse
import json
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Bagel"))
from simit_sweep import ALL_BENCHES, BENCHMARKS, SIMIT_DATA, SIMIT_ROOT, Job, _slots, read_score, report, run_sweep  # noqa: E402

LFRL_ROOT = Path(SIMIT_ROOT) / "LFRL_Baselines"
EPOCHS = [3, 4, 5, 6, 10]


def sample_params(method):
    def sample(trial):
        p = dict(temperature=trial.suggest_categorical("temperature", [0.6, 1.0, 1.3, 1.5]),
                 lr=trial.suggest_categorical("lr", [5e-6, 1e-5, 2e-5]),
                 lora_rank=trial.suggest_categorical("lora_rank", [16, 32, 64]), method_param=0)
        if method == "empo":
            p["method_param"] = trial.suggest_categorical("jaccard_threshold", [0.4, 0.6, 0.8])
        elif method == "ttrv":
            p["method_param"] = trial.suggest_categorical("entropy_weight", [0.0, 0.25, 0.5, 0.75, 1.5])
        return p
    return sample


def eval_task(method, bench, split):
    """SRLM decodes greedily; TTRL/EMPO/TTRV use maj@32 (*_sc32 tasks)."""
    task = BENCHMARKS[bench]["task"] + ("" if method == "srlm" else "_sc32")
    return task + "_val50" if split == "val50" else task


def epoch_scores(method, bench, out_dir):
    task = eval_task(method, bench, "val50")
    return {e: read_score(Path(out_dir) / f"epoch{e}", task, BENCHMARKS[bench]["metric"], any_filter=True) for e in EPOCHS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["ttrl", "empo", "ttrv", "srlm"])
    ap.add_argument("--benches", default=",".join(ALL_BENCHES))
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    benches, name = args.benches.split(","), f"lfrl_{args.method}"

    def trial_command(bench, p, task, out_dir, tag):
        return (f"cd {LFRL_ROOT} && ./run_lfrl_trial.sh {args.method} {bench} {name}_{tag} {p['temperature']} "
                f"{p['lr']} {p['lora_rank']} {p['method_param']} {out_dir}")

    def best_epoch_score(bench, out_dir):
        scores = [s for s in epoch_scores(args.method, bench, out_dir).values() if s is not None]
        return max(scores) if scores else None

    if not args.finalize:
        run_sweep(name, benches, args.n_trials, sample_params(args.method), trial_command, mem_gb=64,
                  score_fn=best_epoch_score)
        return

    root = Path(SIMIT_DATA) / "sweeps" / name
    hp_dir = Path(SIMIT_DATA) / "hparams" / name
    hp_dir.mkdir(parents=True, exist_ok=True)
    final = []
    for i, bench in enumerate(benches):
        best = optuna.load_study(storage=f"sqlite:///{root}/optuna.db", study_name=f"{name}_{bench}").best_trial
        tag = f"{bench}_t{best.number:04d}"
        scores = epoch_scores(args.method, bench, root / "trials" / tag)
        epoch = max((e for e in EPOCHS if scores[e] is not None), key=lambda e: scores[e])
        ckpt_dir = Path(SIMIT_DATA) / "checkpoints" / "lfrl" / ("ttrl" if args.method == "empo" else args.method) / f"{name}_{tag}"
        ckpt = sorted(ckpt_dir.glob("step_*"))[epoch - 1]  # one checkpoint per epoch, zero-padded step names
        out_dir = root / "final" / bench
        cmd = (f"cd {LFRL_ROOT} && ./eval_lfrl_ckpt.sh {args.method} {ckpt} {best.params['lora_rank']} "
               f"{BENCHMARKS[bench]['task']} lite {out_dir}")
        print(f"{bench}: best trial {best.number}, epoch {epoch} (val={scores[epoch]:.4f}) -> eval split", flush=True)
        job = Job(f"{name}_final_{bench}", cmd, root / "jobs", _slots()[i % len(_slots())], 64)
        if job.proc is not None:
            job.wait(poll=5)
        final.append((bench, best, tag, epoch, out_dir, job))

    results = {}
    for bench, best, tag, epoch, out_dir, job in final:
        job.wait()
        score = read_score(out_dir, eval_task(args.method, bench, "lite"), BENCHMARKS[bench]["metric"], any_filter=True)
        results[bench] = score
        (hp_dir / f"{bench}.json").write_text(json.dumps(
            {"bench": bench, "trial": tag, "epoch": epoch, "params": best.params, "val_score": best.value,
             "eval_score": score, "zero_shot_eval_score": BENCHMARKS[bench]["zero_shot"]}, indent=2))
    report(results)


if __name__ == "__main__":
    main()
