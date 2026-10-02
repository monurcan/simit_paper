"""SIMIT-ICL hyperparameter sweep (paper Sec. 3.3 / App. G.1).

Searches, per benchmark, the adaptive-budget parameters (epsilon, A0, A1, B, and
Kmax through the choice of synthetic pool: v2 = 6 candidates, nodf4 = 4), the
difficulty-filter band [T_low, T_high], and the maximum number of in-context
demonstrations. DF and ABA select demos at eval time from the cached candidate
pool (bagel_icl.py::DFASCSyntheticSampleRetriever), so a trial is a single
lmms-eval run of run_bagel_icl_eval.sh on the benchmark's validation split.

    python run_icl_sweep.py [--benches ai2d,gqa] [--n-trials 50]
    python run_icl_sweep.py --finalize      # best val config -> 500-sample eval split
"""
from simit_sweep import BAGEL_ROOT, MODEL, cli

NAME = "simit_icl"


def sample_params(trial):
    pool = trial.suggest_categorical("pool", ["v2", "nodf4"])
    epsilon = trial.suggest_float("epsilon", 0.0, 0.5)
    A0 = trial.suggest_float("A0", 0.05, 0.6)
    A1 = trial.suggest_float("A1", A0, 0.9)
    B = trial.suggest_float("B", 0.0, 0.5)
    T_low = trial.suggest_float("T_low", 0.0, 0.5)
    T_high = trial.suggest_float("T_high", T_low, 1.0)
    num_icl = trial.suggest_categorical("num_icl_examples", [2, 4, 6, 8])
    return dict(pool=pool, epsilon=epsilon, A0=A0, A1=A1, B=B, T_low=T_low, T_high=T_high, num_icl_examples=num_icl)


def icl_command(bench, p, task, out_dir, tag, lora_ckpt=""):
    return (f"cd {BAGEL_ROOT} && ./run_bagel_icl_eval.sh {MODEL} {task} {out_dir} {p['num_icl_examples']} 1 1 "
            f"{p['epsilon']} {p['A0']} {p['A1']} {p['B']} {p['T_low']} {p['T_high']} {p['pool']} \"{lora_ckpt}\"")


if __name__ == "__main__":
    cli(NAME, sample_params, icl_command, icl_command, default_trials=50)
