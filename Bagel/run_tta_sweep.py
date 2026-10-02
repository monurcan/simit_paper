"""SIMIT-FT (Single-Query) hyperparameter sweep (paper Sec. 3.3 / App. G.1).

For every test query, bagel_tta.py trains a fresh LoRA adapter on that query's
own DF/ABA-selected synthetic examples, answers it with the standard zero-shot
prompt, and resets the adapter. A trial is one run_bagel_tta_eval.sh call on the
validation split. Search space (App. G.1): LoRA rank in {2,...,64}, alpha in
{4,...,128} (independent of the rank), learning rate log-uniform in
[3e-5, 1e-3], steps in [2, 40], plus the DF/ABA selection parameters.

    python run_tta_sweep.py [--benches ai2d,gqa] [--n-trials 20]
    python run_tta_sweep.py --finalize      # best val config -> 500-sample eval split
"""
from simit_sweep import BAGEL_ROOT, MODEL, cli

NAME = "simit_ft_sq"


def sample_params(trial):
    pool = trial.suggest_categorical("pool", ["v2", "nodf4"])
    df_enabled = trial.suggest_categorical("df_enabled", [True, False])
    asc_enabled = trial.suggest_categorical("asc_enabled", [True, False])
    epsilon = trial.suggest_float("epsilon", 0.0, 0.5)
    A0 = trial.suggest_float("A0", 0.05, 0.6)
    A1 = trial.suggest_float("A1", A0, 0.9)
    B = trial.suggest_float("B", 0.0, 0.5)
    T_low = trial.suggest_float("T_low", 0.0, 0.5)
    T_high = trial.suggest_float("T_high", T_low, 1.0)
    num_icl_examples = trial.suggest_categorical("num_icl_examples", [2, 4, 6, 8])
    lora_rank = trial.suggest_categorical("lora_rank", [2, 4, 8, 16, 32, 64])
    lora_alpha = trial.suggest_categorical("lora_alpha", [4, 8, 16, 32, 64, 128])
    adapt_lr = trial.suggest_float("adapt_lr", 3e-5, 1e-3, log=True)
    adapt_steps = trial.suggest_int("adapt_steps", 2, 40)
    return dict(pool=pool, df_enabled=df_enabled, asc_enabled=asc_enabled, epsilon=epsilon, A0=A0, A1=A1, B=B,
                T_low=T_low, T_high=T_high, num_icl_examples=num_icl_examples, lora_rank=lora_rank,
                lora_alpha=lora_alpha, adapt_lr=adapt_lr, adapt_steps=adapt_steps)


def tta_command(bench, p, task, out_dir, tag):
    df, asc = int(bool(p["df_enabled"])), int(bool(p["asc_enabled"]))
    return (f"cd {BAGEL_ROOT} && ./run_bagel_tta_eval.sh {MODEL} {task} {out_dir} {p['num_icl_examples']} {p['pool']} "
            f"{df} {asc} {p['epsilon']} {p['A0']} {p['A1']} {p['B']} {p['T_low']} {p['T_high']} "
            f"{p['lora_rank']} {p['lora_alpha']} 0.0 {p['adapt_steps']} {p['adapt_lr']}")


if __name__ == "__main__":
    cli(NAME, sample_params, tta_command, tta_command, default_trials=20)
