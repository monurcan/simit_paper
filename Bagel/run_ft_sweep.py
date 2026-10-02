"""SIMIT-FT (Test-Set) hyperparameter sweep (paper Sec. 3.3 / App. G.1).

A trial pools the synthetic samples of a benchmark's 500 unlabeled eval queries,
selects them with DF+ABA, trains one LoRA adapter and scores it on the 50-sample
validation split (run_simit_ft.sh). Eval-split labels are never used: the
training data are built from unlabeled queries only, and --finalize evaluates the
already-trained checkpoint of the best validation trial on the eval split.
Search space: the DF/ABA parameters, Kmax via the pool (v2 = 6, nodf4 = 4),
LoRA rank r in {8, 64} with alpha = 2r, and learning rate in {7e-5, 1e-4, 2e-4}.

    python run_ft_sweep.py [--benches ai2d,gqa] [--n-trials 12]
    python run_ft_sweep.py --finalize
"""
from simit_sweep import BAGEL_ROOT, SIMIT_DATA, cli

NAME = "simit_ft_ts"


def sample_params(trial):
    pool = trial.suggest_categorical("pool", ["v2", "nodf4"])
    epsilon = trial.suggest_float("epsilon", 0.0, 0.5)
    A0 = trial.suggest_float("A0", 0.05, 0.6)
    A1 = trial.suggest_float("A1", A0, 0.9)
    B = trial.suggest_float("B", 0.0, 0.5)
    T_low = trial.suggest_float("T_low", 0.0, 0.5)
    T_high = trial.suggest_float("T_high", T_low, 1.0)
    lora_rank = trial.suggest_categorical("lora_rank", [8, 64])
    lr = trial.suggest_categorical("lr", [7e-5, 1e-4, 2e-4])
    return dict(pool=pool, epsilon=epsilon, A0=A0, A1=A1, B=B, T_low=T_low, T_high=T_high, lora_rank=lora_rank, lr=lr)


def trial_command(bench, p, task, out_dir, tag):
    return (f"cd {BAGEL_ROOT} && ./run_simit_ft.sh {bench} {tag} {p['pool']} {p['epsilon']} {p['A0']} {p['A1']} "
            f"{p['B']} {p['T_low']} {p['T_high']} {p['lora_rank']} {p['lr']} {task} {out_dir}")


def final_command(bench, p, task, out_dir, trial_tag):
    return f"cd {BAGEL_ROOT} && ./eval_lora_ckpt.sh {SIMIT_DATA}/checkpoints/{NAME}/{trial_tag} {task} {out_dir}"


if __name__ == "__main__":
    # LoRA training with FSDP + CPU offload needs a large host-RAM reservation.
    cli(NAME, sample_params, trial_command, final_command, default_trials=12, mem_gb=170)
