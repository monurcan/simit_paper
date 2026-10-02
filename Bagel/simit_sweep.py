"""Shared hyperparameter-sweep machinery for SIMIT (paper App. A and G.1).

Tuning protocol: every trial is scored on the benchmark's 50-sample
validation split (lmms-eval task `<task>_val50`). The 500-sample evaluation
split (`<task>`) is touched exactly once per benchmark, by `--finalize`, which
re-runs the best validation trial on it and writes
$SIMIT_DATA/hparams/<sweep>/<bench>.json (params, validation score, eval score).

Each benchmark gets an independent Optuna TPE study stored in
$SIMIT_DATA/sweeps/<sweep>/optuna.db, so a sweep can be stopped and resumed.
Trials are submitted as LSF jobs, at most one per slot in SIMIT_LSF_SLOTS
(comma-separated queue names, default "gpuh100,gpuh100,p1,p1"). With
SIMIT_LAUNCHER=local they run one at a time on this machine instead.

The sweep scripts built on this module are run_icl_sweep.py (SIMIT-ICL),
run_tta_sweep.py (SIMIT-FT, Single-Query) and run_ft_sweep.py (SIMIT-FT, Test-Set).
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import optuna

SIMIT_ROOT = os.environ.get("SIMIT_ROOT") or str(Path(__file__).resolve().parents[1])
SIMIT_DATA = os.environ.get("SIMIT_DATA") or os.path.join(SIMIT_ROOT, "workdir")
BAGEL_ROOT = Path(SIMIT_ROOT) / "Bagel"
MODEL = str(BAGEL_ROOT / "models" / "BAGEL-7B-MoT")

# lmms-eval task, headline metric, and BAGEL-7B's zero-shot score on the 500-sample
# eval split (paper Tab. 8; only used to report relative gains after --finalize).
BENCHMARKS = {
    "flickr30k_test": dict(task="flickr30k_test_lite", metric="flickr_CIDEr,none", zero_shot=0.877),
    "nocaps_val": dict(task="nocaps_val_lite", metric="nocaps_CIDEr,none", zero_shot=0.755),
    "textcaps_val": dict(task="textcaps_val_lite", metric="textcaps_CIDEr,none", zero_shot=1.586),
    "refcoco_bbox_val": dict(task="refcoco_bbox_val_lite", metric="refcoco_CIDEr,none", zero_shot=0.779),
    "coco2017_cap_val": dict(task="coco2017_cap_val_lite", metric="coco_CIDEr,none", zero_shot=1.380),
    "gqa": dict(task="gqa_lite", metric="exact_match,none", zero_shot=0.760),
    "ok_vqa_val2014": dict(task="ok_vqa_val2014_lite", metric="exact_match,none", zero_shot=0.376),
    "vizwiz_vqa_val": dict(task="vizwiz_vqa_val_lite", metric="exact_match,none", zero_shot=0.470),
    "vqav2_val": dict(task="vqav2_val_lite", metric="exact_match,none", zero_shot=0.781),
    "textvqa_val": dict(task="textvqa_val_lite", metric="exact_match,none", zero_shot=0.845),
    "ai2d": dict(task="ai2d_lite", metric="exact_match,flexible-extract", zero_shot=0.864),
    "chartqa": dict(task="chartqa_lite", metric="relaxed_overall,none", zero_shot=0.806),
    "docvqa_val": dict(task="docvqa_val_lite", metric="anls,none", zero_shot=0.932),
    "infovqa_val": dict(task="infovqa_val_lite", metric="anls,none", zero_shot=0.615),
    "mmbench_cn_dev": dict(task="mmbench_cn_dev_lite", metric="gpt_eval_score,none", zero_shot=84.8),
    "mmbench_en_dev": dict(task="mmbench_en_dev_lite", metric="gpt_eval_score,none", zero_shot=84.09),
    "seedbench": dict(task="seedbench_lite", metric="seed_all,none", zero_shot=0.776),
}
ALL_BENCHES = list(BENCHMARKS)
DOMAINS = {
    "Image": ["flickr30k_test", "nocaps_val", "textcaps_val", "refcoco_bbox_val", "coco2017_cap_val"],
    "VQA": ["gqa", "ok_vqa_val2014", "vizwiz_vqa_val", "vqav2_val", "textvqa_val"],
    "Math Sci": ["ai2d"],
    "Doc Info": ["chartqa", "docvqa_val", "infovqa_val"],
    "Multi": ["mmbench_cn_dev", "mmbench_en_dev", "seedbench"],
}

def val_task(bench: str) -> str:
    return BENCHMARKS[bench]["task"] + "_val50"


def read_score(out_dir, task: str, metric: str, any_filter: bool = False):
    """lmms-eval writes <output_path>/<model_name>/<timestamp>_results.json (caption tasks also
    write list-valued submissions/*_results.json, which are skipped). With any_filter=True the
    metric is matched by name under any filter, e.g. "exact_match,maj@32" for "exact_match,none"."""
    for rj in sorted(Path(out_dir).glob("*/*_results.json"), key=lambda p: p.name, reverse=True):
        d = json.loads(rj.read_text())
        r = d.get("results", {}).get(task, {}) if isinstance(d, dict) else {}
        if metric in r:
            return r[metric]
        if any_filter:
            hits = [v for k, v in r.items() if k.split(",")[0] == metric.split(",")[0] and "stderr" not in k]
            if hits:
                return hits[0]
    return None


def load_hparams(sweep: str, bench: str) -> dict:
    """Validation-selected configuration written by `<sweep script> --finalize`."""
    path = Path(SIMIT_DATA) / "hparams" / sweep / f"{bench}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the {sweep} sweep and its --finalize step first")
    return json.loads(path.read_text())


class Job:
    """One shell command, run as an LSF job or (SIMIT_LAUNCHER=local) a local subprocess."""

    def __init__(self, name: str, command: str, log_dir: Path, queue: str, mem_gb: int):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        if os.environ.get("SIMIT_LAUNCHER", "bsub") == "local":
            self.proc = subprocess.Popen(["bash", "-c", command], stdout=open(log_dir / f"{name}.log", "w"),
                                         stderr=subprocess.STDOUT)
            self.job_id = None
            return
        self.proc = None
        script = log_dir / f"{name}.sh"
        script.write_text(
            "#!/bin/bash\n"
            f"#BSUB -q {queue}\n#BSUB -n 4\n#BSUB -gpu \"num=1:mode=exclusive_process\"\n#BSUB -W 24:00\n"
            f"#BSUB -J {name}\n#BSUB -R \"span[hosts=1]\"\n#BSUB -R \"rusage[mem={mem_gb}GB]\"\n"
            f"#BSUB -o {log_dir}/{name}_%J.out\n"
            "set -e\n"
            f"source {SIMIT_ROOT}/simit_env.sh\n"
            f"{command}\n"
        )
        out = subprocess.run(["bsub"], stdin=open(script), capture_output=True, text=True)
        m = re.search(r"Job <(\d+)>", out.stdout)
        if not m:
            raise RuntimeError(f"bsub failed for {name}: {out.stdout} {out.stderr}")
        self.job_id = m.group(1)

    def done(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is not None
        stat = subprocess.run(["bjobs", "-a", "-o", "stat", "-noheader", self.job_id],
                              capture_output=True, text=True).stdout.strip()
        return stat in ("DONE", "EXIT", "")

    def wait(self, poll: int = 60) -> None:
        while not self.done():
            time.sleep(poll)


def _slots():
    if os.environ.get("SIMIT_LAUNCHER", "bsub") == "local":
        return ["local"]
    return os.environ.get("SIMIT_LSF_SLOTS", "gpuh100,gpuh100,p1,p1").split(",")


def run_sweep(name, benches, n_trials, sample_params, make_command, mem_gb=32, poll=60, score_fn=None):
    """Optuna ask/tell loop. make_command(bench, params, task, out_dir, tag) -> shell command
    that evaluates one configuration on `task` and writes lmms-eval results to out_dir.
    score_fn(bench, out_dir) overrides how a finished trial's validation score is read."""
    root = Path(SIMIT_DATA) / "sweeps" / name
    root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{root}/optuna.db"
    studies = {b: optuna.create_study(storage=storage, study_name=f"{name}_{b}", direction="maximize",
                                      load_if_exists=True) for b in benches}
    finished = lambda b: sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL)
                             for t in studies[b].trials)
    active = {}  # slot index -> (bench, trial, out_dir, job)
    while True:
        for si, queue in enumerate(_slots()):
            if si in active:
                continue
            running = {b: sum(a[0] == b for a in active.values()) for b in benches}
            todo = [b for b in benches if finished(b) + running[b] < n_trials]
            if not todo:
                break
            bench = min(todo, key=lambda b: finished(b) + running[b])
            trial = studies[bench].ask()
            params = sample_params(trial)
            tag = f"{bench}_t{trial.number:04d}"
            out_dir = root / "trials" / tag
            cmd = make_command(bench, params, val_task(bench), str(out_dir), tag)
            print(f"[{time.strftime('%H:%M:%S')}] launch {tag}: {params}", flush=True)
            try:
                job = Job(f"{name}_{tag}", cmd, root / "jobs", queue, mem_gb)
            except Exception as e:
                print(f"  launch failed: {e}", flush=True)
                studies[bench].tell(trial, state=optuna.trial.TrialState.FAIL)
                continue
            active[si] = (bench, trial, out_dir, job)

        for si, (bench, trial, out_dir, job) in list(active.items()):
            if not job.done():
                continue
            score = (score_fn(bench, out_dir) if score_fn else
                     read_score(out_dir, val_task(bench), BENCHMARKS[bench]["metric"]))
            if score is None:
                studies[bench].tell(trial, state=optuna.trial.TrialState.FAIL)
                print(f"[{time.strftime('%H:%M:%S')}] {bench} trial {trial.number} FAILED", flush=True)
            else:
                studies[bench].tell(trial, score)
                print(f"[{time.strftime('%H:%M:%S')}] {bench} trial {trial.number} val={score:.4f}", flush=True)
                with open(root / "trials.jsonl", "a") as f:
                    f.write(json.dumps({"bench": bench, "trial": trial.number, "params": trial.params,
                                        "val_score": score}) + "\n")
            del active[si]

        if not active and all(finished(b) >= n_trials for b in benches):
            break
        time.sleep(poll if os.environ.get("SIMIT_LAUNCHER", "bsub") != "local" else 5)
    print("sweep finished; run again with --finalize to evaluate the best validation trials on the eval split")


def finalize(name, benches, make_final_command, mem_gb=32, any_filter=False):
    """Evaluate each benchmark's best validation trial once on the 500-sample eval split.
    make_final_command(bench, params, task, out_dir, trial_tag) -> shell command."""
    root = Path(SIMIT_DATA) / "sweeps" / name
    storage = f"sqlite:///{root}/optuna.db"
    hp_dir = Path(SIMIT_DATA) / "hparams" / name
    hp_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for i, bench in enumerate(benches):
        best = optuna.load_study(storage=storage, study_name=f"{name}_{bench}").best_trial
        trial_tag = f"{bench}_t{best.number:04d}"
        out_dir = root / "final" / bench
        cmd = make_final_command(bench, best.params, BENCHMARKS[bench]["task"], str(out_dir), trial_tag)
        print(f"{bench}: best validation trial {best.number} (val={best.value:.4f}) -> eval split", flush=True)
        job = Job(f"{name}_final_{bench}", cmd, root / "jobs", _slots()[i % len(_slots())], mem_gb)
        if job.proc is not None:
            job.wait(poll=5)
        pending.append((bench, best, trial_tag, out_dir, job))

    scores = {}
    for bench, best, trial_tag, out_dir, job in pending:
        job.wait()
        cfg = BENCHMARKS[bench]
        score = read_score(out_dir, cfg["task"], cfg["metric"], any_filter=any_filter)
        record = {"bench": bench, "trial": trial_tag, "params": best.params, "val_score": best.value,
                  "eval_score": score, "zero_shot_eval_score": cfg["zero_shot"]}
        (hp_dir / f"{bench}.json").write_text(json.dumps(record, indent=2))
        scores[bench] = score
        if score is None:
            print(f"{bench:18s} eval run FAILED (see {root}/jobs)")
    report(scores)


def report(scores: dict) -> None:
    """Per-benchmark scores, domain means (CIDEr native, others x100) and the mean
    benchmark-level relative gain over zero-shot ("Av. Gain", paper Sec. 4)."""
    gains = []
    for bench, score in scores.items():
        if score is None:
            continue
        z = BENCHMARKS[bench]["zero_shot"]
        gains.append(100.0 * (score - z) / z)
        print(f"  {bench:18s} {score:8.4f}  (zero-shot {z:.4f}, {gains[-1]:+.2f}%)")
    # CIDEr stays in native units and MMBench's gpt_eval_score is already 0-100.
    scale = lambda b, v: v if any(k in BENCHMARKS[b]["metric"] for k in ("CIDEr", "gpt_eval_score")) else 100 * v
    for dom, benches in DOMAINS.items():
        vals = [scale(b, scores[b]) for b in benches if scores.get(b) is not None]
        if len(vals) == len(benches):
            print(f"  domain {dom:9s} {sum(vals) / len(vals):8.3f}")
    if gains:
        print(f"  Av. Gain over {len(gains)} benchmarks: {sum(gains) / len(gains):+.2f}%")


def cli(name, sample_params, make_command, make_final_command, default_trials, mem_gb=32, score_fn=None,
        any_filter=False):
    ap = argparse.ArgumentParser(description=f"{name} sweep (tunes on <task>_val50; --finalize scores the eval split)")
    ap.add_argument("--benches", default=",".join(ALL_BENCHES), help="comma-separated benchmark names")
    ap.add_argument("--n-trials", type=int, default=default_trials, help="Optuna trials per benchmark")
    ap.add_argument("--finalize", action="store_true", help="evaluate each benchmark's best validation trial on the eval split")
    args = ap.parse_args()
    benches = args.benches.split(",")
    unknown = sorted(set(benches) - set(BENCHMARKS))
    if unknown:
        ap.error(f"unknown benchmarks {unknown}; choose from {ALL_BENCHES}")
    if args.finalize:
        finalize(name, benches, make_final_command, any_filter=any_filter)
    else:
        run_sweep(name, benches, args.n_trials, sample_params, make_command, mem_gb, score_fn=score_fn)
