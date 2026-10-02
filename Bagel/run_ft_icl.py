"""SIMIT-FT -> ICL (paper Sec. 3.3): the SIMIT-FT (Test-Set) checkpoint answers each query
with the same self-generated demonstrations as SIMIT-ICL, without additional tuning.
Uses the validation-selected configurations of both sweeps
(run_icl_sweep.py --finalize and run_ft_sweep.py --finalize) and reports the eval split.

    python run_ft_icl.py [--benches ai2d,gqa]
"""
import argparse
from pathlib import Path

from run_icl_sweep import icl_command
from simit_sweep import ALL_BENCHES, BENCHMARKS, SIMIT_DATA, Job, _slots, load_hparams, read_score, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default=",".join(ALL_BENCHES))
    benches = ap.parse_args().benches.split(",")
    root = Path(SIMIT_DATA) / "simit_ft_icl"
    slots = _slots()
    jobs = []
    for i, bench in enumerate(benches):
        out_dir = root / bench
        task = BENCHMARKS[bench]["task"]
        if read_score(out_dir, task, BENCHMARKS[bench]["metric"]) is not None:
            continue
        ckpt_root = Path(SIMIT_DATA) / "checkpoints" / "simit_ft_ts" / load_hparams("simit_ft_ts", bench)["trial"]
        steps = sorted(p for p in ckpt_root.glob("[0-9]*") if p.is_dir())
        cmd = icl_command(bench, load_hparams("simit_icl", bench)["params"], task, out_dir, bench,
                          lora_ckpt=str(steps[-1]) if steps else "")
        job = Job(f"ft_icl_{bench}", cmd, root / "jobs", slots[i % len(slots)], 32)
        if job.proc is not None:
            job.wait(poll=5)
        jobs.append(job)
    for job in jobs:
        job.wait()
    report({b: read_score(root / b, BENCHMARKS[b]["task"], BENCHMARKS[b]["metric"]) for b in benches})


if __name__ == "__main__":
    main()
