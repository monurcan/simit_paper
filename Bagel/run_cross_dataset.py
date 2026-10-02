"""Cross-dataset generalization (paper App. J, Fig. 11 / 12).

For each pair X -> Y, the validation-selected SIMIT-FT (Test-Set) checkpoint trained on
benchmark X's synthetic data (run_ft_sweep.py --finalize) is evaluated on benchmark Y's
eval split without further adaptation. Gains are relative to zero-shot BAGEL-7B on Y.
The same/cross-domain averages follow Eq. 51: per source, average over its targets in
that setting, then average over the sources that have targets in both settings.

    python run_cross_dataset.py [--pairs same|cross|all]
"""
import argparse
from pathlib import Path

from simit_sweep import BAGEL_ROOT, BENCHMARKS, DOMAINS, SIMIT_DATA, Job, _slots, load_hparams, read_score

SAME = [
    ("vizwiz_vqa_val", "ok_vqa_val2014"), ("ok_vqa_val2014", "vizwiz_vqa_val"), ("vqav2_val", "ok_vqa_val2014"),
    ("gqa", "ok_vqa_val2014"), ("coco2017_cap_val", "nocaps_val"), ("textvqa_val", "vizwiz_vqa_val"),
    ("mmbench_en_dev", "seedbench"), ("mmbench_cn_dev", "seedbench"), ("mmbench_en_dev", "mmbench_cn_dev"),
    ("mmbench_cn_dev", "mmbench_en_dev"), ("docvqa_val", "infovqa_val"), ("infovqa_val", "chartqa"),
    ("seedbench", "mmbench_cn_dev"), ("ok_vqa_val2014", "gqa"), ("docvqa_val", "chartqa"), ("chartqa", "infovqa_val"),
    ("coco2017_cap_val", "flickr30k_test"), ("textcaps_val", "coco2017_cap_val"), ("vizwiz_vqa_val", "vqav2_val"),
    ("infovqa_val", "docvqa_val"), ("chartqa", "docvqa_val"), ("nocaps_val", "coco2017_cap_val"),
    ("coco2017_cap_val", "textcaps_val"), ("vizwiz_vqa_val", "textvqa_val"), ("seedbench", "mmbench_en_dev"),
]
CROSS = [
    ("infovqa_val", "ok_vqa_val2014"), ("textvqa_val", "seedbench"), ("coco2017_cap_val", "gqa"),
    ("nocaps_val", "seedbench"), ("vqav2_val", "mmbench_en_dev"), ("textvqa_val", "chartqa"),
    ("infovqa_val", "seedbench"), ("mmbench_cn_dev", "nocaps_val"), ("nocaps_val", "chartqa"), ("gqa", "ai2d"),
    ("coco2017_cap_val", "mmbench_en_dev"), ("chartqa", "gqa"), ("mmbench_cn_dev", "gqa"), ("vqav2_val", "ai2d"),
    ("docvqa_val", "ai2d"), ("mmbench_en_dev", "ai2d"), ("seedbench", "docvqa_val"), ("ok_vqa_val2014", "ai2d"),
    ("chartqa", "mmbench_cn_dev"), ("textcaps_val", "mmbench_cn_dev"), ("vqav2_val", "chartqa"),
    ("textcaps_val", "docvqa_val"), ("seedbench", "vqav2_val"), ("vizwiz_vqa_val", "infovqa_val"),
    ("vizwiz_vqa_val", "mmbench_cn_dev"),
]


def domain(bench):
    return next(d for d, benches in DOMAINS.items() if bench in benches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="all", choices=["same", "cross", "all"])
    args = ap.parse_args()
    pairs = {"same": SAME, "cross": CROSS, "all": SAME + CROSS}[args.pairs]
    assert all(domain(x) == domain(y) for x, y in SAME) and all(domain(x) != domain(y) for x, y in CROSS)
    root = Path(SIMIT_DATA) / "cross_dataset"
    slots = _slots()
    jobs = []
    for i, (src, tgt) in enumerate(pairs):
        out_dir = root / f"{src}__{tgt}"
        if read_score(out_dir, BENCHMARKS[tgt]["task"], BENCHMARKS[tgt]["metric"]) is not None:
            continue
        ckpt = Path(SIMIT_DATA) / "checkpoints" / "simit_ft_ts" / load_hparams("simit_ft_ts", src)["trial"]
        cmd = f"cd {BAGEL_ROOT} && ./eval_lora_ckpt.sh {ckpt} {BENCHMARKS[tgt]['task']} {out_dir}"
        job = Job(f"xdata_{src}__{tgt}", cmd, root / "jobs", slots[i % len(slots)], 32)
        if job.proc is not None:
            job.wait(poll=5)
        jobs.append(job)
    for job in jobs:
        job.wait()

    gains = {}
    for src, tgt in pairs:
        cfg = BENCHMARKS[tgt]
        score = read_score(root / f"{src}__{tgt}", cfg["task"], cfg["metric"])
        if score is not None:
            gains[(src, tgt)] = 100.0 * (score - cfg["zero_shot"]) / cfg["zero_shot"]
            print(f"{src:18s} -> {tgt:18s} {gains[(src, tgt)]:+6.1f}%")
    per_source = {}
    for setting, plist in (("same", SAME), ("cross", CROSS)):
        for src in {x for x, _ in plist}:
            g = [gains[p] for p in plist if p[0] == src and p in gains]
            if g:
                per_source.setdefault(src, {})[setting] = sum(g) / len(g)
    shared = [s for s, d in per_source.items() if len(d) == 2]
    for setting in ("same", "cross"):
        if shared:
            print(f"{setting}-domain transfer (Eq. 51, {len(shared)} shared sources): "
                  f"{sum(per_source[s][setting] for s in shared) / len(shared):+.2f}%")


if __name__ == "__main__":
    main()
