"""Step 1: for each of Table 2's 6 non-baseline settings, extract ALL
retained synthesized (image, question, answer) triplets for N_QUERIES_
PER_BENCH real queries per benchmark, using the SAME build_records()
selection logic as the real pipeline (Bagel/prepare_vqa_data_v2_dfadaptive.
py, imported read-only, never reimplemented).

build_records()'s SRC_ROOT is a module-level constant hardcoded to
generated_samples_discard_unverified_v2, so for the other settings we
monkey-patch that constant to the setting's own generated_samples folder
before calling build_records() (build_records() re-reads the module global
on every call, so this is safe and does not require modifying the file).

Per explicit user instructions (2026-09-16):

1. FAIRNESS MASK: a query where the real full pipeline (real DF+ASC config,
   ABA on) retains ZERO candidates is excluded from EVERY setting, not just
   full's -- comparing settings on different query subsets isn't a fair
   comparison, and a query the full pipeline produces no data for shouldn't
   be scored for anyone (never scored as a "0" for full either -- just
   dropped). Since DF's confidence-band filtering is identical between full
   and no_aba (only the ASC/adaptive-k cap differs), this mask is computed
   once from full's own build_records() output and applied uniformly.
   Benchmarks confirmed sourced from the exact same 5 generated_samples_*
   roots and the same real per-benchmark DF+ASC hyperparameters already
   validated against the label-noise audit's simit_selected.jsonl (72/170
   available, byte-for-byte match).

2. NO SUBSAMPLING WITHIN A QUERY: every retained candidate for a given real
   query, under a given setting's own selection logic, is used -- not a
   random subset. "All the labels used in that variant for the same real
   sample" are the actual supervision signal that setting would contribute
   for that query, so all of them go into the data-quality comparison.

3. SPEED: only N_QUERIES_PER_BENCH=2 real queries per benchmark (chosen from
   the full-pipeline-eligible set, seeded) are audited, to keep the agentic
   judging pass small.

NOTE on the 4 fixed-budget settings (disable_df=True, disable_adaptive_k=
True): build_records() unconditionally requires original_sample_confidence.
json to exist before it will include ANY candidate for a query, even though
confidence values are never actually read on this code path (T_low/T_high
and p0/tau are only consulted when disable_df/disable_adaptive_k are False).
3 of these 4 raw-generation folders (no_diversity_prompt, force_natural,
keep_unverified) were never run through confidence scoring at all, so
calling build_records() on them silently returns zero records for every
query. Rather than reimplementing build_records() itself, these 4 settings
use collect_fixed_budget_candidates() below -- the same candidate-collection
loop (same file checks, same clean_question()) minus the confidence gate,
which is exactly what build_records() reduces to when both flags are True
and confidence files happen to be present (verified: identical record counts
on the one folder that has them, generated_samples_discard_unverified_no_
diff_filtering_4_samples).
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import BENCHMARKS, CACHE_DIR, SETTINGS, load_selected_queries, df_asc_kwargs, BAGEL_ROOT

sys.path.insert(0, BAGEL_ROOT)
import prepare_vqa_data_v2_dfadaptive as pvd  # noqa: E402 (read-only import)

BASE_SAMPLING_SEED = 20260916
N_QUERIES_PER_BENCH = 2


def collect_fixed_budget_candidates(root: Path, bench: str):
    """records, stats -- same shape as pvd.build_records(disable_df=True,
    disable_adaptive_k=True), but without requiring confidence JSONs."""
    bench_dir = root / bench
    records = []
    n_samples = 0
    for sample_dir in sorted(
        (p for p in bench_dir.iterdir() if p.is_dir() and p.name.startswith("sample")),
        key=lambda p: int(p.name[len("sample"):]),
    ):
        n_samples += 1
        for i in range(pvd.NUM_GENERATIONS):
            img = sample_dir / f"synthesized_sample_{i}.png"
            prompt_f = sample_dir / f"synthesized_sample_{i}_prompt.txt"
            ans_f = sample_dir / f"synthesized_sample_{i}_answers.txt"
            if not (img.exists() and prompt_f.exists() and ans_f.exists()):
                continue
            question = pvd.clean_question(prompt_f.read_text())
            answer = ans_f.read_text().strip()
            if not question or not answer:
                continue
            records.append({
                "image": str(img.relative_to(root)),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
            })
    return records, {"total_records": len(records), "n_real_samples": n_samples}


def compute_full_eligible_mask(by_bench_sample_ids):
    """{bench: set(sample_id)} -- queries where the real full pipeline
    (real per-benchmark DF+ASC config, ABA on) retains >=1 candidate. Applied
    as a uniform filter to every setting (see module docstring)."""
    full_cfg = SETTINGS["full"]
    pvd.SRC_ROOT = Path(full_cfg["root"])
    eligible = {}
    for bench in BENCHMARKS:
        wanted = set(by_bench_sample_ids[bench])
        kwargs = df_asc_kwargs("full", bench)
        records, _ = pvd.build_records(
            bench,
            epsilon=kwargs["epsilon"], A0=kwargs["A0"], A1=kwargs["A1"], B=kwargs["B"],
            T_low=kwargs["T_low"], T_high=kwargs["T_high"],
            disable_df=kwargs["disable_df"], disable_adaptive_k=False,
        )
        have = {r["image"].split("/")[1] for r in records if r["image"].split("/")[1] in wanted}
        eligible[bench] = have
    return eligible


def pick_audit_queries(eligible_mask):
    """{bench: [sample_id, ...]} -- N_QUERIES_PER_BENCH queries per
    benchmark, seeded, drawn from the full-pipeline-eligible set."""
    picked = {}
    for bench in BENCHMARKS:
        pool = sorted(eligible_mask[bench], key=lambda s: int(s.replace("sample", "")))
        rng = random.Random(f"{BASE_SAMPLING_SEED}:audit_queries:{bench}")
        k = min(len(pool), N_QUERIES_PER_BENCH)
        picked[bench] = sorted(rng.sample(pool, k), key=lambda s: int(s.replace("sample", "")))
        print(f"[audit queries] {bench}: {len(pool)} full-pipeline-eligible (of 10 selected) "
              f"-> picked {picked[bench]}", flush=True)
    return picked


def extract_one_setting(setting_name, audit_queries):
    cfg = SETTINGS[setting_name]
    root = Path(cfg["root"])
    pvd.SRC_ROOT = root
    fixed_budget = cfg["disable_df"] is True and cfg["disable_adaptive_k"] is True

    out_rows = []
    for bench in BENCHMARKS:
        wanted_sample_ids = audit_queries[bench]
        if fixed_budget:
            records, stats = collect_fixed_budget_candidates(root, bench)
        else:
            kwargs = df_asc_kwargs(setting_name, bench)
            records, stats = pvd.build_records(
                bench,
                epsilon=kwargs["epsilon"], A0=kwargs["A0"], A1=kwargs["A1"], B=kwargs["B"],
                T_low=kwargs["T_low"], T_high=kwargs["T_high"],
                disable_df=kwargs["disable_df"], disable_adaptive_k=cfg["disable_adaptive_k"],
            )

        by_sample = {}
        for r in records:
            sample_id = r["image"].split("/")[1]  # "<bench>/sampleN/..."
            if sample_id in wanted_sample_ids:
                by_sample.setdefault(sample_id, []).append(r)

        n_total = sum(len(by_sample.get(sid, [])) for sid in wanted_sample_ids)
        print(f"[{setting_name}] {bench}: {n_total} retained candidates across "
              f"{wanted_sample_ids} (using ALL of them)", flush=True)

        for sample_id in wanted_sample_ids:
            for r in by_sample.get(sample_id, []):
                question = r["conversations"][0]["value"].replace("<image>\n", "", 1)
                answer = r["conversations"][1]["value"]
                out_rows.append({
                    "setting": setting_name, "benchmark": bench, "sample_id": sample_id,
                    "image_rel_path": r["image"], "image_path": str(root / r["image"]),
                    "question": question, "answer": answer,
                })
    return out_rows


def main():
    by_bench_sample_ids = load_selected_queries()
    for bench in BENCHMARKS:
        assert len(by_bench_sample_ids[bench]) == 10, f"{bench}: expected 10 selected queries"

    eligible_mask = compute_full_eligible_mask(by_bench_sample_ids)
    audit_queries = pick_audit_queries(eligible_mask)
    print()

    for setting_name in SETTINGS:
        rows = extract_one_setting(setting_name, audit_queries)
        out_path = os.path.join(CACHE_DIR, f"{setting_name}_selected.jsonl")
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{setting_name}] wrote {len(rows)} triplets (all retained candidates "
              f"for {N_QUERIES_PER_BENCH} queries/benchmark) -> {out_path}\n")


if __name__ == "__main__":
    main()
