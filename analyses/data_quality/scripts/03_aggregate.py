"""Step 3: merge audit_manifest.jsonl with the collected eval_verdicts.jsonl
and report ONE number per setting: the average binary rubric score (fraction
of non-ambiguous audited triplets judged correct). No c/c_wrong/c_correct
decomposition -- this is a plain data-quality average, not the label-noise
audit.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import BENCHMARKS, PROVENANCE_DIR, RESULTS_DIR, SETTINGS


def main():
    manifest = [json.loads(l) for l in open(os.path.join(PROVENANCE_DIR, "audit_manifest.jsonl"))]
    verdicts = {}
    for l in open(os.path.join(PROVENANCE_DIR, "eval_verdicts.jsonl")):
        row = json.loads(l)
        verdicts[row["audit_id"]] = row

    # per_setting[setting] = list of 0/1 (excludes missing verdicts / ambiguous)
    per_setting = {s: [] for s in SETTINGS}
    per_setting_per_bench = {s: {b: [] for b in BENCHMARKS} for s in SETTINGS}
    excluded = []

    for row in manifest:
        v = verdicts.get(row["audit_id"])
        if v is None:
            excluded.append({**row, "reason": "no_verdict_returned"})
            continue
        if v.get("flagged_ambiguous"):
            excluded.append({**row, "reason": "flagged_ambiguous"})
            continue
        score = 1 if v["verdict"] else 0
        per_setting[row["setting"]].append(score)
        per_setting_per_bench[row["setting"]][row["benchmark"]].append(score)

    summary = {}
    for setting_name, scores in per_setting.items():
        summary[setting_name] = {
            "table_row": SETTINGS[setting_name]["table_row"],
            "n_scored": len(scores),
            "n_expected": 170,
            "data_quality_avg": sum(scores) / len(scores) if scores else None,
            "per_benchmark_avg": {
                b: (sum(bs) / len(bs) if bs else None)
                for b, bs in per_setting_per_bench[setting_name].items()
            },
        }

    summary["_excluded"] = excluded

    out_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'setting':<22} {'table row':<28} {'n':>5} {'data quality':>13}")
    for setting_name in SETTINGS:
        s = summary[setting_name]
        avg = f"{100 * s['data_quality_avg']:.1f}%" if s["data_quality_avg"] is not None else "N/A"
        print(f"{setting_name:<22} {s['table_row']:<28} {s['n_scored']:>5} {avg:>13}")
    if excluded:
        print(f"\n{len(excluded)} triplets excluded (no verdict / flagged ambiguous) -- see summary.json['_excluded']")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
