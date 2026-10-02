"""Step 9: join L and M annotations, compute the four statistics per method
with bootstrap CIs, per-benchmark breakdown, and the common-eligible-subset
comparison.

  c           = P(L=1)
  c_wrong     = P(L=1 | M=1)
  c_correct   = P(L=1 | M=0)
  Delta       = c_wrong - c_correct   (also cross-checked via Cov(L,M)/Var(M))

L = 1 - verdict for an L (supervision) packet (verdict=1 means the audited
triplet was judged correct); M = 1 - verdict for the baseline (M) packet,
i.e. M=1 means the frozen model's own greedy answer was judged incorrect.

Bootstrap resamples ORIGINAL QUERIES (benchmark, sample_id) with
replacement, keeping every method's record for that query together in each
resample -- not independent per-method resampling.
"""
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import BENCHMARKS, PROVENANCE_DIR, RESULTS_DIR

METHODS = ["ttrl", "empo", "ttrv", "srlm", "simit"]
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = "20260913:bootstrap"


def load_manifest_and_verdicts():
    manifest = [json.loads(l) for l in open(os.path.join(PROVENANCE_DIR, "audit_manifest.jsonl"))]
    verdicts = {}
    verdicts_path = os.path.join(PROVENANCE_DIR, "eval_verdicts.jsonl")
    for l in open(verdicts_path):
        row = json.loads(l)
        verdicts[row["audit_id"]] = row
    return manifest, verdicts


def build_query_table(manifest, verdicts):
    """Returns {(benchmark, sample_id): {"M": 0/1/None, "L": {method: 0/1/None}}}"""
    table = {}
    unresolved = []  # audit_ids with no verdict, or flagged ambiguous -- reported as exclusions
    for row in manifest:
        key = (row["benchmark"], row["sample_id"])
        table.setdefault(key, {"M": None, "L": {m: None for m in METHODS}})
        v = verdicts.get(row["audit_id"])
        if v is None:
            unresolved.append({**row, "reason": "no_verdict_returned"})
            continue
        if v.get("flagged_ambiguous"):
            unresolved.append({**row, "reason": "flagged_ambiguous"})
            continue
        correct = bool(v["verdict"])
        error_flag = 0 if correct else 1
        if row["kind"] == "M":
            table[key]["M"] = error_flag
        else:
            table[key]["L"][row["method"]] = error_flag
    return table, unresolved


def stats_for(records, method):
    """records: list of {"M":..., "L":{method:...}}. Returns dict with c,
    c_wrong, c_correct, delta (each None if undefined), plus n's."""
    pairs = [(r["L"][method], r["M"]) for r in records if r["L"][method] is not None and r["M"] is not None]
    l_only = [r["L"][method] for r in records if r["L"][method] is not None]

    n = len(l_only)
    c = statistics.mean(l_only) if n > 0 else None

    wrong = [l for l, m in pairs if m == 1]
    correct = [l for l, m in pairs if m == 0]
    c_wrong = statistics.mean(wrong) if wrong else None
    c_correct = statistics.mean(correct) if correct else None
    delta = (c_wrong - c_correct) if (c_wrong is not None and c_correct is not None) else None

    delta_cov = None
    if len(pairs) >= 2:
        ls = [p[0] for p in pairs]
        ms = [p[1] for p in pairs]
        mean_l, mean_m = statistics.mean(ls), statistics.mean(ms)
        cov = sum((l - mean_l) * (m - mean_m) for l, m in zip(ls, ms)) / len(pairs)
        var_m = sum((m - mean_m) ** 2 for m in ms) / len(pairs)
        delta_cov = cov / var_m if var_m > 1e-12 else None

    return {
        "c": c, "c_wrong": c_wrong, "c_correct": c_correct, "delta": delta,
        "delta_via_cov_var": delta_cov,
        "n_L_available": n, "n_paired_with_M": len(pairs),
        "n_wrong_available": len(wrong), "n_correct_available": len(correct),
    }


def bootstrap_ci(records, method, seed, n_boot=N_BOOTSTRAP):
    rng = random.Random(seed)
    n = len(records)
    if n == 0:
        return None
    samples = {"c": [], "c_wrong": [], "c_correct": [], "delta": []}
    for _ in range(n_boot):
        resample = [records[rng.randrange(n)] for _ in range(n)]
        s = stats_for(resample, method)
        for k in samples:
            if s[k] is not None:
                samples[k].append(s[k])
    out = {}
    for k, vals in samples.items():
        if len(vals) < max(50, 0.5 * n_boot):  # too many undefined replicates to trust a CI
            out[k] = None
            continue
        vals_sorted = sorted(vals)
        lo = vals_sorted[int(0.025 * len(vals_sorted))]
        hi = vals_sorted[min(len(vals_sorted) - 1, int(0.975 * len(vals_sorted)))]
        out[k] = {"lo": lo, "hi": hi, "n_replicates_defined": len(vals), "n_replicates_total": n_boot}
    return out


def joint_decomposition_for(records, method):
    """Full 2x3 outcome table per M stratum: every query with a known M
    either abstains (no positive-advantage/retained supervision), reinforces
    a correct answer, or reinforces a wrong one. The three sum to 1 within
    each stratum -- this is what Delta silently discards (it conditions away
    abstention entirely), and it's what actually determines a method's
    expected marginal contribution to accuracy, not just noise-model
    correlation."""
    with_m1 = [r for r in records if r["M"] == 1]
    with_m0 = [r for r in records if r["M"] == 0]

    def rates(sub):
        n = len(sub)
        if n == 0:
            return None
        avail = [r for r in sub if r["L"][method] is not None]
        reinforce_wrong = sum(1 for r in avail if r["L"][method] == 1)
        reinforce_correct = sum(1 for r in avail if r["L"][method] == 0)
        abstain = n - len(avail)
        return {
            "n": n,
            "p_abstain": abstain / n,
            "p_reinforce_correct": reinforce_correct / n,
            "p_reinforce_wrong": reinforce_wrong / n,
        }

    m1 = rates(with_m1)
    m0 = rates(with_m0)
    net1 = (m1["p_reinforce_correct"] - m1["p_reinforce_wrong"]) if m1 else None
    net0 = (m0["p_reinforce_correct"] - m0["p_reinforce_wrong"]) if m0 else None
    return {"given_M1": m1, "given_M0": m0, "net_M1": net1, "net_M0": net0}


def bootstrap_joint_ci(records, method, prevalence_m1, prevalence_m0, seed, n_boot=N_BOOTSTRAP):
    rng = random.Random(seed)
    n = len(records)
    if n == 0:
        return None
    keys = ["net_M1", "net_M0", "net_weighted"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        resample = [records[rng.randrange(n)] for _ in range(n)]
        d = joint_decomposition_for(resample, method)
        if d["net_M1"] is None or d["net_M0"] is None:
            continue
        samples["net_M1"].append(d["net_M1"])
        samples["net_M0"].append(d["net_M0"])
        samples["net_weighted"].append(prevalence_m1 * d["net_M1"] + prevalence_m0 * d["net_M0"])
    out = {}
    for k, vals in samples.items():
        if len(vals) < max(50, 0.5 * n_boot):
            out[k] = None
            continue
        vals_sorted = sorted(vals)
        lo = vals_sorted[int(0.025 * len(vals_sorted))]
        hi = vals_sorted[min(len(vals_sorted) - 1, int(0.975 * len(vals_sorted)))]
        out[k] = {"lo": lo, "hi": hi, "n_replicates_defined": len(vals), "n_replicates_total": n_boot}
    return out


def availability_breakdown(records, method):
    with_m = [r for r in records if r["M"] is not None]
    avail = [r for r in with_m if r["L"][method] is not None]
    avail_given_m1 = [r for r in with_m if r["M"] == 1 and r["L"][method] is not None]
    avail_given_m0 = [r for r in with_m if r["M"] == 0 and r["L"][method] is not None]
    n_m1 = sum(1 for r in with_m if r["M"] == 1)
    n_m0 = sum(1 for r in with_m if r["M"] == 0)
    return {
        "n_total": len(with_m),
        "n_available": len(avail),
        "availability_rate": len(avail) / len(with_m) if with_m else None,
        "availability_rate_given_M1": len(avail_given_m1) / n_m1 if n_m1 else None,
        "availability_rate_given_M0": len(avail_given_m0) / n_m0 if n_m0 else None,
        "n_M1": n_m1, "n_M0": n_m0,
    }


def main():
    manifest, verdicts = load_manifest_and_verdicts()
    table, unresolved = build_query_table(manifest, verdicts)
    all_records = list(table.values())
    by_benchmark_records = {}
    for (bench, sid), rec in table.items():
        by_benchmark_records.setdefault(bench, []).append(rec)

    result = {"pooled": {}, "per_benchmark": {}, "availability": {}, "common_eligible_subset": {},
              "joint_decomposition": {}, "class_prevalence": {},
              "unresolved_exclusions": unresolved, "n_queries_total": len(all_records)}

    n_m1_total = sum(1 for r in all_records if r["M"] == 1)
    n_m0_total = sum(1 for r in all_records if r["M"] == 0)
    n_m_known = n_m1_total + n_m0_total
    prevalence_m1 = n_m1_total / n_m_known if n_m_known else None
    prevalence_m0 = n_m0_total / n_m_known if n_m_known else None
    result["class_prevalence"] = {"n_M1": n_m1_total, "n_M0": n_m0_total,
                                   "prevalence_M1": prevalence_m1, "prevalence_M0": prevalence_m0}

    for method in METHODS:
        result["pooled"][method] = stats_for(all_records, method)
        result["pooled"][method]["bootstrap_ci_95"] = bootstrap_ci(
            all_records, method, seed=f"{BOOTSTRAP_SEED}:pooled:{method}"
        )
        result["availability"][method] = availability_breakdown(all_records, method)

        jd = joint_decomposition_for(all_records, method)
        jd["net_weighted"] = (
            prevalence_m1 * jd["net_M1"] + prevalence_m0 * jd["net_M0"]
            if jd["net_M1"] is not None and jd["net_M0"] is not None else None
        )
        jd["bootstrap_ci_95"] = bootstrap_joint_ci(
            all_records, method, prevalence_m1, prevalence_m0, seed=f"{BOOTSTRAP_SEED}:joint:{method}"
        )
        result["joint_decomposition"][method] = jd

        # macro: mean of per-benchmark c/c_wrong/c_correct (equal benchmark weight)
        per_bench_stats = {}
        for bench in BENCHMARKS:
            recs = by_benchmark_records.get(bench, [])
            s = stats_for(recs, method)
            s["bootstrap_ci_95"] = bootstrap_ci(recs, method, seed=f"{BOOTSTRAP_SEED}:{bench}:{method}", n_boot=2000)
            per_bench_stats[bench] = s
        result["per_benchmark"][method] = per_bench_stats
        defined_c = [s["c"] for s in per_bench_stats.values() if s["c"] is not None]
        defined_delta = [s["delta"] for s in per_bench_stats.values() if s["delta"] is not None]
        result["pooled"][method]["macro_c_mean_over_benchmarks"] = (
            statistics.mean(defined_c) if defined_c else None
        )
        result["pooled"][method]["macro_delta_mean_over_benchmarks"] = (
            statistics.mean(defined_delta) if defined_delta else None
        )
        result["pooled"][method]["macro_n_benchmarks_with_defined_c"] = len(defined_c)
        result["pooled"][method]["macro_n_benchmarks_with_defined_delta"] = len(defined_delta)

    # common eligible subset: queries where every method has an available L
    common_keys = [k for k, rec in table.items() if all(rec["L"][m] is not None for m in METHODS)]
    common_records = [table[k] for k in common_keys]
    result["common_eligible_subset"]["n_queries"] = len(common_records)
    result["common_eligible_subset"]["per_method"] = {
        method: {
            **stats_for(common_records, method),
            "bootstrap_ci_95": bootstrap_ci(common_records, method, seed=f"{BOOTSTRAP_SEED}:common:{method}"),
        }
        for method in METHODS
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Persist every (L, M) realization per query x method, flat, for reuse
    # without re-deriving from the manifest+verdicts join.
    lm_path = os.path.join(RESULTS_DIR, "lm_realizations.jsonl")
    with open(lm_path, "w") as f:
        for (bench, sample_id), rec in sorted(table.items()):
            for method in METHODS:
                f.write(json.dumps({
                    "benchmark": bench, "sample_id": sample_id, "method": method,
                    "M": rec["M"], "L": rec["L"][method],
                }, ensure_ascii=False) + "\n")
    print(f"Wrote {lm_path} ({len(table) * len(METHODS)} rows)")

    out_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=False)

    print(f"Wrote {out_path}")
    print(f"\nPooled results (n_queries={len(all_records)}):")
    for method in METHODS:
        s = result["pooled"][method]
        av = result["availability"][method]
        print(
            f"  {method:6s}: c={s['c']}, c_wrong={s['c_wrong']}, c_correct={s['c_correct']}, "
            f"delta={s['delta']} (n_L={s['n_L_available']}, n_paired={s['n_paired_with_M']}, "
            f"availability={av['availability_rate']})"
        )
    print(f"\nUnresolved/excluded audit items: {len(unresolved)}")
    print(f"Common eligible subset size: {result['common_eligible_subset']['n_queries']}")

    print(f"\nJoint decomposition (abstain / reinforce-correct / reinforce-wrong), "
          f"class prevalence M=1:{prevalence_m1:.1%} M=0:{prevalence_m0:.1%}:")
    for method in METHODS:
        jd = result["joint_decomposition"][method]
        m1, m0 = jd["given_M1"], jd["given_M0"]
        print(f"  {method:6s}: M=1[abstain={m1['p_abstain']:.1%} ok={m1['p_reinforce_correct']:.1%} "
              f"wrong={m1['p_reinforce_wrong']:.1%} net={jd['net_M1']:+.3f}]  "
              f"M=0[abstain={m0['p_abstain']:.1%} ok={m0['p_reinforce_correct']:.1%} "
              f"wrong={m0['p_reinforce_wrong']:.1%} net={jd['net_M0']:+.3f}]  "
              f"net_weighted={jd['net_weighted']:+.3f}")


if __name__ == "__main__":
    main()
