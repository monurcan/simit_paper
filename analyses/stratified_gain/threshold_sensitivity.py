#!/usr/bin/env python3
"""
Sensitivity check: does the CIDEr two-threshold choice (HIGH/LOW) change the
qualitative finding (SIMIT has the highest macro-averaged P(wrong->correct)
across all 17 benchmarks)? Re-runs stratified_gain.main() under a few
different, still-"conservative" threshold pairs and prints the 3x5 macro
table (P(c2w), P(w2c), diff) for each. Only the 5 CIDEr benchmarks' numbers
change across pairs; the 12 binary benchmarks are threshold-independent.

Run with the Bagel venv (needs pycocoevalcap):
    $SIMIT_ROOT/Bagel/.venv/bin/python3 threshold_sensitivity.py
"""
import os as _os
from pathlib import Path as _Path
SIMIT_ROOT = _os.environ.get("SIMIT_ROOT") or str(_Path(__file__).resolve().parents[2])
import sys
sys.path.insert(0, SIMIT_ROOT + "/analyses/stratified_gain")
import stratified_gain as sg

METHODS = ["SIMIT", "TTRL", "EMPO", "TTRV", "SRLM"]

# (label, HIGH, LOW) -- broader grid, still requiring a meaningful gap
# (HIGH - LOW >= 0.3) so the "ambiguous, dropped" middle always means something.
THRESHOLD_SETS = [
    ("HIGH=0.8, LOW=0.1", 0.8, 0.1),
    ("HIGH=0.8, LOW=0.2", 0.8, 0.2),
    ("HIGH=0.8, LOW=0.3", 0.8, 0.3),
    ("HIGH=0.9, LOW=0.2", 0.9, 0.2),
    ("HIGH=1.0, LOW=0.1", 1.0, 0.1),
    ("HIGH=1.0, LOW=0.2", 1.0, 0.2),
    ("HIGH=1.0, LOW=0.3 (current)", 1.0, 0.3),
    ("HIGH=1.0, LOW=0.4", 1.0, 0.4),
    ("HIGH=1.1, LOW=0.3", 1.1, 0.3),
    ("HIGH=1.2, LOW=0.1", 1.2, 0.1),
    ("HIGH=1.2, LOW=0.2", 1.2, 0.2),
    ("HIGH=1.2, LOW=0.3", 1.2, 0.3),
    ("HIGH=1.2, LOW=0.4", 1.2, 0.4),
    ("HIGH=1.4, LOW=0.2", 1.4, 0.2),
    ("HIGH=1.4, LOW=0.4", 1.4, 0.4),
]

all_diff = {m: [] for m in METHODS}
all_w2c = {m: [] for m in METHODS}
all_c2w = {m: [] for m in METHODS}

print(f"{'threshold pair':30s} {'SIMIT_c2w':>10s} {'TTRV_c2w':>9s} {'SIMIT_w2c':>10s} {'TTRV_w2c':>9s}  beats_TTRV_on_both")
n_both = 0
for label, high, low in THRESHOLD_SETS:
    sg.CIDER_HIGH = high
    sg.CIDER_LOW = low
    results = sg.main()

    p_c2w = {m: sum(results[m][b]["p_c2w"] for b in sg.ALL_BENCHES) / len(sg.ALL_BENCHES) for m in METHODS}
    p_w2c = {m: sum(results[m][b]["p_w2c"] for b in sg.ALL_BENCHES) / len(sg.ALL_BENCHES) for m in METHODS}
    diff = {m: p_w2c[m] - p_c2w[m] for m in METHODS}

    for m in METHODS:
        all_diff[m].append(diff[m])
        all_w2c[m].append(p_w2c[m])
        all_c2w[m].append(p_c2w[m])

    lower_c2w = p_c2w["SIMIT"] < p_c2w["TTRV"]  # SIMIT "forgets" less
    higher_w2c = p_w2c["SIMIT"] > p_w2c["TTRV"]  # SIMIT "recovers" more
    beats_both = lower_c2w and higher_w2c
    n_both += beats_both
    print(f"{label:30s} {p_c2w['SIMIT']*100:9.2f}% {p_c2w['TTRV']*100:8.2f}% "
          f"{p_w2c['SIMIT']*100:9.2f}% {p_w2c['TTRV']*100:8.2f}%  {'YES' if beats_both else 'no'}")

print(f"\nSIMIT beats TTRV on BOTH P(c2w) [lower] and P(w2c) [higher] in {n_both}/{len(THRESHOLD_SETS)} threshold pairs.")

print("\n=== Range across all 15 threshold pairs (Difference = P(w2c)-P(c2w), macro-avg over 17 benches) ===")
for m in METHODS:
    lo, hi = min(all_diff[m]), max(all_diff[m])
    print(f"  {m:6s} diff range: [{lo*100:5.2f}%, {hi*100:5.2f}%]   P(w2c) range: [{min(all_w2c[m])*100:5.2f}%, {max(all_w2c[m])*100:5.2f}%]   P(c2w) range: [{min(all_c2w[m])*100:5.2f}%, {max(all_c2w[m])*100:5.2f}%]")

worst_case_margin = min(all_diff["SIMIT"][i] - max(all_diff[m][i] for m in METHODS if m != "SIMIT") for i in range(len(THRESHOLD_SETS)))
print(f"\nWorst-case SIMIT margin over best baseline, across all 15 pairs: {worst_case_margin*100:+.2f} pp")
