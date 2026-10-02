from global_vars import *
from matplotlib.lines import Line2D

################### DATA HERE #####################
# BAGEL-7B (pretrained zero-shot) and SIMIT-FT (C) (in-domain, continual FT,
# best DF+ASC checkpoint), per evaluation target -- read off Table 6 of the
# paper draft. unit "pct" = percentage-point metric (ANLS/accuracy/gpt-eval
# score already on 0-100), "raw" = unbounded metric averaged as-is (CIDEr).
baseline = {
    "ChartQA":       (80.6, 83.0, "pct"),
    "DocVQA":        (93.2, 93.2, "pct"),
    "InfoVQA":       (61.5, 62.0, "pct"),
    "Flickr30k":     (0.877, 0.896, "raw"),
    "NoCaps":        (0.755, 0.895, "raw"),
    "TextCaps":      (1.586, 1.591, "raw"),
    "COCO":          (1.380, 1.405, "raw"),
    "GQA":           (76.0, 77.0, "pct"),
    "OKVQA":         (37.6, 54.2, "pct"),
    "VizWiz":    (47.0, 53.0, "pct"),
    "VQA-v2":        (78.1, 78.5, "pct"),
    "TextVQA":       (84.5, 84.6, "pct"),
    "AI2D":          (86.4, 87.6, "pct"),
    "MM-B (CN)": (84.8, 86.4, "pct"),
    "MM-B (EN)": (84.1, 85.6, "pct"),
    "SEED":    (77.6, 79.8, "pct"),
}

domain = {
    "ChartQA": "Doc. & Infographic", "DocVQA": "Doc. & Infographic", "InfoVQA": "Doc. & Infographic",
    "Flickr30k": "Image & Captioning", "NoCaps": "Image & Captioning",
    "TextCaps": "Image & Captioning", "COCO": "Image & Captioning",
    "GQA": "VQA", "OKVQA": "VQA", "VizWiz": "VQA", "VQA-v2": "VQA", "TextVQA": "VQA",
    "AI2D": "Math & Science",
    "MM-B (CN)": "Multi-discipline", "MM-B (EN)": "Multi-discipline", "SEED": "Multi-discipline",
}

# (finetuned-on, eval-on, kind, cross-dataset score), same units as `baseline`
# above. Source: 50 LSF eval jobs in Bagel/benchmark_eval_outputs/crossgen*/,
# each running the winning SIMIT-FT (C) checkpoint for `src` on `tgt`'s
# lmms-eval task. This is the matched-design set: the same 14 source
# benchmarks appear on both sides (25 same-domain, 25 cross-domain pairs);
# AI2D never appears as a source since it has no domain sibling to be
# "matched" against, so its 5 old AI2D->X pairs are excluded here.
pairs = [
    ("ChartQA", "DocVQA", "same", 92.72), ("DocVQA", "InfoVQA", "same", 62.37),
    ("InfoVQA", "DocVQA", "same", 92.96), ("DocVQA", "ChartQA", "same", 81.20),
    ("COCO", "NoCaps", "same", 0.7847), ("COCO", "Flickr30k", "same", 0.8814),
    ("COCO", "TextCaps", "same", 1.5743), ("NoCaps", "COCO", "same", 1.3703),
    ("TextCaps", "COCO", "same", 1.3855), ("TextVQA", "VizWiz", "same", 48.66),
    ("OKVQA", "VizWiz", "same", 54.96), ("GQA", "OKVQA", "same", 42.36),
    ("VizWiz", "TextVQA", "same", 83.84), ("OKVQA", "GQA", "same", 76.60),
    ("MM-B (EN)", "MM-B (CN)", "same", 86.40), ("MM-B (CN)", "MM-B (EN)", "same", 85.61),
    ("MM-B (EN)", "SEED", "same", 79.40), ("SEED", "MM-B (EN)", "same", 83.33),
    ("ChartQA", "InfoVQA", "same", 61.86), ("InfoVQA", "ChartQA", "same", 81.40),
    ("MM-B (CN)", "SEED", "same", 79.40), ("SEED", "MM-B (CN)", "same", 85.60),
    ("VizWiz", "OKVQA", "same", 51.12), ("VizWiz", "VQA-v2", "same", 78.12),
    ("VQA-v2", "OKVQA", "same", 43.80),

    ("OKVQA", "AI2D", "cross", 86.40), ("GQA", "AI2D", "cross", 87.40),
    ("VQA-v2", "AI2D", "cross", 86.80), ("DocVQA", "AI2D", "cross", 86.80),
    ("MM-B (EN)", "AI2D", "cross", 86.80), ("ChartQA", "GQA", "cross", 76.60),
    ("ChartQA", "MM-B (CN)", "cross", 84.80), ("InfoVQA", "OKVQA", "cross", 41.20),
    ("InfoVQA", "SEED", "cross", 78.80), ("COCO", "GQA", "cross", 77.60),
    ("COCO", "MM-B (EN)", "cross", 84.85), ("NoCaps", "ChartQA", "cross", 81.60),
    ("NoCaps", "SEED", "cross", 79.20), ("TextCaps", "DocVQA", "cross", 92.94),
    ("TextCaps", "MM-B (CN)", "cross", 84.80), ("TextVQA", "ChartQA", "cross", 82.00),
    ("TextVQA", "SEED", "cross", 80.00), ("VizWiz", "InfoVQA", "cross", 58.54),
    ("VizWiz", "MM-B (CN)", "cross", 76.00), ("MM-B (CN)", "NoCaps", "cross", 0.7646),
    ("MM-B (CN)", "GQA", "cross", 76.40), ("SEED", "DocVQA", "cross", 93.26),
    ("SEED", "VQA-v2", "cross", 77.86), ("VQA-v2", "ChartQA", "cross", 80.60),
    ("VQA-v2", "MM-B (EN)", "cross", 85.61),
]
####################################################
################### STYLING ########################
GROUP_ORDER = ["Same-Domain Transfer",#\n(25 pairs, 14 matched sources)
               "Cross-Domain Transfer"]#\n(25 pairs, same 14 sources)
GROUP_COLOR = {GROUP_ORDER[0]: "Color 5", GROUP_ORDER[1]: "Color 8"}
CEILING_COLOR = "Color 11"  # neutral gray marker for the in-domain ceiling
####################################################


def relative_gain(value, base):
    return 100.0 * (value - base) / base


# Build rows: (group, label, gain_cross, gain_ceiling)
rows = []
for src, tgt, kind, cross_val in pairs:
    base_val, ceil_val, _unit = baseline[tgt]
    gain_cross = relative_gain(cross_val, base_val)
    gain_ceil = relative_gain(ceil_val, base_val)
    group = GROUP_ORDER[0] if kind == "same" else GROUP_ORDER[1]
    label = f"{src}\n→{tgt}"
    rows.append((group, label, gain_cross, gain_ceil))

grouped = {g: [r for r in rows if r[0] == g] for g in GROUP_ORDER}
for g in GROUP_ORDER:
    grouped[g].sort(key=lambda r: r[2], reverse=True)

counts = [len(grouped[g]) for g in GROUP_ORDER]
fig, axes = plt.subplots(
    len(GROUP_ORDER), 1, figsize=(0.8*0.72 * max(counts) + 2.0, 0.8*6.4 * len(GROUP_ORDER)),
)

bar_width = 0.6
legend_handles = []
for gi, (group, ax) in enumerate(zip(GROUP_ORDER, axes)):
    items = grouped[group]
    x_pos = np.arange(len(items))
    labels = [r[1] for r in items]
    gains_cross = [r[2] for r in items]
    gains_ceil = [r[3] for r in items]
    color = color_palette_primary[GROUP_COLOR[group]]

    # Each panel gets its own y-range: the huge AI2D->OKVQA cross-domain
    # outlier would otherwise flatten the near-parity panel's tiny +-1% bars.
    panel_vals = gains_cross + gains_ceil
    y_lo, y_hi = min(0, *panel_vals), max(0, *panel_vals)
    pad = max(0.18 * (y_hi - y_lo), 1.0)

    bars = ax.bar(
        x_pos, gains_cross, bar_width,
        color=color, alpha=global_alpha, edgecolor="white", linewidth=0.6,
        zorder=3,
    )
    ax.scatter(
        x_pos, gains_ceil, marker="_", s=340, linewidths=2.4,
        color=color_palette_primary[CEILING_COLOR], zorder=4,
    )

    for xi, (h, c) in enumerate(zip(gains_cross, gains_ceil)):
        va = "bottom" if h >= 0 else "top"
        off = pad * 0.12 if h >= 0 else -pad * 0.12
        ax.text(
            xi, h + off, f"{h:+.1f}%", ha="center", va=va,
            fontsize=global_fontsize_small*0.9, color=color, fontweight="bold",
        )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.1, alpha=0.6, zorder=1,
               label="BAGEL-7B zero-shot" if gi == 0 else None)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_xlim(-0.7, len(items) - 0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        labels, fontsize=global_fontsize_small,
        rotation=90, ha="right", rotation_mode="anchor",
        va="center",
        # rotation_mode="anchor",# , ha="right"
    )
    ax.grid(True, axis="y", alpha=global_grid_alpha, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=global_fontsize_small)

    title = group.replace("\n", " ")
    ax.set_title(title, fontsize=global_fontsize*1.3, fontweight="bold", pad=10)

for ax in axes:
    ax.set_ylabel("Relative gain over\nBAGEL-7B zero-shot (%)", fontsize=global_fontsize)

# Shared legend: zero-shot reference line, and the in-domain ceiling marker.
legend_elems = [
    Line2D([0], [0], color="gray", linestyle="--", linewidth=1.1, label="BAGEL-7B (Baseline)"),
    Line2D([0], [0], marker="_", color=color_palette_primary[CEILING_COLOR], linestyle="None",
           markersize=14, markeredgewidth=2.4, label="Target-specific reference (SIMIT-FT (TS), source benchmark used for synthesis = target benchmark used for evaluation)"),
]
fig.legend(
    handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.03),
    ncol=2, fontsize=global_fontsize, frameon=False,
    # decrease the legend spacing a bit to avoid overlap with the x-axis labels
    columnspacing=0.8,
    # decrease the width of legend marker
    handlelength=0.7,
)

# fig.suptitle(
#     "Cross-Dataset Generalization of SIMIT-FT (C),\nFinetune-for-source-benchmark-$X$ → Evaluate-on-target-benchmark-$Y$",
#     fontsize=global_fontsize, y=1.005,
# )

fig.subplots_adjust(left=0.06, right=1.01, bottom=0.09, top=0.955, hspace=0.5)
fig.savefig(f"{save_path}/cross_dataset_generalization.pdf", bbox_inches="tight", pad_inches=0.05)
plt.show()
