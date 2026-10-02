from global_vars import *
from matplotlib.transforms import offset_copy

################### DATA HERE #####################
# Label-noise / model-noise alignment audit, main-text figure. Restricted to
# queries the base model answers incorrectly (M=1, n=36 of 170). Each bar
# sums to 100%: correct supervision / no positive supervision (abstain) /
# incorrect supervision. Delta (pp) is c_wrong - c_correct, computed over
# both M=1 and M=0 (Appendix Table 6), shown here as a per-method summary
# column. Source: results/summary.json -> joint_decomposition[method].given_M1
# and pooled[method].delta. Unrounded values.
METHODS = [
    # (label, is_ours, p_correct, p_abstain, p_wrong, delta_pp)
    ("SIMIT", True,   41.6667, 55.5556,  2.7778,  -6.5),
    ("SRLM",  False,  27.7778, 38.8889, 33.3333, +22.3),
    ("TTRL",  False,  27.7778, 27.7778, 44.4444, +39.9),
    ("EMPO",  False,  22.2222, 27.7778, 50.0000, +43.8),
    ("TTRV",  False,   8.3333, 44.4444, 47.2222, +79.1),
]
####################################################

color_correct = color_palette_primary["Color 4"]   # correct supervision
color_wrong = color_palette_primary["Color 10"]     # incorrect supervision
color_abstain = "#C9C7BE"                            # no positive supervision

fig, ax = plt.subplots(figsize=(5.6, 3.4))

y = np.arange(len(METHODS))[::-1]
labels = [m[0] for m in METHODS]

# Keep the summary column outside the percentage axis and inside export bounds.
delta_artists = []
delta_transform = ax.get_yaxis_transform()

for yi, (label, is_ours, p_c, p_a, p_w, delta) in zip(y, METHODS):
    left = 0.0
    for val, color in [(p_c, color_correct), (p_a, color_abstain), (p_w, color_wrong)]:
        ax.barh(yi, val, left=left, height=0.62, color=color, alpha=global_alpha,
                edgecolor="white", linewidth=0.8)
        if val >= 6:
            ax.text(left + val / 2, yi, f"{val:.0f}", ha="center", va="center",
                    fontsize=global_fontsize_small * 0.65, color="#222222")
        left += val
    delta_artists.append(ax.text(
        1.03, yi, f"{delta:+.1f}", transform=delta_transform, clip_on=False,
        ha="left", va="center", fontsize=global_fontsize_small * 0.75,
        fontweight="bold", color="#333333",
    ))

delta_artists.append(ax.text(
    1.03, len(METHODS) - 0.35 + 0.55, r"$\delta$ (pp)",
    transform=delta_transform, clip_on=False, ha="left", va="center",
    fontsize=global_fontsize_small * 0.75, color="#333333",
))

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=global_fontsize_small)
for yi, (label, is_ours, *_ ) in zip(y, METHODS):
    if is_ours:
        idx = list(y).index(yi)
        ax.get_yticklabels()[idx].set_fontweight("bold")
ax.tick_params(axis="y", length=0)

ax.set_xlabel("Supervision quality on initially\nmisanswered queries (%)", fontsize=global_fontsize_small)
ax.set_xticks([0, 25, 50, 75, 100])
ax.tick_params(axis="x", labelsize=global_fontsize_small)
ax.set_xlim(0, 100)
ax.set_ylim(-0.65, len(METHODS) - 0.35 + 0.55)

ax.grid(True, axis="x", alpha=0.25)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=color_correct, alpha=global_alpha),
    plt.Rectangle((0, 0), 1, 1, color=color_abstain, alpha=global_alpha),
    plt.Rectangle((0, 0), 1, 1, color=color_wrong, alpha=global_alpha),
]
# Separate rows so the single entry is centered under the first two.
legend_fontsize = global_fontsize_small * 0.8
legend_options = dict(
    fontsize=legend_fontsize, frameon=False,
    loc="lower center", bbox_to_anchor=(0.42, 0.99),
    handletextpad=0.5, handlelength=1.4, borderpad=0, borderaxespad=0,
)
top_legend = ax.legend(
    [legend_handles[0], legend_handles[2]], ["Correct supervision", "Incorrect supervision"],
    ncol=2,
    bbox_transform=offset_copy(
        ax.transAxes, fig=fig, y=legend_fontsize * 1.3, units="points",
    ),
    **legend_options,
)
ax.add_artist(top_legend)
bottom_legend = ax.legend(
    [legend_handles[1]], ["No positive supervision"],
    ncol=1, **legend_options,
)

fig.subplots_adjust(left=0.16, right=0.8, bottom=0.16, top=0.72)

fig.savefig(
    f"{save_path}/label_noise_stacked.pdf", bbox_inches="tight",
    bbox_extra_artists=(top_legend, bottom_legend, *delta_artists), pad_inches=0.02,
)
plt.show()
