from global_vars import *

################### DATA HERE #####################
# Full two-stratum outcome decomposition, companion to label_noise_tornado.py
# (which shows only the M=1 / hard-query panel for the main text). Source:
# analyses/label_noise_model_noise/results/summary.json -> joint_decomposition.
METHODS = [
    # (label, is_ours, wrong_M1, correct_M1, wrong_M0, correct_M0)
    ("SIMIT", True,   2.8, 41.7,  4.7, 31.8),
    ("SRLM",  False, 33.3, 27.8, 15.5, 32.6),
    ("TTRL",  False, 44.4, 27.8, 10.1, 36.4),
    ("EMPO",  False, 50.0, 22.2, 12.4, 36.4),
    ("TTRV",  False, 47.2,  8.3,  1.6, 24.8),
]
####################################################

color_harm = color_palette_primary["Color 10"]
color_recover = color_palette_primary["Color 4"]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), sharey=True)

y = np.arange(len(METHODS))[::-1]
labels = [m[0] for m in METHODS]

panels = [
    (axes[0], "Hard queries ($M=1$, model initially wrong)", [m[2] for m in METHODS], [m[3] for m in METHODS]),
    (axes[1], "Easy queries ($M=0$, model initially correct)", [m[4] for m in METHODS], [m[5] for m in METHODS]),
]

for ax, title, wrong, correct in panels:
    wrong = np.array(wrong)
    correct = np.array(correct)
    diffs = correct - wrong

    ax.barh(y, -wrong, height=0.6, color=color_harm, alpha=global_alpha,
            edgecolor="white", linewidth=0.8, label="Reinforces wrong answer")
    ax.barh(y, correct, height=0.6, color=color_recover, alpha=global_alpha,
            edgecolor="white", linewidth=0.8, label="Reinforces correct answer")

    for yi, wv, cv, d in zip(y, wrong, correct, diffs):
        ax.text(-wv - 2.4, yi, f"{wv:.0f}", ha="right", va="center",
                fontsize=global_fontsize_small * 0.65, color="#333333")
        ax.text(cv + 1.2, yi, f"{cv:.0f}", ha="left", va="center",
                fontsize=global_fontsize_small * 0.65, color="#333333")

    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(title, fontsize=global_fontsize_small * 0.85, pad=8)
    ax.set_xlabel("Share of queries (%)", fontsize=global_fontsize_small * 0.85)
    ax.tick_params(axis="x", labelsize=global_fontsize_small * 0.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-62, 55)
    ax.set_ylim(-0.65, len(METHODS) - 0.35)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

axes[0].set_yticks(y)
axes[0].set_yticklabels(labels, fontsize=global_fontsize_small)
for yi, (label, is_ours, *_ ) in zip(y, METHODS):
    if is_ours:
        idx = list(y).index(yi)
        axes[0].get_yticklabels()[idx].set_fontweight("bold")

handles, leg_labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, leg_labels, fontsize=global_fontsize_small, frameon=False,
    loc="upper center", bbox_to_anchor=(0.55, 1.08), ncol=2,
    columnspacing=1.65, handletextpad=0.5, handlelength=1.4,
)

fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.80, wspace=0.12)

fig.savefig(
    f"{save_path}/label_noise_decomposition_full.pdf",
    bbox_inches="tight", pad_inches=0.02,
)

plt.show()
