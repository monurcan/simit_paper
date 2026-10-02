from global_vars import *

################### DATA HERE #####################
# Label-noise / model-noise alignment audit (analyses/label_noise_model_noise/).
# 170 original queries (10/benchmark x 17 benchmarks), one positive-advantage
# (LFRL) or retained (SIMIT) supervision sample audited per query by a blind
# external evaluator. M=1 = base model already wrong on that query (n=36,
# 21.8% of queries with a resolved M). Bars show, of the queries where the
# base model already fails, what fraction of each method's supervision
# reinforces the wrong answer (harm) vs. reinforces the correct answer
# (recovery); the gap between them is not 100% because a method can also
# abstain (no positive-advantage / no retained example at all) -- see
# Appendix F.3 for the full three-way decomposition including abstention.
# Source: results/summary.json -> joint_decomposition[method].given_M1.
METHODS = [
    # (label, is_ours, p_reinforce_wrong, p_reinforce_correct)
    ("SIMIT", True,   2.8, 41.7),
    ("SRLM",  False, 33.3, 27.8),
    ("TTRL",  False, 44.4, 27.8),
    ("EMPO",  False, 50.0, 22.2),
    ("TTRV",  False, 47.2,  8.3),
]
####################################################

color_harm = color_palette_primary["Color 10"]      # reinforces the wrong answer
color_recover = color_palette_primary["Color 4"]    # reinforces the correct answer

fig, ax = plt.subplots(figsize=(5.6, 3.3))

y = np.arange(len(METHODS))[::-1]
labels = [m[0] for m in METHODS]
wrong = np.array([m[2] for m in METHODS])
correct = np.array([m[3] for m in METHODS])
diffs = correct - wrong

ax.barh(
    y, -wrong, height=0.6, color=color_harm, alpha=global_alpha,
    edgecolor="white", linewidth=0.8, label="reinforces wrong answer",
)
ax.barh(
    y, correct, height=0.6, color=color_recover, alpha=global_alpha,
    edgecolor="white", linewidth=0.8, label="reinforces correct answer",
)

for yi, wv, cv, d in zip(y, wrong, correct, diffs):
    ax.text(-wv - 2.2, yi, f"{wv:.0f}%", ha="right", va="center",
            fontsize=global_fontsize_small * 0.7, color="#333333")
    ax.text(cv + 0.8, yi, f"{cv:.0f}%", ha="left", va="center",
            fontsize=global_fontsize_small * 0.7, color="#333333")
    ax.text(cv + 8.5, yi, f"net={d:+.0f}pp", ha="left", va="center",
            fontsize=global_fontsize_small * 0.7, fontweight="bold", color="#333333")

ax.axvline(0, color="#333333", linewidth=0.8)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=global_fontsize_small)
for yi, (label, is_ours, _, _) in zip(y, METHODS):
    if is_ours:
        idx = list(y).index(yi)
        ax.get_yticklabels()[idx].set_fontweight("bold")
ax.tick_params(axis="y", length=0)

ax.set_xlabel("Share of model-wrong queries (%)", fontsize=global_fontsize_small)
ax.tick_params(axis="x", labelsize=global_fontsize_small)

xmax = (correct + 8.5).max() + 6.5
xmin = -(wrong.max() + 10.0)
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.65, len(METHODS) - 0.35)

ax.grid(True, axis="x", alpha=0.25)
ax.set_axisbelow(True)

for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)

handles, leg_labels = ax.get_legend_handles_labels()
ax.legend(
    handles, leg_labels, fontsize=global_fontsize_small, frameon=False,
    loc="lower center", bbox_to_anchor=(0.42, 0.98), ncol=1,
    columnspacing=1.65, handletextpad=0.5, handlelength=1.4,
)

fig.subplots_adjust(left=0.15, right=0.97, bottom=0.14, top=0.80)

fig.savefig(
    f"{save_path}/label_noise_tornado.pdf",
    bbox_inches="tight", pad_inches=0.02,
)

plt.show()
