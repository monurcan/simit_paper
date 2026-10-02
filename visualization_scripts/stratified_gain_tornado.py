from global_vars import *

################### DATA HERE #####################
# Macro-averaged (unweighted across all 17 benchmarks) stratified-gain
# transition probabilities: samples are split by whether the zero-shot
# maj@32 baseline got them right, then we track where each method moves
# them. Source: analyses/stratified_gain/stratified_gain_results.json,
# recomputed with CIDEr two-threshold HIGH=0.8/LOW=0.2 (see
# analyses/stratified_gain/threshold_sensitivity.py for the robustness
# sweep across threshold choices -- this pair was one of 4/15 where SIMIT
# beats TTRV on both P(c2w) and P(w2c) simultaneously, not just net gain).
METHODS = [
    # (label, is_ours, p_correct_to_wrong, p_wrong_to_correct)
    ("SIMIT",  True,  2.01, 14.38),
    ("TTRV",   False, 2.03,  9.30),
    ("TTRL",   False, 2.39,  7.68),
    ("EMPO",   False, 2.30,  7.26),
    ("SRLM",   False, 2.94,  6.71),
]
####################################################

color_forget = color_palette_primary["Color 10"]   # P(correct -> wrong), harm
color_recover = color_palette_primary["Color 4"]    # P(wrong -> correct), recovery

fig, ax = plt.subplots(figsize=(5.2 * 0.8, 3.6  * 0.8))

y = np.arange(len(METHODS))[::-1]
labels = [m[0] for m in METHODS]
c2w = np.array([m[2] for m in METHODS])
w2c = np.array([m[3] for m in METHODS])
diffs = w2c - c2w
delta_x = w2c.max() + 4.2

ax.barh(
    y, -c2w, height=0.75, color=color_forget, alpha=global_alpha,
    edgecolor="white", linewidth=0.8, label=r"$P$(correct $\to$ wrong)",
)
ax.barh(
    y, w2c, height=0.75, color=color_recover, alpha=global_alpha,
    edgecolor="white", linewidth=0.8, label=r"$P$(wrong $\to$ correct)",
)

for yi, cv, wv, d in zip(y, c2w, w2c, diffs):
    ax.text(-cv - 0.3, yi, f"{cv:.1f}%", ha="right", va="center",
            fontsize=global_fontsize_small, color="#333333")
    ax.text(wv + 0.3, yi, f"{wv:.1f}%", ha="left", va="center",
            fontsize=global_fontsize_small, color="#333333")
    ax.text(delta_x, yi, f"{d:+.1f}", ha="left", va="center",
            fontsize=global_fontsize_small, fontweight="bold", color="#333333")

ax.text(delta_x, len(METHODS) - 0.35 + 0.35, r"$\Delta$ (pp)",
        ha="left", va="center", fontsize=global_fontsize_small, color="#333333")

ax.axvline(0, color="#333333", linewidth=0.8)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=global_fontsize_small)
for yi, (label, is_ours, _, _) in zip(y, METHODS):
    if is_ours:
        idx = list(y).index(yi)
        ax.get_yticklabels()[idx].set_fontweight("bold")
ax.tick_params(axis="y", length=0)

ax.set_xlabel("Transition probability (%)", fontsize=global_fontsize_small)
ax.tick_params(axis="x", labelsize=global_fontsize_small)

xmax = delta_x - 2.5 # + 1.5
xmin = -(c2w.max() + 3.1)
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.65, len(METHODS) - 0.35)

ax.grid(True, axis="x", alpha=0.25)
ax.set_axisbelow(True)

for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)

handles, leg_labels = ax.get_legend_handles_labels()
ax.legend(
    handles, leg_labels, fontsize=global_fontsize_small, frameon=False,
    loc="lower center", bbox_to_anchor=(0.55, 1.08), ncol=2,
    columnspacing=1.65, handletextpad=0.5, handlelength=1.4,
)

fig.subplots_adjust(left=0.1, right=0.97, bottom=0.14, top=0.88)

fig.savefig(
    f"{save_path}/stratified_gain_tornado.pdf",
    bbox_inches="tight", pad_inches=0.02,
)

plt.show()
