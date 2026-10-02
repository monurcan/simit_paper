from global_vars import *
from scipy import stats

################### DATA HERE #####################
# Per-benchmark % improvement over the BAGEL-7B zero-shot baseline, matched at
# the same demo count / sample budget for the real-data and synthetic-data arm
# of each mechanism (k=4 demos for ICL and single-sample TTA; N=2000 finetuning
# samples, N=1000 for chartqa, for test-set finetuning). Source values are the
# no-mechanism (DF/ASC-disabled) scaling studies in Bagel/artifacts/lora_vs_fullft.html
# ("SG-ICL ... best dataset per benchmark", "TT-SI (finetuning) ... best dataset
# per benchmark", "SIMIT-FT (SS) ... synthetic vs. real-data retrieval") and the
# real-data counterparts ("Appendix: Real ICL on the capped-2000 retrieval pool",
# "Real data: accuracy vs. N_totalsamples", and the TTT-NN column of the same
# SIMIT-FT (SS) table). Synthetic values take the better of the two unfiltered
# generation pools (discard_unverified_v2 / no_diff_filtering_4_samples) at the
# matched budget, mirroring how the real-vs-synthetic comparison is drawn in the
# rest of this scaling study.
data = {
    "Context-Space\n(RICES vs. SIMIT-ICL, $K=4$)": {
        "real":     [45.43, 15.06, -2.37, -3.28, -2.18, 1.24, -0.66, -3.47, 14.62, 57.58, -10.90, 32.99, -1.04, 14.41, 8.49, 1.80, -2.08],
        "synthetic":[28.94, 16.09, -4.21, -2.15, -1.96, -11.41, -1.47, -6.26, -0.89, 53.91, -25.36, -13.86, -3.21, 0.90, -0.94, 0.26, -2.78],
    },
    "Weight-Space, Single-Query\n(TTT-NN vs. SIMIT-FT (SQ), $K=4$)": {
        "real":     [24.47, 38.26, -5.21, 0.73, -4.30, -2.37, 2.32, 0.83, -16.20, 0.17, 0.19, -13.42, -10.12, -0.46, -16.07, 10.81, 10.38],
        "synthetic":[15.74, 4.43, -5.21, -0.00, -1.62, -1.41, 2.06, 2.53, -1.59, -0.97, -0.05, -1.05, -3.20, 0.23, -0.13, 0.00, -0.94],
    },
    "Weight-Space, Test-Set\n(SFT vs. SIMIT-FT (TS), " + r"$N_{\mathrm{train}}=2000$)": {
        "real":     [71.07, 73.01, 61.06, 10.81, 8.49, 28.43, 4.71, 3.61, 1.80, 4.26, 0.45, 0.46, 0.80, 0.26, 0.15, -0.22, -2.39],
        "synthetic":[4.71, 14.52, 31.17, 0.90, 0.00, 13.62, 2.98, 2.84, 1.65, 1.76, -0.37, 0.69, -0.65, 0.53, 0.13, -1.59, -0.52],
    },
}
####################################################

marker_colors = [
    color_palette_primary["Color 1"],
    color_palette_primary["Color 7"],
    color_palette_primary["Color 4"],
]


def scatter_real_vs_synthetic(data, savename):
    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    all_real, all_synthetic = [], []
    for (name, values), color in zip(data.items(), marker_colors):
        real = np.asarray(values["real"])
        synthetic = np.asarray(values["synthetic"])
        all_real.extend(real)
        all_synthetic.extend(synthetic)

        r, p = stats.pearsonr(real, synthetic)
        ax.scatter(
            real, synthetic, s=42, color=color, alpha=global_alpha,
            edgecolor="white", linewidth=0.6,
            label=f"{name}  (r={r:.2f})",
        )

    all_real = np.asarray(all_real)
    all_synthetic = np.asarray(all_synthetic)
    r, p = stats.pearsonr(all_real, all_synthetic)
    slope, intercept, *_ = stats.linregress(all_real, all_synthetic)
    xs = np.linspace(all_real.min() - 5, all_real.max() + 5, 100)
    ax.plot(
        xs, slope * xs + intercept, "--", color=color_palette_primary["Color 11"],
        linewidth=1.6, alpha=0.9, zorder=0,
        label=f"Pooled Linear Fit, (r={r:.2f}, p={p:.1e})",
    )

    ax.axhline(0, color="gray", linewidth=0.8, alpha=global_grid_alpha)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=global_grid_alpha)
    ax.set_xlabel("Real-data gain over baseline (%)", fontsize=global_fontsize)
    ax.set_ylabel("Synthetic-data gain over baseline (%)", fontsize=global_fontsize)
    ax.grid(True, alpha=global_grid_alpha)
    ax.tick_params(labelsize=global_fontsize - 2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=global_fontsize - 4, loc="upper left", frameon=False)
    
    

    fig.tight_layout()
    fig.savefig(f"{save_path}/{savename}", bbox_inches="tight", pad_inches=0.02)
    plt.show()


scatter_real_vs_synthetic(data, savename="real_vs_synthetic_scatter.pdf")
