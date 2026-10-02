from global_vars import *
from matplotlib.ticker import MaxNLocator, FuncFormatter, NullFormatter

################### DATA HERE #####################
data = {
    "Real + FT (C)": {
        "n_aug": [0, 1, 2, 3, 4],
        "scores": [
            { "name": "Visual Question Answering", "unit": "avg %", "values": [64.64, 70.46, 71.12, 71.44, 71.55], "errors": [0.84, 0.72, 0.71, 0.63, 0.70] },
            { "name": "Doc & Infographic Understanding", "unit": "avg %", "values": [78.43, 78.73, 79.37, 79.78, 79.84], "errors": [0.95, 0.71, 0.67, 0.86, 0.66] },
            { "name": "Image Understanding & Captioning", "unit": "avg CIDEr", "values": [1.08, 1.26, 1.32, 1.31, 1.31], "errors": [0.02, 0.02, 0.02, 0.01, 0.02] },
            { "name": "Multi-discipline", "unit": "avg %", "values": [82.16, 83.99, 85.67, 87.01, 87.98], "errors": [1.63, 1.38, 1.21, 1.02, 1.04] },
            { "name": "Math & Science", "unit": "avg %", "values": [86.4, 86.56, 86.5, 86.23, 86.29], "errors": [1.47, 1.22, 1.32, 1.39, 1.35] },
        ]
    },
    "Real + FT (SS)": {
        "n_aug": [0, 1, 2, 3, 4],
        "scores": [
            { "name": "Doc & Infographic Understanding", "unit": "avg %", "values": [78.43333333333334, 75.9258681242833, 76.30357825543146, 76.8875238931271, 77.29448152003413], "errors": [0.95, 0.98, 0.98, 0.99, 0.98] },
            { "name": "Image Understanding & Captioning", "unit": "avg CIDEr", "values": [1.0754, 1.0219165674032282, 1.0368664918260933, 1.0371962164982522, 1.0599628655905058], "errors": [0.02, 0.02, 0.02, 0.02, 0.02] },
            { "name": "Visual Question Answering", "unit": "avg %", "values": [64.64, 56.988000000000014, 59.160000000000004, 61.02, 62.488], "errors": [0.84, 0.88, 0.85, 0.86, 0.85] },
            { "name": "Math & Science", "unit": "avg %", "values": [86.4, 86.6, 86.2, 86.2, 86.0], "errors": [1.47, 1.61, 1.61, 1.62, 1.63] },
            { "name": "Multi-discipline", "unit": "avg %", "values": [82.16333333333334, 86.85050505050505, 87.82222222222221, 87.47474747474747, 88.72727272727273], "errors": [1.63, 1.32, 1.23, 1.28, 1.18] },
        ]   
    },      
    "Real + ICL": {
        "n_aug": [0, 1, 2, 3, 4],
        "scores": [
            { "name": "Doc & Infographic Understanding", "unit": "avg %", "values": [78.43, 78.63, 78.06, 78.32, 77.85], "errors": [0.95, 0.94, 0.94, 0.95, 0.94] },
            { "name": "Image Understanding & Captioning", "unit": "avg CIDEr", "values": [1.075, 1.112, 1.160, 1.188, 1.202], "errors": [0.02, 0.02, 0.02, 0.02, 0.02] },
            { "name": "Visual Question Answering", "unit": "avg %", "values": [64.64, 66.65, 67.71, 68.18, 68.23], "errors": [0.82, 0.82, 0.82, 0.83, 0.82] },
            { "name": "Math & Science", "unit": "avg %", "values": [86.40, 86.00, 84.80, 83.80, 84.60], "errors": [1.62, 1.57, 1.64, 1.68, 1.64] },
            { "name": "Multi-discipline", "unit": "avg %", "values": [82.16, 88.03, 89.30, 89.17, 89.07], "errors": [1.63, 1.23, 1.13, 1.13, 1.15] },
        ]   
    },
    "Synthetic + FT (C)": {
        "n_aug": [0, 1, 2, 3, 4],
        "scores": [
            { "name": "Doc & Infographic Understanding", "unit": "avg %", "values": [78.43333333333334, 78.52765716733138, 78.74, 78.93295957832954, 78.98417169954168], "errors": [0.95, 0.95, 0.98, 0.96, 0.96] },
            { "name": "Image Understanding & Captioning", "unit": "avg CIDEr", "values": [1.0754, 1.09672, 1.10986, 1.1106196209477222, 1.107249031285383], "errors": [0.02, 0.02, 0.02, 0.02, 0.02] },
            { "name": "Visual Question Answering", "unit": "avg %", "values": [64.64, 65.532, 67.12800000000001, 67.984, 68.276], "errors": [0.84, 0.83, 0.82, 0.81, 0.82] },
            { "name": "Math & Science", "unit": "avg %", "values": [86.4, 86.4, 87.2, 87.4, 87.0], "errors": [1.47, 1.40, 1.50, 1.46, 1.52] },
            { "name": "Multi-discipline", "unit": "avg %", "values": [82.16333333333334, 83.28282828282828, 83.23030303030303, 82.4969696969697, 83.14949494949495], "errors": [1.63, 1.60, 1.62, 1.62, 1.60] },
        ]
    },
    "Synthetic + FT (SS)": {
        "n_aug": [0, 1, 2, 3, 4],
        "scores": [
            { "name": "Doc & Infographic Understanding", "unit": "avg %", "values": [78.43333333333334, 77.26226043722403, 77.34124765979958, 76.90651599575409, 76.83507141073625], "errors": [0.95, 0.97, 0.97, 0.97, 0.98] },
            { "name": "Image Understanding & Captioning", "unit": "avg CIDEr", "values": [1.0754, 1.0720856329571748, 1.0784205261642152, 1.0816317820156243, 1.0776934987784024], "errors": [0.02, 0.02, 0.02, 0.02, 0.02] },
            { "name": "Visual Question Answering", "unit": "avg %", "values": [64.64, 64.10400000000001, 64.83200000000002, 64.68800000000002, 65.00400000000002], "errors": [0.84, 0.82, 0.82, 0.82, 0.82] },
            { "name": "Math & Science", "unit": "avg %", "values": [86.4, 86.6, 86.4, 86.2, 86.6], "errors": [1.47, 1.60, 1.61, 1.62, 1.61] },
            { "name": "Multi-discipline", "unit": "avg %", "values": [82.16333333333334, 81.89696969696969, 82.16363636363637, 82.04444444444444, 82.43030303030304], "errors": [1.63, 1.63, 1.65, 1.62, 1.63] },
        ]
    },
    "Synthetic + ICL": {
        "n_aug": [0, 1, 2, 3, 4],
        "scores": [
            { "name": "Doc & Infographic Understanding", "unit": "avg %", "values": [78.43333333333334, 76.81451048652214, 75.20695111920573, 74.33329617796788, 73.62666666666667], "errors": [0.95, 1.00, 1.01, 1.03, 1.02] },
            { "name": "Image Understanding & Captioning", "unit": "avg CIDEr", "values": [1.0754, 1.041881268850641, 1.0173555743970852, 1.0387926786719643, 1.04434], "errors": [0.02, 0.02, 0.02, 0.02, 0.02] },
            { "name": "Visual Question Answering", "unit": "avg %", "values": [64.64, 66.11200000000001, 67.084, 66.39200000000001, 67.02000000000001], "errors": [0.84, 0.85, 0.84, 0.85, 0.83] },
            { "name": "Math & Science", "unit": "avg %", "values": [86.4, 84.2, 84.0, 84.39999999999999, 84.0], "errors": [1.47, 1.71, 1.62, 1.60, 1.65] },
            { "name": "Multi-discipline", "unit": "avg %", "values": [82.16333333333334, 82.4969696969697, 81.74949494949495, 82.54949494949494, 82.21616666666667], "errors": [1.63, 1.63, 1.64, 1.63, 1.64] },
        ]
    },
}
####################################################


def subplots(data, xlabels, savename):
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    # fig.suptitle('Model Performance vs Number of Augmentations', fontsize=14, y=0.98)

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Assign each benchmark a color once so it stays consistent across subplots.
    benchmark_names = [score["name"] for score in next(iter(data.values()))["scores"]]
    palette = list(color_palette_primary.values())[::3]
    benchmark_colors = {
        name: palette[i % len(palette)] for i, name in enumerate(benchmark_names)
    }
    average_color = "#000" #color_palette_primary["Color 11"]

    for i, (dataset, values) in enumerate(data.items()):
        ax = axes[i]
        all_percentages = []
        benchmark_percentages = []
        benchmark_errors = []

        for score in values["scores"]:
            scores = score["values"]
            first_score = scores[0]
            if first_score == 0:
                if any(value != 0 for value in scores):
                    raise ValueError(
                        f'Cannot calculate percentage improvement for "{score["name"]}" '
                        'because its first score is zero.'
                    )
                percentages = [0.0] * len(scores)
            else:
                percentages = [100 * value / first_score for value in scores]

            color = benchmark_colors[score["name"]]
            all_percentages.extend(percentages)
            benchmark_percentages.append(percentages)
            ax.plot(
                values["n_aug"], percentages, 'o-',
                linewidth=1.5, markersize=4,
                color=color, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor=color,
                alpha=global_alpha, label=score["name"]
            )
            if "errors" in score:
                # Convert raw standard deviations to the same normalized scale.
                normalized_errors = 100 * np.asarray(score["errors"]) / first_score
                benchmark_errors.append(normalized_errors)
                percentages_array = np.asarray(percentages)
                lower_bound = percentages_array - normalized_errors
                upper_bound = percentages_array + normalized_errors
                all_percentages.extend(lower_bound)
                all_percentages.extend(upper_bound)
                ax.fill_between(
                    values["n_aug"],
                    lower_bound,
                    upper_bound,
                    color=color, alpha=0.15, linewidth=0, zorder=0
                )

        # Normalize each benchmark first, then average the normalized values.
        # This gives every benchmark equal weight regardless of its raw metric scale.
        average_percentages = np.mean(benchmark_percentages, axis=0)
        all_percentages.extend(average_percentages)
        if len(benchmark_errors) == len(benchmark_percentages):
            # Propagate independent standard deviations through the mean.
            average_errors = (
                np.sqrt(np.sum(np.square(benchmark_errors), axis=0))
                / len(benchmark_errors)
            )
            average_lower_bound = average_percentages - average_errors
            average_upper_bound = average_percentages + average_errors
            all_percentages.extend(average_lower_bound)
            all_percentages.extend(average_upper_bound)
            ax.fill_between(
                values["n_aug"],
                average_lower_bound,
                average_upper_bound,
                color=average_color, alpha=0.15, linewidth=0, zorder=1
            )
        ax.plot(
            values["n_aug"], average_percentages, 'o-',
            linewidth=2.2, markersize=4.5,
            color=average_color, markerfacecolor='white',
            markeredgewidth=1.8, markeredgecolor=average_color,
            alpha=global_alpha, label='Average', zorder=3
        )

        # All curves are normalized to their first point, so the shared baseline is 100%.
        ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.0,
                   alpha=global_grid_alpha)

        # Calculate y-axis limits from normalized percentages.
        min_val = min(all_percentages)
        max_val = max(all_percentages)
        
        data_range = max_val - min_val
        if data_range == 0:
            # Preserve a visible range for curves with no measured change.
            if min_val > 0:
                y_min, y_max = min_val * 0.99, min_val * 1.01
            else:
                y_min, y_max = -0.1, 0.1
        else:
            padding = data_range * 0.1
            # Avoid unused negative space when every benchmark improves.
            y_min = min_val - padding
            y_max = max_val + padding
        
        ax.set_ylim(y_min, y_max)
        # Percentage-of-initial-value data is positive, so a regular log scale
        # gives an undistorted view around the 100% reference line.  Placeholder
        # zero-only series use a linear fallback because zero cannot be logged.
        # ax.set_yscale('log')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=2))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))

        col = i % 3
        ax.set_xlabel(xlabels[col], fontsize=global_fontsize_small)
        # ax.set_ylabel('Percentage of initial value', fontsize=global_fontsize_small)
        ax.set_xticks(values["n_aug"])
        if col == 0:
            # Test-Set column: n_aug counts groups of 500 real samples, not
            # individual samples -- show the real totals so this isn't read
            # as "1-4 samples" the way the other two columns' K is.
            ax.set_xticklabels(['0', '500', '1000', '1500', '2000'])
        if i < 3:
            # The bottom row provides the shared x-axis for the full grid.
            ax.set_xlabel('')
        ax.grid(True, which='both', axis="y", alpha=global_grid_alpha)
        ax.tick_params(labelsize=global_fontsize_small - 3)
        ax.minorticks_on()
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='minor')
        
        
        

    # Use one shared legend; labels and colors are identical in every subplot.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='upper center', bbox_to_anchor=(0.5, 0.15),
        ncol=3, fontsize=global_fontsize_small, frameon=False
    )

    # Use explicit bounds to maximize the plotting area while reserving space
    # for the row labels, column headers, and shared legend.
    fig.subplots_adjust(
        left=0.055, right=0.995, bottom=0.245, top=0.855,
        wspace=0.22, hspace=0.22
    )

    column_titles = [
        'Weight-Space, Test-Set',
        'Weight-Space, Single-Query',
        'Context-Space',
    ]
    for ax, title in zip(axes[:3], column_titles):
        position = ax.get_position()
        fig.text(
            position.x0 + position.width / 2, position.y1 + 0.025, title,
            ha='center', va='bottom', fontsize=global_fontsize,
            fontweight='bold'
        )

    for ax, title in zip((axes[0], axes[3]), ('Real Data', 'Synthetic Data')):
        position = ax.get_position()
        fig.text(
            position.x0 - 0.065, position.y0 + position.height / 2, title,
            ha='center', va='center', rotation=90,
            fontsize=global_fontsize, fontweight='bold'
        )

    fig.savefig(f"{save_path}/{savename}", bbox_inches='tight', pad_inches=0.02)
    plt.show()



subplots(
    data,
    xlabels=[r'Number of Samples, $N_{\mathrm{train}}$', 'Number of Demonstrations, $K$', 'Number of Demonstrations, $K$'],
    savename='scaling_k_subplots.pdf',
)
