from global_vars import *
from matplotlib.offsetbox import AnnotationBbox, TextArea, HPacker, VPacker


################### DATA HERE #####################
CONDITIONS = [
    # (label components, value, color)
    (
        [
            [("Finetuned directly for", False)],
            [("the eval. benchmark", False)],
            [("(SIMIT-FT, Test-Set)", False)],
        ],
        6.39,
        "Color 1",
    ),
    (
        [
            [("Finetuned for", False)],
            [("a ", False), ("different benchmark", True)],
            [("from the ", False), ("same domain", True)],
        ],
        4.28,
        "Color 5",
    ),
    (
        [
            [("Finetuned for", False)],
            [("a ", False), ("different benchmark", True)],
            [("from a ", False), ("different domain", True)],
        ],
        0.53,
        "Color 8",
    ),
]
####################################################


def add_mixed_ylabel(ax, y_position, lines, fontsize=7.4):
    """Add a multiline y-axis label with selectively bold text."""

    packed_lines = []

    for line in lines:
        text_parts = [
            TextArea(
                text,
                textprops={
                    "fontsize": fontsize,
                    "fontweight": "bold" if is_bold else "normal",
                },
            )
            for text, is_bold in line
        ]

        packed_lines.append(
            HPacker(
                children=text_parts,
                align="baseline",
                pad=0,
                sep=0,
            )
        )

    packed_label = VPacker(
        children=packed_lines,
        align="right",
        pad=0,
        sep=2,
    )

    annotation = AnnotationBbox(
        packed_label,
        (0, y_position),
        xycoords=("axes fraction", "data"),
        xybox=(-7, 0),
        boxcoords="offset points",
        box_alignment=(1, 0.5),
        frameon=False,
        annotation_clip=False,
    )

    ax.add_artist(annotation)


fig, ax = plt.subplots(figsize=(3.7, 2.0))

y = np.arange(len(CONDITIONS))[::-1]
values = [condition[1] for condition in CONDITIONS]
colors = [
    color_palette_primary[condition[2]]
    for condition in CONDITIONS
]

bars = ax.barh(
    y,
    values,
    height=0.6,
    color=colors,
    alpha=global_alpha,
    edgecolor="white",
    linewidth=0.8,
    # zorder=3,
)

# Value labels
for yi, value in zip(y, values):
    label_value = f"+{value:.1f}%" if value > 0 else "0%"

    ax.text(
        value + 0.15,
        yi,
        label_value,
        ha="left",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#333333",
        # zorder=4,
    )

# Hide standard tick labels and add mixed-weight labels
ax.set_yticks(y)
ax.set_yticklabels([])
ax.tick_params(axis="y", length=0)

for yi, condition in zip(y, CONDITIONS):
    label_lines = condition[0]
    add_mixed_ylabel(ax, yi, label_lines)

ax.set_xlabel(
    "Gain over BAGEL-7B zero-shot (%)",
    fontsize=8,
)

ax.tick_params(
    axis="x",
    labelsize=7.5,
)

ax.set_xlim(
    0,
    max(values) * 1.2,
)

ax.set_ylim(
    -0.65,
    len(CONDITIONS) - 0.35,
)

ax.grid(
    True,
    axis="x",
    alpha=0.25,
    # zorder=0,
)

ax.set_axisbelow(True)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.subplots_adjust(
    left=0.42,
    right=0.97,
    bottom=0.18,
    top=0.97,
)

fig.savefig(
    f"{save_path}/cross_dataset_generalization_compact.pdf",
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()