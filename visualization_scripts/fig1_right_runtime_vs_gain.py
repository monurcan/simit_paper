#!/usr/bin/env python3
r"""
SIMIT -- Figure 1 (right panel). Replacement for the radar chart.

Cost vs. benefit: end-to-end runtime (x, log, relative to zero-shot) against
mean relative gain over the zero-shot baseline (y). Shows the two strongest
SIMIT configurations against all TTS and LFRL baselines, each cited.

Data: Table 1 (runtimes, per-domain gains) and Table 5 (per-benchmark means).

METRIC
------
"domain_macro"    y = mean of the 5 domain-level relative gains of Tab. 1.
                  Reproducible from the paper as submitted, but NOT the number
                  in the abstract (SIMIT-ICL -> +3.86%, not +6.98%): averaging
                  inside a domain first washes out NoCaps / OK-VQA.

"benchmark_mean"  y = mean of the 17 benchmark-level relative gains, i.e. the
                  headline metric of the abstract, Tab. 5 and Sec. 4.1.
                  SIMIT's values are known (+6.98 / +7.20); fill in `bench=`
                  for the baselines from your own per-benchmark results.
                  ---> SHIP THIS ONE. Mixing metrics across methods would
                       overstate SIMIT, and the script refuses to run until
                       every baseline value is supplied.

Note that the gap band and its caption are computed from the data, so they
stay correct when the metric changes.

CITATIONS
---------
Baked in as dead text, so keep them in sync with the .bib by hand. CITE_STYLE
"full" does not fit -- "(Zuo et al., 2025)" overruns the SRLM marker -- so the
"venue" form is the default. The caption should still carry the authoritative
\citep{} for every method.

SIZE
----
Square, to match the shape of the Fig. 1 left panel (the radar/spider chart,
teaser_spider.py). SCALE bumps every hand-tuned pt size (fonts, markers,
line widths, label offsets) up together so the layout stays proportioned
the same way it was at the old tiny print size -- just larger and square.
include with \includegraphics[width=0.55\textwidth]{...}.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from global_vars import *
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from matplotlib.artist import Artist
from matplotlib.colors import to_rgba
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
from matplotlib.offsetbox import DrawingArea, VPacker
from matplotlib.markers import MarkerStyle
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.transforms import Affine2D, Bbox

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
METRIC      = "benchmark_mean"  # "domain_macro" | "benchmark_mean"
CITE_STYLE  = "author_year"     # "author_year" -> Singh 2026 | "venue_only"  -> CVPR 2026
                                # "venue"       -> Singh, CVPR 2026  (does NOT fit, see below)
                                # "venue_short" -> Singh, CVPR'26    (does NOT fit)
                                # "full"        -> Singh et al., CVPR 2026 (does NOT fit)
SCALE       = 1.4               # uniform bump for every pt-sized constant below
FIGSIZE     = (4.0 * 0.88, 4.3 * 0.86)        # inches, square like the Fig. 1 left (spider) panel
OUT         = f"{save_path}/fig1_right_runtime_vs_gain"
BREAK_BELOW = -5.0              # points below this go in the broken strip
SHOW_PARETO = False             # staircase frontier; the gap band replaces it
SHOW_VALUES = True              # print the gain next to the SIMIT markers
SHOW_GAP    = True              # wash + caption for the gap to the best prior method
TOP_PAD     = 0.13              # headroom above the top point, as a frac of range
BOT_PAD     = 0.06              # ditto below the bottom point
LINE        = 6.0 * SCALE       # title lift for labels above their markers
LABEL_GAP   = 1.0 * SCALE       # visible gap between title and citation/gain, in points
PAD_V       = 3                 # points: marker-to-label vertical gap, all bottom/top labels

C_SIMIT = color_palette_primary["Color 9"]   # shared SIMIT annotations
C_SIMIT_ICL = color_palette_primary["Color 9"]
C_SIMIT_FT = color_palette_primary["Color 5"]
C_LFRL  = color_palette_primary["Color 10"]    # label-free RL
C_TTS   = color_palette_primary["Color 2"]    # test-time scaling
C_BASE  = color_palette_primary["Color 6"]
C_TEXT  = "#3A3A3A"
C_CITE  = "#8A8A8A"
STAR_SPLIT_ANGLE = 0.0        # degrees from the original vertical split
STAR_BLEND_WIDTH = 0.35       # transition width relative to marker size

C_BAND  = "#F2F2F3"             # "worse than zero-shot" wash

plt.rcParams.update({
    "font.family": global_font,
    "mathtext.fontset": "dejavusans",
    "font.size": 9 * SCALE,
    "axes.labelsize": 7.4 * SCALE,
    "xtick.labelsize": 6.6 * SCALE,
    "ytick.labelsize": 6.6 * SCALE,
    "axes.linewidth": 0.55 * SCALE,
    "axes.edgecolor": "#4A4A4A",
    "xtick.color": "#4A4A4A",
    "ytick.color": "#4A4A4A",
    "xtick.major.width": 0.55 * SCALE,
    "ytick.major.width": 0.55 * SCALE,
    "xtick.major.size": 2.4 * SCALE,
    "ytick.major.size": 2.4 * SCALE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def cite(first, venue, year):
    """Citation string for in-figure use. Keep in sync with the .bib."""
    yr = f"'{str(year)[-2:]}" if CITE_STYLE == "venue_short" else f" {year}"
    if CITE_STYLE == "venue_only":
        return f"{venue}{yr}"
    if CITE_STYLE == "author_year":
        return f"{first} {year}"
    if CITE_STYLE == "full":
        return f"{first} et al., {venue}{yr}"
    return f"{first}, {venue}{yr}"


# --------------------------------------------------------------------------- #
# data   domain = (Image, VQA, Math&Sci, Doc&Info, Multi) relative gains, %
#        off = label anchor in points; grp joins a hyper-parameter sweep
# --------------------------------------------------------------------------- #
ROWS = [
    dict(label="BAGEL-7B\nZero-Shot", cite=cite("Deng", "arXiv", 2025), fam="base", rt=1, grp=None,
         domain=(0, 0, 0, 0, 0), bench=0.00,
         off=(-1.5, -4), ha="left", va="top"),

    # ---- test-time scaling ------------------------------------------------ #
    dict(label="Thinking", cite=cite("Wei", "NeurIPS", 2022), fam="tts", rt=19, grp="think",
         domain=(-54.0, -8.1, -25.7, -7.1, -4.3), bench=-21.59,
         off=(-6, -4), ha="right", va="bottom"),
    dict(label=None, fam="tts", rt=44, grp="think",
         domain=(-53.9, -8.0, -25.2, -6.5, -3.7), bench=-21.28),

    dict(label=None, fam="tts", rt=5, grp="refine",
         domain=(-7.1, -3.2, 0.0, -1.9, 1.4), bench=-2.90),
    dict(label=None, fam="tts", rt=16, grp="refine",
         domain=(-7.1, -3.2, 0.0, -1.9, 1.4), bench=-2.87),
    dict(label="Self-Refine", cite=cite("Madaan", "NeurIPS", 2023), fam="tts", rt=25,
         grp="refine", domain=(-7.1, -3.2, 0.0, -1.8, 1.5), bench=-2.84,
         off=(-31, PAD_V), ha="left", va="bottom"),

    dict(label=None, fam="tts", rt=5, grp="sc",
         domain=(-0.3, -0.3, -0.2, -0.4, 0.6), bench=-0.20),
    dict(label=None, fam="tts", rt=9, grp="sc",
         domain=(0.0, 0.2, 0.0, -0.4, 0.3), bench=0.17),
    dict(label="Self-Consistency", cite=cite("Wang", "ICLR", 2023), fam="tts", rt=19,
         grp="sc", domain=(1.1, 0.2, 0.5, 0.0, -0.2), bench=0.41,
         off=(-3, PAD_V), ha="right", va="bottom"),

    # ---- label-free RL ---------------------------------------------------- #
    dict(label="TTRV", cite=cite("Singh", "CVPR", 2026), fam="lfrl", rt=41, grp=None,
         domain=(1.2, 2.8, 0.5, 1.5, 0.7), bench=2.07,
         off=(0, PAD_V), ha="right", va="bottom"),
    dict(label="EMPO", cite=cite("Zhang", "NeurIPS", 2026), fam="lfrl", rt=58, grp=None,
         domain=(1.9, 2.3, 0.0, 1.1, -0.1), bench=1.82,
         off=(4, PAD_V), ha="left", va="bottom"),
    dict(label="TTRL", cite=cite("Zuo", "NeurIPS", 2025), fam="lfrl", rt=75, grp=None,
         domain=(0.6, 0.3, 0.2, 1.5, -0.3), bench=0.59,
         off=(2, -8), ha="right", va="top"),
    dict(label="SRLM", cite=cite("Yuan", "ICML", 2024), fam="lfrl", rt=251, grp=None,
         domain=(-1.0, 2.5, -0.2, 0.2, 0.2), bench=0.87,
         off=(-1, -PAD_V - 0.5), ha="left", va="top"),

    # ---- ours (strongest configurations) ----------------------------------- #
    dict(label="SIMIT-ICL", fam="simit", color=C_SIMIT_ICL, rt=25, grp=None,
         domain=(8.3, 6.5, 0.9, 1.1, 2.5), bench=6.98,
         off=(-22, 6.5), ha="left", va="bottom",
         align_title_with=r"SIMIT-FT$\rightarrow$ICL"),
    dict(label="SIMIT-FT", fam="simit", color=C_SIMIT_FT, rt=55, grp=None,
         domain=(5.8, 7.5, 1.4, 1.3, 2.2), bench=6.39,
         off=(8, 3.5), ha="left", va="top"),
    dict(label=r"SIMIT-FT$\rightarrow$ICL", fam="simit",
         color=C_SIMIT_ICL, color_alt=C_SIMIT_FT, rt=55, grp=None,
         domain=(6.9, 7.7, 1.6, 1.6, 3.2), bench=7.20,
         off=(8, 6.5), ha="left", va="bottom"),
]

STYLE = {
    "base":  dict(m="X", s=27 * SCALE**2,  c=C_BASE,  ec="white", lw=0.5 * SCALE, z=6),
    "tts":   dict(m="o", s=15 * SCALE**2,  c=C_TTS,   ec="white", lw=0.4 * SCALE, z=4),
    "lfrl":  dict(m="s", s=15 * SCALE**2,  c=C_LFRL,  ec="white", lw=0.4 * SCALE, z=4),
    "simit": dict(m="*", s=105 * SCALE**2, c=C_SIMIT, ec="white", lw=0.5 * SCALE, z=7),
}


def gain(r):
    if METRIC == "benchmark_mean":
        if r["bench"] is None:
            raise SystemExit(
                f"\n  No benchmark-level mean for '{r['label'] or r['fam']}' "
                f"({r['rt']}x).\n"
                "  Fill bench=<mean of the 17 per-benchmark relative gains> for\n"
                "  every baseline row, or set METRIC = 'domain_macro'.\n")
        return float(r["bench"])
    return float(np.mean(r["domain"]))


for r in ROWS:
    r["x"], r["y"] = float(r["rt"]), gain(r)

print(f"{'method':<24}{'runtime':>9}{'gain':>9}   [{METRIC}]")
for r in ROWS:
    print(f"{r['label'] or '   (sweep)':<24}{r['x']:>8.0f}x{r['y']:>8.2f}%")

# --------------------------------------------------------------------------- #
# axes
# --------------------------------------------------------------------------- #
low = [r for r in ROWS if r["y"] < BREAK_BELOW]
hi = [r for r in ROWS if r["y"] >= BREAK_BELOW]

if low:
    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=FIGSIZE, sharex=True,
        gridspec_kw=dict(height_ratios=[8.0, 1.0], hspace=0.04))
    axes = [ax, axb]
else:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    axb, axes = None, [ax]

ys = [r["y"] for r in hi]
rng = max(ys) - min(ys)
ax.set_ylim(min(ys) - BOT_PAD * rng, max(ys) + TOP_PAD * rng)
if axb is not None:
    lo = [r["y"] for r in low]
    axb.set_ylim(min(lo) - 1.9, max(lo) + 1.9)
    axb.set_yticks([round(np.mean(lo))])

ax.set_xscale("log")
ax.set_xlim(0.75, 900)
ticks = [1, 5, 10, 25, 50, 100, 250]
axes[-1].set_xticks(ticks)
axes[-1].set_xticklabels([f"{t}$\\times$" for t in ticks])
axes[-1].xaxis.set_minor_locator(NullLocator())

for a in axes:
    a.set_axisbelow(True)
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.axhspan(a.get_ylim()[0], 0, color=C_BAND, lw=0, zorder=0)
    a.yaxis.grid(True, ls=(0, (1, 3)), lw=0.45 * SCALE, color="#C9CDD2", zorder=0.5)

ax.axhline(0, ls=(0, (4, 2.5)), lw=0.6 * SCALE, c="#8A8A8A", zorder=1)

# headroom no prior self-improvement method reaches
best_prior = max(r["y"] for r in ROWS if r["fam"] not in ("simit", "base"))
best_ours = max(r["y"] for r in ROWS if r["fam"] == "simit")
if SHOW_GAP:
    ax.axhspan(best_prior, best_ours, color=C_SIMIT, alpha=0.055, lw=0, zorder=-1)
    ax.annotate(f"+{best_ours - best_prior:.1f} points over SOTA self-improvement methods",
                (ax.get_xlim()[0], 0.5 * (best_prior + best_ours)),
                textcoords="offset points", xytext=(3.5 * SCALE, 0), ha="left",
                va="center", fontsize=5.9 * SCALE, color=C_SIMIT, alpha=0.8,
                linespacing=1.2, zorder=8,
                fontweight="bold",)

# broken-axis cosmetics
if axb is not None:
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False)
    axb.spines["top"].set_visible(False)
    # Extend the lower band's shading through the gap, retaining the axis break.
    upper_box, lower_box = ax.get_position(), axb.get_position()
    fig.add_artist(Rectangle(
        (lower_box.x0, lower_box.y1), lower_box.width,
        upper_box.y0 - lower_box.y1, transform=fig.transFigure,
        facecolor=C_BAND, edgecolor="none", zorder=-1))
    # Marker coordinates preserve a 45-degree angle regardless of axes aspect.
    kw = dict(marker=[(-1, -1), (1, 1)], markersize=6 * SCALE,
              linestyle="none", color="#4A4A4A", markeredgewidth=0.55 * SCALE,
              clip_on=False)
    ax.plot([0], [0], transform=ax.transAxes, **kw)
    axb.plot([0], [1], transform=axb.transAxes, **kw)


def host(r):
    return axb if (axb is not None and r["y"] < BREAK_BELOW) else ax


# hyper-parameter sweeps of one method, joined
for g in {r["grp"] for r in ROWS if r["grp"]}:
    pts = sorted([r for r in ROWS if r["grp"] == g], key=lambda r: r["x"])
    if len({id(host(p)) for p in pts}) == 1:
        host(pts[0]).plot([p["x"] for p in pts], [p["y"] for p in pts], "-",
                          lw=0.8 * SCALE, c=C_TTS, alpha=0.75, zorder=3,
                          solid_capstyle="round")

# Pareto frontier: best gain reachable at or below a given runtime
if SHOW_PARETO:
    seq = sorted([(r["x"], r["y"]) for r in ROWS])
    fx, fy, best = [], [], -np.inf
    for x, y in seq:
        if y > best:
            best, _ = y, fx.append(x)
            fy.append(y)
    sx, sy = [fx[0]], [fy[0]]
    for i in range(1, len(fx)):
        sx += [fx[i], fx[i]]
        sy += [sy[-1], fy[i]]
    ax.plot(sx, sy, lw=0.75 * SCALE, c=C_SIMIT, alpha=0.22, zorder=2, ls=(0, (3.5, 2)))


class GradientTextEffect(AbstractPathEffect):
    """Fill text outlines with the same orange-to-green blend as the star."""

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        bounds = tpath.get_extents(affine)
        if bounds.width <= 0 or bounds.height <= 0:
            return
        blend = np.clip(0.05 + np.linspace(-0.5, 0.5, 512)
                        / STAR_BLEND_WIDTH, 0, 1)
        blend = (blend * blend * (3 - 2 * blend))[None, :, None]
        rgba = ((1 - blend) * np.array(to_rgba(C_SIMIT_FT))
                + blend * np.array(to_rgba(C_SIMIT_ICL)))
        bitmap = BboxImage(bounds, origin="lower", interpolation="bilinear")
        bitmap.set_data(np.repeat(rgba, 2, axis=0))
        bitmap.set_clip_path(tpath, affine)
        bitmap.draw(renderer)


def label_rows(r, a):
    """Place every citation/gain the same distance below its title's bounds."""
    ours = r["fam"] == "simit"
    dx, dy = r["off"][0] * SCALE, r["off"][1] * SCALE
    va, ref = r["va"], r.get("cite")
    label_y = r["y"]
    if "align_title_with" in r:
        target = next(row for row in ROWS if row["label"] == r["align_title_with"])
        label_y = target["y"]
        dy = target["off"][1] * SCALE
        va = target["va"]
    if ref and va == "bottom":
        dy += LINE
    elif ref and va == "center":
        dy += LINE / 2
        va = "bottom"
    effects = [GradientTextEffect()] if "color_alt" in r else []
    title = a.annotate(
        r["label"], (r["x"], label_y), textcoords="offset points",
        xytext=(dx, dy), ha=r["ha"], va=va,
        fontsize=(6.5 if ours else 6.3) * SCALE,
        color=r["color"] if ours else C_TEXT,
        fontweight="bold" if ours else "normal", path_effects=effects, zorder=8)
    detail = f"{r['y']:+.2f}%" if ours and SHOW_VALUES else ref
    if detail:
        anchor = {"left": 0, "center": 0.5, "right": 1}[r["ha"]]
        a.annotate(
            detail, (anchor, 0), xycoords=title, textcoords="offset points",
            xytext=(0, -LABEL_GAP), ha=r["ha"], va="top",
            fontsize=(5.9 if ours else 5.1) * SCALE,
            color=r["color"] if ours else C_CITE,
            path_effects=effects, zorder=8)



class GradientStar(Artist):
    """Fixed-size star with an angled color blend, shared by plot and legend."""

    def __init__(self, x, y, size, edgewidth, **kwargs):
        super().__init__()
        self.set(**kwargs)
        self.x, self.y = x, y
        self.size, self.edgewidth = size, edgewidth
        marker = MarkerStyle("*")
        self.path = marker.get_path().transformed(marker.get_transform())
        coords = np.linspace(-0.5, 0.5, 256)
        xx, yy = np.meshgrid(coords, coords)
        angle = np.deg2rad(STAR_SPLIT_ANGLE)
        distance = xx * np.cos(angle) - yy * np.sin(angle)
        blend = np.clip(0.5 + distance / STAR_BLEND_WIDTH, 0, 1)
        blend = (blend * blend * (3 - 2 * blend))[..., None]
        self.rgba = ((1 - blend) * np.array(to_rgba(C_SIMIT_FT))
                     + blend * np.array(to_rgba(C_SIMIT_ICL)))

    def draw(self, renderer):
        if not self.get_visible():
            return
        x, y = self.get_transform().transform((self.x, self.y))
        size = renderer.points_to_pixels(self.size)
        transform = Affine2D().scale(size).translate(x, y)
        patch = PathPatch(self.path, transform=transform, facecolor="none",
                          edgecolor="white", linewidth=self.edgewidth)
        patch.set_figure(self.get_figure())
        bitmap = BboxImage(Bbox.from_bounds(x - size / 2, y - size / 2, size, size),
                           origin="lower", interpolation="bilinear")
        bitmap.set_figure(self.get_figure())
        bitmap.set_data(self.rgba)
        bitmap.set_clip_path(self.path, transform)
        bitmap.draw(renderer)
        patch.draw(renderer)
        self.stale = False


class HandlerGradientStar(HandlerBase):
    def create_artists(self, legend, handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        return [GradientStar(width / 2 - xdescent, height / 2 - ydescent,
                             handle.size, handle.edgewidth, transform=trans)]


for r in ROWS:
    a, st = host(r), STYLE[r["fam"]]
    if "color_alt" in r:
        a.add_artist(GradientStar(r["x"], r["y"], np.sqrt(st["s"]), st["lw"],
                                  transform=a.transData, zorder=st["z"],
                                  clip_on=False))
    elif r["fam"] == "simit":
        a.plot(r["x"], r["y"], ls="", marker=st["m"], ms=np.sqrt(st["s"]),
               markerfacecolor=r["color"],
               markeredgecolor=st["ec"], markeredgewidth=st["lw"],
               zorder=st["z"], clip_on=False)
    else:
        a.scatter(r["x"], r["y"], marker=st["m"], s=st["s"], c=st["c"],
                  edgecolors=st["ec"], linewidths=st["lw"], zorder=st["z"],
                  clip_on=False)
    if not r["label"]:
        continue
    label_rows(r, a)

ax.set_ylabel("Mean relative gain over zero-shot (%)", labelpad=2.5 * SCALE, color="#2A2A2A")
axes[-1].set_xlabel("End-to-end runtime (relative to zero-shot)",
                    labelpad=2.5 * SCALE, color="#2A2A2A")

handles = [
    GradientStar(0, 0, 6.0 * SCALE, 0.4 * SCALE, label="SIMIT (Ours)"),
    Line2D([], [], ls="", marker="s", ms=2.6 * SCALE, c=C_LFRL, label="Label-Free RL"),
    Line2D([], [], ls="", marker="o", ms=2.6 * SCALE, c=C_TTS, label="Test-Time Scaling"),
]
leg = fig.legend(handles=handles, handler_map={GradientStar: HandlerGradientStar()}, loc="lower right", fontsize=5.9 * SCALE,
                # frameon=False,
                handlelength=1.0, handletextpad=0.2, borderpad=0.3, labelspacing=0.1,
                borderaxespad=0.0,
                bbox_to_anchor=(1.0, -0.13), bbox_transform=ax.transAxes)
leg.get_frame().set_linewidth(0.5)


# Use narrow side padding with separate top/bottom padding.
legend_vertical_pad = 0.1 * leg.get_texts()[0].get_fontsize()
legend_entries = leg._legend_box.get_children()
legend_entries[-1] = VPacker(
    children=[DrawingArea(0, legend_vertical_pad), legend_entries[-1],
              DrawingArea(0, legend_vertical_pad)],
    align="left", pad=0, sep=0)

fig.savefig(f"{OUT}.pdf", bbox_inches="tight", pad_inches=0.012)
fig.savefig(f"{OUT}.png", dpi=440, bbox_inches="tight", pad_inches=0.012)
print(f"\nwrote {OUT}.pdf / {OUT}.png")
