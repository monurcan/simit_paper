"""Deterministic helpers shared across renderers: geometry notation helpers
(injected into the model's sandbox) and automatic text-overlap avoidance
(used both in-process by the deterministic renderers and injected into the
model's sandbox)."""

# Pre-defined geometry notation helpers, injected ahead of every model-written
# script so the model calls them by name instead of constructing this
# notation by hand -- it's unreliable at hand-rolling right-angle squares,
# equal-length tick marks, and angle arcs from scratch (wrong vertex, wrong
# offset direction, text overlapping the shape), but reliable at calling a
# documented function with the right arguments (see MASTER_PROMPT Example 26).
GEOMETRY_HELPERS_SRC = '''
import numpy as np
from matplotlib.patches import Polygon as _HelperPolygon, Arc as _HelperArc

def right_angle_marker(ax, vertex, p1, p2, size=0.3, **kwargs):
    vertex, p1, p2 = np.array(vertex, dtype=float), np.array(p1, dtype=float), np.array(p2, dtype=float)
    u1 = (p1 - vertex) / np.linalg.norm(p1 - vertex)
    u2 = (p2 - vertex) / np.linalg.norm(p2 - vertex)
    pts = [vertex, vertex + size * u1, vertex + size * u1 + size * u2, vertex + size * u2]
    kwargs.setdefault("fill", False)
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("linewidth", 1.2)
    ax.add_patch(_HelperPolygon(pts, closed=True, **kwargs))

def tick_marks(ax, p1, p2, count=1, size=0.15, gap=0.08, **kwargs):
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    mid = (p1 + p2) / 2
    along = (p2 - p1) / np.linalg.norm(p2 - p1)
    perp = np.array([-along[1], along[0]])
    kwargs.setdefault("color", "black")
    kwargs.setdefault("linewidth", 1.5)
    for off in (np.arange(count) - (count - 1) / 2) * gap:
        c = mid + along * off
        ax.plot([c[0] - perp[0] * size, c[0] + perp[0] * size],
                [c[1] - perp[1] * size, c[1] + perp[1] * size], **kwargs)

def angle_arc(ax, vertex, theta1, theta2, radius=0.5, label=None, label_radius_mult=1.4, **kwargs):
    kwargs.setdefault("color", "black")
    kwargs.setdefault("linewidth", 1.2)
    ax.add_patch(_HelperArc(vertex, 2 * radius, 2 * radius, angle=0, theta1=theta1, theta2=theta2, **kwargs))
    if label:
        mid = np.radians((theta1 + theta2) / 2)
        lx = vertex[0] + radius * label_radius_mult * np.cos(mid)
        ly = vertex[1] + radius * label_radius_mult * np.sin(mid)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=10)

def side_label(ax, p1, p2, text=None, offset=None, fmt="{:.2f}", **kwargs):
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    length = np.linalg.norm(p2 - p1)
    if text is None:
        text = fmt.format(length)
    if offset is None:
        offset = 0.35
    mid = (p1 + p2) / 2
    along = (p2 - p1) / length
    perp = np.array([-along[1], along[0]])
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "center")
    kwargs.setdefault("fontsize", 10)
    ax.text(*(mid + perp * offset), text, **kwargs)
'''


def autofix_text_overlaps(fig, max_iter: int = 40, min_font_size: float = 6.0) -> None:
    """Detects overlapping ax.text() bounding boxes across every Axes in
    `fig` and nudges colliding labels apart along the line between their
    centers, iterating until none overlap. Each text has a displacement
    budget (proportional to its own size) -- once a text has moved further
    than that without resolving its overlap, repositioning stops for it and
    its font shrinks instead, so a tight cluster that can't be cleanly
    separated ends up smaller rather than flung arbitrarily far from its
    meaningful anchor point. Used directly (in-process) by every
    deterministic renderer below; the model-facing FIGURE sandbox gets the
    equivalent logic via the separately-injected TEXT_OVERLAP_FIX_SRC string
    instead, since that runs in an isolated subprocess that can't import
    this module."""
    fig.canvas.draw()
    for ax in fig.get_axes():
        texts = [t for t in ax.texts if t.get_text().strip()]
        if len(texts) < 2:
            continue
        renderer = fig.canvas.get_renderer()
        budget = [1.3 * max(t.get_window_extent(renderer=renderer).width,
                             t.get_window_extent(renderer=renderer).height) for t in texts]
        traveled = [0.0] * len(texts)
        for _ in range(max_iter):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bboxes = [t.get_window_extent(renderer=renderer) for t in texts]
            moved = False
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    bi, bj = bboxes[i], bboxes[j]
                    if not bi.overlaps(bj):
                        continue
                    cxi, cyi = bi.x0 + bi.width / 2, bi.y0 + bi.height / 2
                    cxj, cyj = bj.x0 + bj.width / 2, bj.y0 + bj.height / 2
                    dx, dy = cxj - cxi, cyj - cyi
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    if dist < 1e-6:
                        dx, dy, dist = float(j - i), 0.0, float(j - i)
                    ux, uy = dx / dist, dy / dist
                    overlap_x = (bi.width + bj.width) / 2 - abs(dx)
                    overlap_y = (bi.height + bj.height) / 2 - abs(dy)
                    push = max(overlap_x, overlap_y, 2.0) * 0.5 + 1.5
                    for idx, t, sign in ((i, texts[i], -1), (j, texts[j], 1)):
                        if traveled[idx] >= budget[idx]:
                            if t.get_fontsize() > min_font_size:
                                t.set_fontsize(max(min_font_size, t.get_fontsize() * 0.85))
                                moved = True
                            continue
                        trans = t.get_transform()
                        x, y = t.get_position()
                        px, py = trans.transform((x, y))
                        nx, ny = trans.inverted().transform((px + sign * ux * push, py + sign * uy * push))
                        t.set_position((nx, ny))
                        traveled[idx] += push
                        moved = True
            if not moved:
                break


# The model can't be relied on to predict where its own labels will end up
# relative to each other -- it places each ax.text() independently with no
# view of the others. This patches plt.savefig (transparently, the model's
# own `import matplotlib.pyplot as plt` binds the same module object) to
# detect overlapping text bounding boxes across every Axes right before the
# real save, and nudge colliding labels apart along the line between their
# centers, iterating until none overlap. A deterministic, geometric fix --
# exactly the kind of thing that's calculable rather than left to chance.
# (Duplicated as source text rather than shared with autofix_text_overlaps()
# above because this runs inside an isolated model-code subprocess that
# can't import this module.)
TEXT_OVERLAP_FIX_SRC = '''
import matplotlib.pyplot as _overlap_plt

def _shift_text_px(text, dx_px, dy_px):
    trans = text.get_transform()
    x, y = text.get_position()
    px, py = trans.transform((x, y))
    nx, ny = trans.inverted().transform((px + dx_px, py + dy_px))
    text.set_position((nx, ny))

def _autofix_text_overlaps(fig, max_iter=40, min_font_size=6.0):
    fig.canvas.draw()
    for ax in fig.get_axes():
        texts = [t for t in ax.texts if t.get_text().strip()]
        if len(texts) < 2:
            continue
        renderer = fig.canvas.get_renderer()
        budget = [1.3 * max(t.get_window_extent(renderer=renderer).width,
                             t.get_window_extent(renderer=renderer).height) for t in texts]
        traveled = [0.0] * len(texts)
        for _ in range(max_iter):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bboxes = [t.get_window_extent(renderer=renderer) for t in texts]
            moved = False
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    bi, bj = bboxes[i], bboxes[j]
                    if not bi.overlaps(bj):
                        continue
                    cxi, cyi = bi.x0 + bi.width / 2, bi.y0 + bi.height / 2
                    cxj, cyj = bj.x0 + bj.width / 2, bj.y0 + bj.height / 2
                    dx, dy = cxj - cxi, cyj - cyi
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    if dist < 1e-6:
                        dx, dy, dist = float(j - i), 0.0, float(j - i)
                    ux, uy = dx / dist, dy / dist
                    overlap_x = (bi.width + bj.width) / 2 - abs(dx)
                    overlap_y = (bi.height + bj.height) / 2 - abs(dy)
                    push = max(overlap_x, overlap_y, 2.0) * 0.5 + 1.5
                    for idx, t, sign in ((i, texts[i], -1), (j, texts[j], 1)):
                        if traveled[idx] >= budget[idx]:
                            if t.get_fontsize() > min_font_size:
                                t.set_fontsize(max(min_font_size, t.get_fontsize() * 0.85))
                                moved = True
                            continue
                        _shift_text_px(t, sign * ux * push, sign * uy * push)
                        traveled[idx] += push
                        moved = True
            if not moved:
                break

_original_savefig = _overlap_plt.savefig

def _savefig_with_overlap_fix(*args, **kwargs):
    try:
        _autofix_text_overlaps(_overlap_plt.gcf())
    except Exception:
        pass
    return _original_savefig(*args, **kwargs)

_overlap_plt.savefig = _savefig_with_overlap_fix
'''
