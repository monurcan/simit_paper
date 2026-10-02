"""Deterministic DSL parsers and renderers, one per category -- each takes
the model's narrow extracted structure (never its layout code) and renders
it with a real plotting/drawing library."""

import re
import textwrap

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import schemdraw
import schemdraw.elements as elm
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D

from .helpers import autofix_text_overlaps

# Bounded by a lookahead for the next NODE:/EDGE: marker (or end of string)
# rather than end-of-line: the model sometimes puts more than one on a
# single line, and a line-anchored ($) regex would swallow the rest of the
# line into the first match's captured text instead of finding the next one.
NODE_RE = re.compile(r"NODE:\s*(.+?)\s*(?=NODE:|EDGE:|$)", re.IGNORECASE | re.DOTALL)


EDGE_RE = re.compile(
    r"EDGE:\s*(.+?)\s*->\s*(.+?)\s*(?::\s*(.+?))?\s*(?=NODE:|EDGE:|$)",
    re.IGNORECASE | re.DOTALL,
)


def parse_dsl(text: str) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Parse NODE:/EDGE: lines into (node_order, edges). Edges referencing an
    undeclared node append it defensively so a slightly malformed response
    still renders something reasonable instead of silently dropping content."""
    nodes: list[str] = []
    seen = set()
    for m in NODE_RE.finditer(text):
        name = m.group(1).strip()
        if name and name not in seen:
            nodes.append(name)
            seen.add(name)

    raw_edges: list[tuple[str, str, str]] = []
    for m in EDGE_RE.finditer(text):
        src, dst, label = m.group(1).strip(), m.group(2).strip(), (m.group(3) or "").strip()
        for name in (src, dst):
            if name not in seen:
                nodes.append(name)
                seen.add(name)
        raw_edges.append((src, dst, label))

    # The model occasionally emits the same (src, dst) connection twice with
    # conflicting labels (e.g. both ": yes" and ": no" on a step that isn't
    # actually the branch point) -- collapse duplicates into one edge and
    # drop the label entirely when it's ambiguous, rather than rendering two
    # overlapping arrows or guessing which label is "right".
    by_pair: dict[tuple[str, str], list[str]] = {}
    pair_order: list[tuple[str, str]] = []
    for src, dst, label in raw_edges:
        key = (src, dst)
        if key not in by_pair:
            by_pair[key] = []
            pair_order.append(key)
        if label:
            by_pair[key].append(label)

    edges: list[tuple[str, str, str]] = []
    for src, dst in pair_order:
        labels = by_pair[(src, dst)]
        label = labels[0] if len(set(labels)) == 1 and labels else ""
        edges.append((src, dst, label))
    return nodes, edges


def render_flowchart(
    nodes: list[str], edges: list[tuple[str, str, str]], out_path: str,
    spacing: float = 2.2, box_h: float = 0.9,
) -> None:
    """Deterministic auto-layout: nodes stack top-to-bottom in declaration
    order; any edge that doesn't connect immediately-adjacent nodes (a
    loop-back, or a forward edge that skips over other nodes -- e.g. one node
    fanning out to two non-adjacent destinations) is drawn as a curved arrow
    around the side instead of straight through intervening boxes. No LLM
    involvement -- this always produces a structurally correct diagram for
    whatever node/edge list it's given."""
    order = {name: i for i, name in enumerate(nodes)}
    positions = {
        name: (0.0, -i * spacing, max(2.6, 0.13 * len(name) + 0.8))
        for i, name in enumerate(nodes)
    }

    fig, ax = plt.subplots(figsize=(7.0, max(4.0, 1.3 * len(nodes))))
    for name, (cx, cy, w) in positions.items():
        patch = FancyBboxPatch(
            (cx - w / 2, cy - box_h / 2), w, box_h, boxstyle="round,pad=0.08",
            edgecolor="black", facecolor="#cfe8ff",
        )
        ax.add_patch(patch)
        ax.text(cx, cy, name, ha="center", va="center", fontsize=10)

    side_edge_count = 0
    max_w = max((w for _, _, w in positions.values()), default=2.6)
    max_extent = max_w / 2  # widest point reached by any box, curve, or label so far
    for src, dst, label in edges:
        if src not in positions or dst not in positions:
            continue
        sx, sy, sw = positions[src]
        dx, dy, dw = positions[dst]
        if order[dst] == order[src] + 1:  # adjacent: straight arrow down
            arrow = FancyArrowPatch(
                (sx, sy - box_h / 2), (dx, dy + box_h / 2),
                arrowstyle="-|>", mutation_scale=15, color="black",
            )
            ax.add_patch(arrow)
            if label:
                ax.text((sx + dx) / 2 + 0.3, (sy + dy) / 2, label, fontsize=9)
        else:  # non-adjacent (loop-back or skip-ahead): route around the side
            # with a right-angle (Manhattan) path rather than a curve -- a
            # bezier's bulge can visually cross straight edges and ambiguously
            # merge into a box's edge for short hops; a right-angle detour
            # stays unambiguous regardless of how close src and dst are.
            side_edge_count += 1
            route_x = max_w / 2 + 0.8 + 0.7 * (side_edge_count - 1)
            p1 = (sx + sw / 2, sy)
            p2 = (dx + dw / 2, dy)
            verts = [p1, (route_x, sy), (route_x, dy), p2]
            codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO]
            arrow = FancyArrowPatch(
                path=MplPath(verts, codes), arrowstyle="-|>",
                mutation_scale=15, color="black", linewidth=1.3,
            )
            ax.add_patch(arrow)
            if label:
                ax.text(route_x + 0.15, (sy + dy) / 2, label, fontsize=9, va="center")
                max_extent = max(max_extent, route_x + len(label) * 0.12 + 0.5)
            else:
                max_extent = max(max_extent, route_x + 0.3)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    margin_x = max_extent + 0.8
    ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax.set_ylim(min(ys) - 1.2, max(ys) + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    autofix_text_overlaps(fig)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


SMILES_RE = re.compile(r"SMILES:\s*(\S+)", re.IGNORECASE)


TITLE_RE = re.compile(r"TITLE:\s*(.+)", re.IGNORECASE)


def parse_molecule(text: str) -> tuple[str | None, str]:
    smiles_m = SMILES_RE.search(text)
    title_m = TITLE_RE.search(text)
    smiles = smiles_m.group(1).strip() if smiles_m else None
    title = title_m.group(1).strip() if title_m else ""
    return smiles, title


# Testing revealed the model's chemistry *knowledge* (not its SMILES syntax)
# is unreliable even for common molecules: asked for ethanol, benzene, and
# glucose, it confidently returned syntactically-valid SMILES for the wrong
# molecule every time (acetic acid, toluene, an unrelated branched alcohol).
# Prompting can't fix a model not knowing a fact. A small lookup table for
# the molecules anyone is actually likely to ask for is far more reliable
# than hoping a 7B general-purpose model recalls them correctly; the LLM
# path remains as a fallback for anything not in this list.
COMMON_MOLECULES = {
    "water": "O", "methane": "C", "ethanol": "CCO", "benzene": "c1ccccc1",
    "carbon dioxide": "O=C=O", "ammonia": "N", "acetic acid": "CC(=O)O",
    "glucose": "C(C1C(C(C(C(O1)O)O)O)O)O", "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "methanol": "CO",
    "propane": "CCC", "butane": "CCCC", "ethylene": "C=C", "acetylene": "C#C",
    "toluene": "Cc1ccccc1", "phenol": "Oc1ccccc1", "glycine": "NCC(=O)O",
    "sodium chloride": "[Na+].[Cl-]", "table salt": "[Na+].[Cl-]",
    "hydrogen peroxide": "OO", "sulfuric acid": "OS(=O)(=O)O",
    "acetone": "CC(=O)C", "formaldehyde": "C=O", "urea": "NC(=O)N",
    "cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    "nicotine": "CN1CCCC1c1cccnc1", "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "adrenaline": "CNCC(O)c1ccc(O)c(O)c1", "epinephrine": "CNCC(O)c1ccc(O)c(O)c1",
}


def lookup_common_molecule(request: str) -> tuple[str, str] | None:
    lowered = request.lower()
    for name, smiles in COMMON_MOLECULES.items():
        if name in lowered:
            return smiles, name.title()
    return None


CIRCUIT_LINE_SPLIT_RE = re.compile(r"(COMPONENT|PARALLEL):\s*", re.IGNORECASE)


CIRCUIT_TYPE_MAP = {
    "battery": elm.SourceV, "resistor": elm.Resistor, "capacitor": elm.Capacitor,
    "inductor": elm.Inductor, "led": elm.LED, "switch": elm.Switch, "diode": elm.Diode,
}


# Each slot in the series chain is either ("single", type, label) or
# ("parallel", [(type, label), ...]) -- a parallel group counts as one slot.
CircuitSlot = tuple


def _parse_component_token(token: str) -> tuple[str, str] | None:
    parts = token.strip().split(None, 1)
    if not parts:
        return None
    comp_type = parts[0].strip().lower()
    label = parts[1].strip() if len(parts) > 1 else comp_type
    return (comp_type, label) if comp_type in CIRCUIT_TYPE_MAP else None


def parse_circuit_dsl(text: str) -> list[CircuitSlot]:
    # Split on the COMPONENT:/PARALLEL: markers themselves rather than
    # anchoring each match to a line -- the model sometimes puts several on
    # one line, and a line-anchored regex would swallow the rest of the line
    # into the first match's label instead of finding the next one.
    slots: list[CircuitSlot] = []
    pieces = CIRCUIT_LINE_SPLIT_RE.split(text)[1:]  # alternating [kind, body, kind, body, ...]
    for kind, body in zip(pieces[0::2], pieces[1::2]):
        line = body.strip().splitlines()[0] if body.strip() else ""
        if not line:
            continue
        if kind.upper() == "PARALLEL":
            group = [c for tok in line.split("|") if (c := _parse_component_token(tok))]
            if group:
                slots.append(("parallel", group))
        else:
            single = _parse_component_token(line)
            if single:
                slots.append(("single", single[0], single[1]))
    return slots


def _draw_parallel_branch(d: "schemdraw.Drawing", group: list[tuple[str, str]]):
    """Draw `group` as parallel branches between the current point and a
    common rejoin point, advancing the drawing past the rejoin. Generalizes
    to any number of branches (verified with 2 and 3)."""
    ends = []
    for i, (comp_type, label) in enumerate(group):
        if i > 0:
            d.add(elm.Line().right(d.unit / 2))
        if i < len(group) - 1:
            d.push()
        d.add(CIRCUIT_TYPE_MAP[comp_type]().down().label(label))
        ends.append(d.here)
        if i < len(group) - 1:
            d.pop()
    for e in ends[1:]:
        d.add(elm.Line().endpoints(ends[0], e))
    return ends[0]


def render_circuit(slots: list[CircuitSlot], out_path: str) -> None:
    """Deterministic layout: slots placed left-to-right (parallel slots as
    branching groups), then the loop is closed back to the first slot's
    start point -- works regardless of how many slots there are."""
    d = schemdraw.Drawing(file=out_path)
    first = None
    for slot in slots:
        if slot[0] == "single":
            _, comp_type, label = slot
            el = d.add(CIRCUIT_TYPE_MAP[comp_type]().right().label(label))
            start_point = el.start
        else:
            _, group = slot
            start_point = d.here
            _draw_parallel_branch(d, group)
        if first is None:
            first = start_point
    d.add(elm.Line().down())
    d.add(elm.Line().left().tox(first[0]))
    d.add(elm.Line().up().toy(first[1]))
    d.save(out_path)


GRAPH_EDGE_RE = re.compile(
    r"EDGE:\s*(.+?)\s*(->|--)\s*(.+?)\s*(?::\s*(\S+))?\s*(?=NODE:|EDGE:|$)",
    re.IGNORECASE | re.DOTALL,
)


def parse_graph_dsl(text: str) -> tuple[list[str], list[tuple[str, str, str, bool]]]:
    """Returns (nodes, edges) where each edge is (a, b, weight_label, directed)."""
    nodes: list[str] = []
    seen = set()
    for m in NODE_RE.finditer(text):
        name = m.group(1).strip()
        if name and name not in seen:
            nodes.append(name)
            seen.add(name)

    edges: list[tuple[str, str, str, bool]] = []
    for m in GRAPH_EDGE_RE.finditer(text):
        a, arrow, b, weight = m.group(1).strip(), m.group(2), m.group(3).strip(), (m.group(4) or "")
        for name in (a, b):
            if name not in seen:
                nodes.append(name)
                seen.add(name)
        edges.append((a, b, weight, arrow == "->"))
    return nodes, edges


def render_graph(
    nodes: list[str], edges: list[tuple[str, str, str, bool]], out_path: str,
) -> None:
    G = nx.DiGraph() if any(directed for _, _, _, directed in edges) else nx.Graph()
    G.add_nodes_from(nodes)
    for a, b, weight, _ in edges:
        # spring_layout uses the 'weight' edge attribute as a numeric spring
        # force, so an empty/non-numeric string here breaks layout entirely
        # -- only set it when there's a real number, never the raw label.
        try:
            G.add_edge(a, b, weight=float(weight)) if weight else G.add_edge(a, b)
        except ValueError:
            G.add_edge(a, b)

    # Trees read far more clearly with the root at top and levels below it
    # than with a generic spring layout (which scatters them with crossing
    # edges and no sense of hierarchy) -- detect and use a BFS layout from
    # the first-declared node when the structure actually is a tree.
    if nx.is_tree(G):
        pos = nx.bfs_layout(G, start=nodes[0], align="horizontal")
        pos = {n: (x, -y) for n, (x, y) in pos.items()}
    else:
        pos = nx.spring_layout(G, seed=42, k=1.2)
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(
        G, pos, ax=ax, with_labels=True, node_color="#cfe8ff", node_size=1200,
        font_size=11, edgecolors="black", arrows=True, arrowsize=18,
    )
    weight_labels = {(a, b): w for a, b, w, _ in edges if w}
    if weight_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=weight_labels, ax=ax, font_size=10)
    autofix_text_overlaps(fig)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


BOX_RE = re.compile(r"BOX:\s*(yes|no)", re.IGNORECASE)


VECTOR_RE = re.compile(
    r"VECTOR:\s*(.+?)\s*\|\s*(-?[\d.]+)\s*\|\s*(-?[\d.]+)\s*(?=VECTOR:|BOX:|$)",
    re.IGNORECASE | re.DOTALL,
)


def parse_vector_dsl(text: str) -> tuple[bool, list[tuple[str, float, float]]]:
    box_m = BOX_RE.search(text)
    origin_box = bool(box_m and box_m.group(1).lower() == "yes")
    vectors = []
    for m in VECTOR_RE.finditer(text):
        label = m.group(1).strip()
        try:
            x, y = float(m.group(2)), float(m.group(3))
        except ValueError:
            continue
        if label:
            vectors.append((label, x, y))
    return origin_box, vectors


_VECTOR_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b")


def render_vectors(origin_box: bool, vectors: list[tuple[str, float, float]], out_path: str) -> None:
    """Deterministic auto-layout: every vector is a real arrow (shaft +
    arrowhead) from the origin to (x, y), with its label placed just beyond
    the tip, radially outward. Labels that would land on top of each other
    (near-parallel vectors) are nudged apart tangentially -- the model never
    controls label placement, so this always avoids overlap."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    if origin_box:
        ax.add_patch(Rectangle((-0.5, -0.5), 1.0, 1.0, facecolor="lightgray",
                                edgecolor="black", zorder=3))

    max_r = max(((x ** 2 + y ** 2) ** 0.5 for _, x, y in vectors), default=1.0) or 1.0
    placed: list[tuple[float, float]] = []
    for i, (label, x, y) in enumerate(vectors):
        color = _VECTOR_COLORS[i % len(_VECTOR_COLORS)]
        ax.add_patch(FancyArrowPatch(
            (0, 0), (x, y), arrowstyle="-|>", mutation_scale=18,
            color=color, linewidth=2, zorder=2,
        ))
        r = (x ** 2 + y ** 2) ** 0.5
        ux, uy = (x / r, y / r) if r > 1e-6 else (0.0, 1.0)
        lx, ly = x + ux * 0.45, y + uy * 0.45
        for _ in range(8):  # nudge away from any already-placed label that's too close
            collided = False
            for px, py in placed:
                if ((lx - px) ** 2 + (ly - py) ** 2) ** 0.5 < 0.55:
                    collided = True
                    lx += -uy * 0.3
                    ly += ux * 0.3
            if not collided:
                break
        placed.append((lx, ly))
        ax.text(lx, ly, label, color=color, ha="center", va="center", fontsize=10,
                zorder=4, bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                     edgecolor="none", alpha=0.85))

    pad = max_r * 0.35 + 0.8
    xs = [x for _, x, _ in vectors] + [p[0] for p in placed] + [0.0]
    ys = [y for _, _, y in vectors] + [p[1] for p in placed] + [0.0]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5, zorder=1)
    ax.axvline(0, color="gray", linewidth=0.5, zorder=1)
    ax.grid(True, alpha=0.3)
    autofix_text_overlaps(fig)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


VENN_FIELD_RE = re.compile(
    r"\b(SET_\w+|ONLY_\w+|OVERLAP)\s*:\s*(.*?)\s*"
    r"(?=\b(?:SET_\w+|ONLY_\w+|OVERLAP)\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)


def parse_venn_dsl(text: str) -> dict[str, str]:
    """Matches by field-name *prefix* (SET_/ONLY_/OVERLAP) and assigns by
    position rather than requiring the exact literal "SET_A"/"SET_B" tokens
    -- the model sometimes mirrors the request's own set names into the
    field name itself (e.g. "SET_P:"/"SET_Q:" for a request about "Set P
    and Set Q"), which would otherwise be perfectly good data lost to a
    literal-string mismatch."""
    fields = {"SET_A": "", "SET_B": "", "ONLY_A": "", "ONLY_B": "", "OVERLAP": ""}
    set_slots = ["SET_A", "SET_B"]
    only_slots = ["ONLY_A", "ONLY_B"]
    for m in VENN_FIELD_RE.finditer(text):
        key, value = m.group(1).upper(), m.group(2).strip()
        if key.startswith("SET_") and set_slots:
            fields[set_slots.pop(0)] = value
        elif key.startswith("ONLY_") and only_slots:
            fields[only_slots.pop(0)] = value
        elif key == "OVERLAP":
            fields["OVERLAP"] = value
    return fields


def render_venn(fields: dict[str, str], out_path: str) -> None:
    """Always exactly two circles with fixed, tested geometry -- the model
    only supplies the text for each region, so it can never invent extra
    shapes or end up with mismatched/duplicate borders."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    r, sep = 2.0, 1.4
    ax.add_patch(Circle((-sep, 0), r, alpha=0.35, facecolor="skyblue", edgecolor="black", linewidth=1.5))
    ax.add_patch(Circle((sep, 0), r, alpha=0.35, facecolor="lightgreen", edgecolor="black", linewidth=1.5))

    ax.text(-sep, r + 0.3, fields["SET_A"], ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.text(sep, r + 0.3, fields["SET_B"], ha="center", va="bottom", fontsize=13, fontweight="bold")

    def place(text: str, x: float, y: float, width: int) -> None:
        if text:
            ax.text(x, y, textwrap.fill(text, width=width), ha="center", va="center", fontsize=10)

    place(fields["ONLY_A"], -sep - 0.7, 0, 14)
    place(fields["ONLY_B"], sep + 0.7, 0, 14)
    place(fields["OVERLAP"], 0, 0, 10)

    ax.set_xlim(-sep - r - 0.8, sep + r + 0.8)
    ax.set_ylim(-r - 0.6, r + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    autofix_text_overlaps(fig)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


TABLE_TITLE_RE = re.compile(r"TITLE:\s*(.*?)\s*(?=COLUMNS:|$)", re.IGNORECASE | re.DOTALL)


TABLE_COLUMNS_RE = re.compile(r"COLUMNS:\s*(.+?)\s*(?=ROW:|$)", re.IGNORECASE | re.DOTALL)


TABLE_ROW_RE = re.compile(r"ROW:\s*(.+?)\s*(?=ROW:|$)", re.IGNORECASE | re.DOTALL)


def parse_table_dsl(text: str) -> tuple[str, list[str], list[list[str]]]:
    title_m = TABLE_TITLE_RE.search(text)
    title = title_m.group(1).strip() if title_m else ""
    columns_m = TABLE_COLUMNS_RE.search(text)
    columns = [c.strip() for c in columns_m.group(1).split("|")] if columns_m else []
    rows = []
    for m in TABLE_ROW_RE.finditer(text):
        cells = [c.strip() for c in m.group(1).split("|")]
        # Defensive: pad/truncate a malformed row to the column count rather
        # than failing the whole table over one ragged row.
        if columns:
            cells = (cells + [""] * len(columns))[:len(columns)]
        rows.append(cells)
    return title, columns, rows


def render_table(title: str, columns: list[str], rows: list[list[str]], out_path: str) -> None:
    """Always a real ax.table() grid with auto-scaled figure size and font
    size -- never a chart or floating text standing in for a table."""
    n_cols, n_rows = len(columns), len(rows)
    fig_w = max(4.0, min(12.0, 1.3 * n_cols))
    fig_h = max(1.5, min(10.0, 0.45 * (n_rows + 1) + 0.6))
    fontsize = max(7, min(12, 16 - max(n_cols, n_rows)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize + 3, pad=12)

    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.6)
    table.auto_set_column_width(col=list(range(n_cols)))

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        if r == 0:
            cell.set_facecolor("#cfe8ff")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white" if r % 2 else "#f5f5f5")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


SCENE_TITLE_RE = re.compile(r"TITLE:\s*(.*?)\s*(?=SHAPE:|$)", re.IGNORECASE | re.DOTALL)


SCENE_SHAPE_RE = re.compile(
    r"SHAPE:\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*(yes|no)\s*(?=SHAPE:|$)",
    re.IGNORECASE | re.DOTALL,
)


def parse_scene_dsl(text: str) -> tuple[str, list[tuple[str, str, int, bool]]]:
    title_m = SCENE_TITLE_RE.search(text)
    title = title_m.group(1).strip() if title_m else ""
    items = []
    for m in SCENE_SHAPE_RE.finditer(text):
        shape, color, count, overlap = m.group(1).lower(), m.group(2).lower(), int(m.group(3)), m.group(4).lower() == "yes"
        if shape in _SCENE_SHAPE_BUILDERS and count > 0:
            items.append((shape, color, count, overlap))
    return title, items


_SCENE_SHAPE_BUILDERS = {
    "circle": lambda center, size, color: Circle(center, size, facecolor=color, edgecolor="black", linewidth=1.2),
    "square": lambda center, size, color: Rectangle(
        (center[0] - size, center[1] - size), 2 * size, 2 * size,
        facecolor=color, edgecolor="black", linewidth=1.2,
    ),
    "triangle": lambda center, size, color: RegularPolygon(
        center, 3, radius=size * 1.3, facecolor=color, edgecolor="black", linewidth=1.2,
    ),
}


def render_scene(title: str, items: list[tuple[str, str, int, bool]], out_path: str, seed: int = 42) -> None:
    """Deterministic rejection-sampling placement: every requested shape is
    actually drawn (never silently dropped), shapes never overlap unless
    their group explicitly asked for it, and the title is set exactly once
    via ax.set_title (never duplicated)."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(6, 6))
    if title:
        ax.set_title(title, fontsize=13, pad=10)

    size = 0.6
    lo, hi = 1.5, 8.5
    placed: list[tuple[float, float, float]] = []
    for shape, color, count, overlap in items:
        group_anchor = None
        for _ in range(count):
            if overlap and group_anchor is not None:
                ax_, ay = group_anchor
                ang = rng.uniform(0, 2 * np.pi)
                dist = size * rng.uniform(0.4, 0.8)
                cx, cy = ax_ + dist * np.cos(ang), ay + dist * np.sin(ang)
            else:
                cx, cy = rng.uniform(lo, hi), rng.uniform(lo, hi)
                for _ in range(300):
                    if all((cx - px) ** 2 + (cy - py) ** 2 >= (size + ps) ** 2 * 0.9 for px, py, ps in placed):
                        break
                    cx, cy = rng.uniform(lo, hi), rng.uniform(lo, hi)
                if overlap and group_anchor is None:
                    group_anchor = (cx, cy)
            placed.append((cx, cy, size))
            ax.add_patch(_SCENE_SHAPE_BUILDERS[shape]((cx, cy), size, color))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


PUZZLE_LAYOUT_RE = re.compile(r"LAYOUT:\s*(\w+)", re.IGNORECASE)


PUZZLE_PANEL_RE = re.compile(
    r"PANEL:\s*(\w+)\s*\|\s*(\d+)\s*\|\s*(-?\d+)\s*(?:\|\s*(\w+)\s*)?(?=PANEL:|ANSWER:|ODD_INDEX:|TRANSFORM_DEGREES:|$)",
    re.IGNORECASE | re.DOTALL,
)


PUZZLE_ANSWER_RE = re.compile(
    r"ANSWER:\s*(\w+)\s*\|\s*(\d+)\s*\|\s*(-?\d+)\s*(?:\|\s*(\w+))?", re.IGNORECASE,
)


PUZZLE_ODD_INDEX_RE = re.compile(r"ODD_INDEX:\s*(\d+)", re.IGNORECASE)


PUZZLE_TRANSFORM_RE = re.compile(r"TRANSFORM_DEGREES:\s*(-?\d+)", re.IGNORECASE)


_PUZZLE_SHAPES = {"circle", "square", "triangle", "right_triangle"}


_PUZZLE_LAYOUTS = {"sequence", "symmetry", "rotation", "odd_one_out", "analogy", "row"}


def parse_puzzle_dsl(text: str) -> dict:
    layout_m = PUZZLE_LAYOUT_RE.search(text)
    layout = layout_m.group(1).lower() if layout_m else ""
    if layout not in _PUZZLE_LAYOUTS:
        layout = ""  # an unrecognized layout must trigger a repair retry, not a silent fallback render
    panels = []
    for m in PUZZLE_PANEL_RE.finditer(text):
        shape = m.group(1).lower()
        if shape in _PUZZLE_SHAPES:
            panels.append({
                "shape": shape, "count": int(m.group(2)), "rotation": int(m.group(3)),
                "color": (m.group(4) or "black").lower(),
            })
    answer_m = PUZZLE_ANSWER_RE.search(text)
    answer = None
    if answer_m and answer_m.group(1).lower() in _PUZZLE_SHAPES:
        answer = {
            "shape": answer_m.group(1).lower(), "count": int(answer_m.group(2)), "rotation": int(answer_m.group(3)),
            "color": (answer_m.group(4) or "black").lower(),
        }
    odd_index_m = PUZZLE_ODD_INDEX_RE.search(text)
    odd_index = int(odd_index_m.group(1)) - 1 if odd_index_m else None
    transform_m = PUZZLE_TRANSFORM_RE.search(text)
    transform_degrees = int(transform_m.group(1)) if transform_m else 90
    return {
        "layout": layout, "panels": panels, "answer": answer,
        "odd_index": odd_index, "transform_degrees": transform_degrees,
    }


def _right_triangle_patch(ax, center, half_leg, flip_x=False, rotation=0, color="black"):
    cx, cy = center
    sx = -1 if flip_x else 1
    pts = [(cx - sx * half_leg, cy - half_leg), (cx - sx * half_leg, cy + half_leg), (cx + sx * half_leg, cy - half_leg)]
    patch = Polygon(pts, closed=True, fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(patch)
    if rotation:
        patch.set_transform(Affine2D().rotate_deg_around(cx, cy, rotation) + ax.transData)
    return patch


_PUZZLE_SHAPE_BUILDERS = {
    "circle": lambda center, size, color: Circle(center, size, facecolor=color, edgecolor="black"),
    "square": lambda center, size, color: Rectangle((center[0] - size, center[1] - size), 2 * size, 2 * size, facecolor=color, edgecolor="black"),
    "triangle": lambda center, size, color: RegularPolygon(center, 3, radius=size * 1.3, facecolor=color, edgecolor="black"),
}


def _draw_puzzle_panel_content(ax, cx, cy, box_size, shape, count, rotation=0, flip_x=False, color="black"):
    """Draws `count` copies of `shape` inside a box of size `box_size`
    centered at (cx, cy). The multi-item grid layout is sized so the full
    grid (including each item's own radius) always stays within the box,
    regardless of count -- it never overflows into a neighboring panel."""
    if count <= 0:
        return
    if shape == "right_triangle":
        half_leg = box_size * 0.30  # safety margin: sqrt(2)*0.30 < 0.45 (90% of half-box) at any rotation
        _right_triangle_patch(ax, (cx, cy), half_leg, flip_x=flip_x, rotation=rotation, color=color)
        return
    cols = min(count, 3) if count != 4 else 2  # 4 items read as a clean 2x2 grid, not 3+1
    rows = (count + cols - 1) // cols
    usable = box_size * 0.65
    spacing = usable / cols if cols > 1 else 0
    item_size = min(box_size * 0.22, spacing * 0.38) if count > 1 else box_size * 0.26
    placed = 0
    for r in range(rows):
        n_this_row = min(cols, count - placed)
        row_w = (n_this_row - 1) * spacing
        for c in range(n_this_row):
            x = cx - row_w / 2 + c * spacing
            y = cy + (rows - 1 - r) * spacing - (rows - 1) * spacing / 2
            patch = _PUZZLE_SHAPE_BUILDERS[shape]((x, y), item_size, color)
            ax.add_patch(patch)
            if rotation:
                patch.set_transform(Affine2D().rotate_deg_around(x, y, rotation) + ax.transData)
            placed += 1


def render_puzzle(spec: dict, out_path: str) -> None:
    layout = spec["layout"]
    box, gap = 1.6, 0.5

    if layout in ("symmetry", "rotation"):
        panel = spec["panels"][0]
        big_box = 2.4
        big_gap = 1.0
        left_cx, right_cx = -big_box / 2 - big_gap / 2, big_box / 2 + big_gap / 2
        fig, ax = plt.subplots(figsize=(7, 4))
        for cx in (left_cx, right_cx):
            ax.add_patch(FancyBboxPatch((cx - big_box / 2, -big_box / 2), big_box, big_box,
                                         boxstyle="round,pad=0.02", facecolor="white", edgecolor="black", linewidth=1.5))
        panel_color = panel.get("color", "black")
        _draw_puzzle_panel_content(ax, left_cx, 0, big_box, panel["shape"], panel["count"], panel.get("rotation", 0), color=panel_color)
        if layout == "symmetry":
            ax.axvline(0, color="gray", linestyle="--", linewidth=1.5)
            _draw_puzzle_panel_content(ax, right_cx, 0, big_box, panel["shape"], panel["count"], panel.get("rotation", 0), flip_x=True, color=panel_color)
        else:
            _draw_puzzle_panel_content(ax, right_cx, 0, big_box, panel["shape"], panel["count"], spec["transform_degrees"], color=panel_color)
        ax.set_xlim(left_cx - big_box / 2 - 0.5, right_cx + big_box / 2 + 0.5)
        ax.set_ylim(-big_box / 2 - 0.5, big_box / 2 + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    # sequence / odd_one_out / analogy: a row of fixed-size panels, the
    # last one either the revealed answer or a "?" mark, with optional
    # separators (analogy) or a red highlight border (odd_one_out).
    panels = list(spec["panels"])
    if layout in ("sequence", "analogy"):
        panels.append(spec["answer"] if spec["answer"] is not None else {"marker": "question"})
    if layout == "odd_one_out" and spec["odd_index"] is not None and 0 <= spec["odd_index"] < len(panels):
        panels[spec["odd_index"]] = {**panels[spec["odd_index"]], "marker": "odd"}
    separators = [":", "::", ":"] if layout == "analogy" and len(panels) >= 4 else None

    n = len(panels)
    total_w = n * box + (n - 1) * gap
    fig, ax = plt.subplots(figsize=(min(14, 1.0 + total_w), 3.2))
    x0 = -total_w / 2 + box / 2
    centers = []
    for i, p in enumerate(panels):
        cx = x0 + i * (box + gap)
        centers.append(cx)
        edgecolor = "red" if p.get("marker") == "odd" else "black"
        lw = 2.5 if p.get("marker") == "odd" else 1.5
        ax.add_patch(FancyBboxPatch((cx - box / 2, -box / 2), box, box, boxstyle="round,pad=0.02",
                                     facecolor="white", edgecolor=edgecolor, linewidth=lw))
        if p.get("marker") == "question":
            ax.text(cx, 0, "?", ha="center", va="center", fontsize=40, fontweight="bold")
        else:
            _draw_puzzle_panel_content(ax, cx, 0, box, p["shape"], p["count"], p.get("rotation", 0), color=p.get("color", "black"))

    if separators:
        for i, sep in enumerate(separators[:n - 1]):
            mid_x = (centers[i] + centers[i + 1]) / 2
            ax.text(mid_x, 0, sep, ha="center", va="center", fontsize=18)

    ax.set_xlim(-total_w / 2 - 0.6, total_w / 2 + 0.6)
    ax.set_ylim(-box / 2 - 0.6, box / 2 + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    autofix_text_overlaps(fig)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


GEOMETRY_LINE_RE = re.compile(
    r"^[ \t]*[•\-]*\s*(POINT|ANGLE_POINT|REGULAR_POLYGON|RIGHT_TRIANGLE|TRAPEZOID|MIDPOINT|CIRCLE|"
    r"POLYGON|SEGMENT|DIAGONAL|MARK_POINT|RIGHT_ANGLE|EQUAL_GROUP|ANGLE_LABEL|SIDE_LABEL|RADIUS_LABEL|"
    r"SLOPE_LABEL|AREA_LABEL|CALLOUT|COORD_LABEL|GRID):\s*",
    re.IGNORECASE | re.MULTILINE,
)


_MARKDOWN_DECORATION_RE = re.compile(r"[`*]")

# POLYGON/AREA_LABEL are inherently "one entry, a list of names" -- when the
# model spreads that list across several markdown-bulleted lines under one
# header, all those lines join back into a single comma list. Every other
# tag is "one entry per line": the model sometimes lists several POINTs (or
# SIDE_LABELs, etc.) under a single shared header instead of repeating the
# tag, so each non-empty line becomes its own separate command of that tag.
_LIST_TAGS = {"POLYGON", "AREA_LABEL"}


def _looks_like_data_line(line: str) -> bool:
    """A real continuation record (another POINT/SIDE_LABEL/... under a
    shared header) is short and never ends in sentence punctuation. The
    model also likes to narrate its reasoning between tag lines (e.g. "...so
    we can use that point name:"), and without this check that commentary
    gets swept up as if it were another instance of the *previous* tag."""
    return bool(line) and line[-1] not in ".:?!;" and len(line.split()) <= 12


def parse_geometry_dsl(text: str) -> list[tuple[str, str]]:
    """Parse into an ordered list of (TAG, body) commands -- order matters: a
    SEGMENT/ANGLE_LABEL/etc. line can reference a point defined by an earlier
    POINT/ANGLE_POINT/REGULAR_POLYGON/RIGHT_TRIANGLE line, so commands are
    replayed in the order the model wrote them rather than grouped by type."""
    text = _MARKDOWN_DECORATION_RE.sub("", text)
    pieces = GEOMETRY_LINE_RE.split(text)[1:]
    commands = []
    for tag, body in zip(pieces[0::2], pieces[1::2]):
        tag = tag.upper()
        all_lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if not all_lines:
            continue
        # The first line is always this tag's primary value, no matter its
        # shape; only *additional* lines need to look like real data before
        # being accepted as further records (or, for list tags, folded in).
        lines = all_lines[:1] + [ln for ln in all_lines[1:] if _looks_like_data_line(ln)]
        if tag in _LIST_TAGS:
            # A real polygon/area-label name list is short (a handful of
            # vertices); a short prose phrase repeated dozens of times (a
            # repetition-collapse artifact) can slip past the line filter
            # above since it never ends in sentence punctuation, so cap how
            # many continuation lines get folded in rather than trusting an
            # unbounded join.
            commands.append((tag, ",".join(lines[:12])))
        else:
            commands.extend((tag, line) for line in lines)
    return commands


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _fmt_num(v: float) -> str:
    """Round to an int when it's close to one (the common case for
    constructed figures -- e.g. a pentagon's interior angle computed from
    its actual vertices comes back as 107.999999...), else one decimal --
    never the raw 53.130102354155598-style float a model previously left
    unrounded."""
    if abs(v - round(v)) < 0.05:
        return str(int(round(v)))
    return f"{v:.1f}"


def _polygon_area(pts: list[np.ndarray]) -> float:
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _angle_between(vertex: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float]:
    """Returns (start_degrees, sweep_degrees, mid_degrees) for the interior
    (<=180 deg) angle at `vertex` between rays to p1 and p2, oriented so a
    matplotlib Arc(theta1=start, theta2=start+sweep) always sweeps the
    correct (minor) arc regardless of which ray was listed first."""
    a1 = np.degrees(np.arctan2(*(p1 - vertex)[::-1])) % 360
    a2 = np.degrees(np.arctan2(*(p2 - vertex)[::-1])) % 360
    sweep = (a2 - a1) % 360
    if sweep > 180:
        a1, sweep = a2, 360 - sweep
    return a1, sweep, a1 + sweep / 2


def render_geometry(commands: list[tuple[str, str]], out_path: str) -> None:
    """Deterministic geometry renderer: the model only supplies points and
    references between them (see GEOMETRY_DSL_PROMPT); every angle, side
    length, area, and slope is computed here from the actual coordinates, so
    a model arithmetic mistake can produce a wrong picture (bad point
    choice) but never a wrong *label* on an otherwise-correct picture, and
    axis limits are always derived from the real geometry, never a
    model-guessed window that could crop it."""
    points: dict[str, np.ndarray] = {}
    circles: dict[str, tuple[float, float, float]] = {}
    aux_circles: set[str] = set()  # REGULAR_POLYGON's circumscribing circle: drawn lighter, not user-requested
    polygons: list[list[str]] = []
    segments: list[tuple[str, str]] = []
    marks: list[str] = []
    right_angles: list[tuple[str, str, str]] = []
    equal_groups: list[tuple[int, list[tuple[str, str]]]] = []
    angle_labels: list[tuple[str, str, str, str]] = []
    side_labels: list[tuple[str, str, str]] = []
    radius_labels: list[tuple[str, str]] = []
    slope_labels: list[tuple[str, str, str]] = []
    area_labels: list[list[str]] = []
    callouts: list[tuple[str, str]] = []
    coord_labels: list[str] = []
    show_grid = False

    def need(name: str) -> np.ndarray:
        if name not in points:
            raise ValueError(f"reference to undefined point '{name}'")
        return points[name]

    def origin_of(name: str) -> np.ndarray:
        # ANGLE_POINT's origin is usually a named POINT, but the model often
        # skips re-declaring a circle's center as its own POINT line when it
        # only needs it to place points on that circle -- accept the
        # circle's own name as an origin too, using its center. And when it
        # references some other invented name (e.g. a generic "O" that was
        # never defined at all) while exactly one circle exists, that's
        # almost always meant to be that circle's center too.
        if name in points:
            return points[name]
        if name in circles:
            cx, cy, _ = circles[name]
            return np.array([cx, cy])
        if len(circles) == 1:
            cx, cy, _ = next(iter(circles.values()))
            return np.array([cx, cy])
        raise ValueError(f"reference to undefined point '{name}'")

    # Point-defining commands always run before everything else, regardless
    # of where the model put them in its response -- it sometimes writes
    # POLYGON: A,B,C,D before the POINT: lines that define A-D, and there's
    # no reason to fail a response just because of line order rather than
    # content. A stable sort preserves relative order *within* each group, so
    # a chained ANGLE_POINT that depends on an earlier ANGLE_POINT still
    # resolves correctly.
    # POLYGON is included here (not just the obvious point-producers) so that
    # when it falls back to synthesizing default vertices (the model skipped
    # POINT: lines entirely), those points already exist if a later-in-text
    # but same-pass CIRCLE needs one for its own AUTO/POINT: radius.
    _POINT_DEFINING = {"POINT", "ANGLE_POINT", "REGULAR_POLYGON", "RIGHT_TRIANGLE", "TRAPEZOID", "MIDPOINT", "CIRCLE", "POLYGON"}
    commands = sorted(commands, key=lambda c: c[0] not in _POINT_DEFINING)

    for tag, body in commands:
        try:
            # "|" is the documented separator, but a markdown-formatted
            # response sometimes uses "," for every tag uniformly (the same
            # habit that spreads POLYGON across bulleted lines) -- fall back
            # to "," only when there's no "|" at all, never both at once.
            p = [x.strip() for x in body.split("|" if "|" in body else ",")]
            if tag == "POINT" and len(p) >= 3:
                points[p[0]] = np.array([float(p[1]), float(p[2])])
            elif tag == "ANGLE_POINT" and len(p) >= 4:
                origin = origin_of(p[1])
                ang, length = float(p[2]), float(p[3])
                points[p[0]] = origin + length * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
            elif tag == "REGULAR_POLYGON" and len(p) >= 5:
                n, cx, cy, r, prefix = int(p[0]), float(p[1]), float(p[2]), float(p[3]), p[4]
                names = []
                for i in range(n):
                    ang = 90 + i * 360.0 / n  # first vertex points straight up -- a natural default orientation
                    points[f"{prefix}{i + 1}"] = np.array([cx + r * np.cos(np.radians(ang)), cy + r * np.sin(np.radians(ang))])
                    names.append(f"{prefix}{i + 1}")
                polygons.append(names)
                circles[prefix] = (cx, cy, r)
                aux_circles.add(prefix)
            elif tag == "RIGHT_TRIANGLE" and len(p) >= 7:
                right_pt, leg1_pt, leg2_pt = p[0], p[1], p[2]
                ox, oy, leg1 = float(p[3]), float(p[4]), float(p[5])
                leg2_spec = p[6]
                if leg2_spec.upper().startswith("ANGLE:"):
                    leg2 = leg1 * np.tan(np.radians(float(leg2_spec.split(":", 1)[1])))
                else:
                    leg2 = float(leg2_spec)
                points[right_pt] = np.array([ox, oy])
                points[leg1_pt] = np.array([ox + leg1, oy])
                points[leg2_pt] = np.array([ox, oy + leg2])
                polygons.append([right_pt, leg1_pt, leg2_pt])
                right_angles.append((right_pt, leg1_pt, leg2_pt))
            elif tag == "TRAPEZOID" and len(p) >= 9:
                bl, br, tr, tl = p[0], p[1], p[2], p[3]
                ox, oy, bottom_len, top_len, height = (float(x) for x in p[4:9])
                points[bl] = np.array([ox, oy])
                points[br] = np.array([ox + bottom_len, oy])
                points[tr] = np.array([ox + bottom_len / 2 + top_len / 2, oy + height])
                points[tl] = np.array([ox + bottom_len / 2 - top_len / 2, oy + height])
                polygons.append([bl, br, tr, tl])
            elif tag == "CIRCLE" and len(p) >= 4:
                cx, cy = float(p[1]), float(p[2])
                r_spec = p[3].strip()
                center = np.array([cx, cy])
                if r_spec.upper().startswith("POINT:"):
                    # The radius is "the distance to this other point" --
                    # e.g. a circle through a square's corner -- computed
                    # here so the model never has to take a square root by
                    # hand (it reliably gets those wrong).
                    r = float(np.linalg.norm(need(r_spec.split(":", 1)[1].strip()) - center))
                elif r_spec.upper() == "AUTO":
                    # The model overgeneralizes the "use AUTO for anything
                    # computed" rule to CIRCLE's radius too, even though the
                    # documented way to say that is POINT:<name> -- default
                    # to the distance to whichever other point is closest,
                    # which is correct whenever (as is typical) this circle
                    # is meant to pass through some already-defined point.
                    others = [pt for pt in points.values() if np.linalg.norm(pt - center) > 1e-6]
                    if not others:
                        raise ValueError(f"CIRCLE '{p[0]}' has radius AUTO but no other point exists to measure to")
                    r = float(min(np.linalg.norm(pt - center) for pt in others))
                else:
                    r = float(r_spec)
                circles[p[0]] = (cx, cy, r)
            elif tag == "MIDPOINT" and len(p) >= 3:
                points[p[0]] = (need(p[1]) + need(p[2])) / 2
            elif tag == "POLYGON":
                # The model sometimes folds coordinates straight into the
                # POLYGON line itself instead of separate POINT: lines --
                # accept either shorthand it tends to invent: "A(0,0),B(4,0)"
                # or a flat "A,0,0,B,4,0" run of name,x,y triples.
                inline = re.findall(r"(\w+)\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)", body)
                raw_tokens = [t.strip() for t in body.split(",") if t.strip()]
                if inline:
                    names = []
                    for name, x, y in inline:
                        points[name] = np.array([float(x), float(y)])
                        names.append(name)
                elif (len(raw_tokens) >= 6 and len(raw_tokens) % 3 == 0
                        and all(_is_float(raw_tokens[i]) for i in range(len(raw_tokens)) if i % 3 != 0)
                        and not any(_is_float(raw_tokens[i]) for i in range(0, len(raw_tokens), 3))):
                    names = []
                    for i in range(0, len(raw_tokens), 3):
                        name, x, y = raw_tokens[i], raw_tokens[i + 1], raw_tokens[i + 2]
                        points[name] = np.array([float(x), float(y)])
                        names.append(name)
                else:
                    names = raw_tokens
                # A real point name is one token, never containing a space --
                # a stray prose fragment that slipped past the line-level
                # filter (e.g. "Right angle at diagonal intersection",
                # repeated) is unmistakable by this point and gets dropped
                # rather than treated as a pile of vertex names.
                names = [n for n in names if " " not in n]
                if len(names) >= 3:
                    if all(n not in points for n in names):
                        # The model skipped point definitions entirely (a
                        # real, recurring failure) -- synthesize a symmetric
                        # default polygon under the names it chose rather
                        # than failing outright. EQUAL_GROUP/ANGLE_LABEL
                        # results on a regular polygon are always
                        # geometrically consistent, which matches the common
                        # case of an under-specified isosceles/equilateral/
                        # rhombus/kite request.
                        n = len(names)
                        for i, name in enumerate(names):
                            ang = 90 + i * 360.0 / n
                            points[name] = np.array([3 * np.cos(np.radians(ang)), 3 * np.sin(np.radians(ang))])
                    polygons.append(names)
            elif tag in ("SEGMENT", "DIAGONAL") and len(p) >= 2:
                # A chord across a circle: the model sometimes jumps straight
                # to "SEGMENT: P | Q" without ever placing P/Q with
                # ANGLE_POINT first. When both ends are otherwise-undefined
                # and exactly one circle exists, treat it as a chord and
                # default its two endpoints onto that circle.
                if p[0] not in points and p[1] not in points and len(circles) == 1:
                    cx, cy, r = next(iter(circles.values()))
                    for name, ang in ((p[0], 35), (p[1], 145)):
                        points[name] = np.array([cx + r * np.cos(np.radians(ang)), cy + r * np.sin(np.radians(ang))])
                segments.append((p[0], p[1]))
            elif tag == "MARK_POINT":
                marks.append(body.strip())
            elif tag == "RIGHT_ANGLE" and len(p) >= 3:
                right_angles.append((p[0], p[1], p[2]))
            elif tag == "EQUAL_GROUP" and len(p) >= 2:
                pairs = []
                for tok in p[1].split(","):
                    if "-" in tok:
                        a, b = tok.split("-", 1)
                        pairs.append((a.strip(), b.strip()))
                if pairs:
                    equal_groups.append((int(p[0]), pairs))
            elif tag == "ANGLE_LABEL" and len(p) >= 4:
                angle_labels.append((p[0], p[1], p[2], p[3]))
            elif tag == "SIDE_LABEL" and len(p) >= 3:
                side_labels.append((p[0], p[1], p[2]))
            elif tag == "RADIUS_LABEL" and len(p) >= 2:
                radius_labels.append((p[0], p[1]))
            elif tag == "SLOPE_LABEL" and len(p) >= 3:
                slope_labels.append((p[0], p[1], p[2]))
            elif tag == "AREA_LABEL":
                names = [n.strip() for n in body.split(",") if n.strip()]
                if len(names) >= 3:
                    area_labels.append(names)
            elif tag == "CALLOUT" and len(p) >= 2:
                callouts.append((p[0], p[1]))
            elif tag == "COORD_LABEL":
                coord_labels.append(body.strip())
            elif tag == "GRID":
                show_grid = body.strip().lower().startswith("y")
        except (ValueError, IndexError):
            continue  # one malformed line (bad number, garbled token) should not sink the whole figure

    if not points and not circles:
        raise ValueError("no POINT/CIRCLE/REGULAR_POLYGON/RIGHT_TRIANGLE definitions found")
    if not polygons and not segments and not circles and not marks:
        # Points were defined but nothing actually visible references them
        # (e.g. every SEGMENT/POLYGON line was malformed and got dropped) --
        # that would render a blank canvas and silently report success,
        # which is worse than failing and letting the retry loop try again.
        raise ValueError("no polygon, segment, or circle ended up referencing any defined point")

    # A RADIUS_LABEL sometimes targets a name the model only ever declared
    # with POINT (e.g. "PLANET") and never actually drew as a CIRCLE, even
    # though asking for "its radius" only makes sense if it is one -- that's
    # clearly meant to be a small circle there, so make one rather than
    # failing outright on an otherwise-clear request.
    for circle_name, _ in radius_labels:
        if circle_name not in circles and circle_name in points:
            px, py = points[circle_name]
            extent_guess = max(
                (max((pt[0] for pt in points.values()), default=0) - min((pt[0] for pt in points.values()), default=0)),
                (max((pt[1] for pt in points.values()), default=0) - min((pt[1] for pt in points.values()), default=0)),
                1.0,
            )
            circles[circle_name] = (px, py, extent_guess * 0.06)
            aux_circles.add(circle_name)

    # Every size (right-angle square, tick marks, arc radius, label offset)
    # scales off the figure's own extent rather than a fixed constant -- a
    # hexagon of radius 4 and a triangle with legs 5/12 need very different
    # absolute marker sizes to look the same proportionally.
    xs = [pt[0] for pt in points.values()] + [cx - r for cx, cy, r in circles.values()] + [cx + r for cx, cy, r in circles.values()]
    ys = [pt[1] for pt in points.values()] + [cy - r for cx, cy, r in circles.values()] + [cy + r for cx, cy, r in circles.values()]
    extent = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    scale = extent if extent > 0 else 4.0

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for name, (cx, cy, r) in circles.items():
        is_aux = name in aux_circles
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor="gray" if is_aux else "black",
                             linewidth=1.0 if is_aux else 1.6, linestyle="--" if is_aux else "-"))

    for names in polygons:
        ax.add_patch(Polygon([need(n) for n in names], closed=True, fill=False, edgecolor="black", linewidth=1.8))

    for p1, p2 in segments:
        a, b = need(p1), need(p2)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", linewidth=1.6)

    mark_off = scale * 0.07
    for name in marks:
        pt = need(name)
        ax.plot(pt[0], pt[1], "o", color="black", markersize=5, zorder=5)
        ax.text(pt[0] + mark_off, pt[1] + mark_off, name, fontsize=10, ha="left", va="bottom")

    for name in coord_labels:
        pt = need(name)
        label = f"({_fmt_num(pt[0])}, {_fmt_num(pt[1])})"
        ax.text(pt[0] + mark_off, pt[1] - mark_off, label, fontsize=10, ha="left", va="top")

    ra_size = scale * 0.09
    for vertex, p1, p2 in right_angles:
        v, a, b = need(vertex), need(p1), need(p2)
        u1, u2 = (a - v) / np.linalg.norm(a - v), (b - v) / np.linalg.norm(b - v)
        pts = [v, v + ra_size * u1, v + ra_size * u1 + ra_size * u2, v + ra_size * u2]
        ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor="black", linewidth=1.2))

    tick_size, tick_gap = scale * 0.045, scale * 0.025
    for count, pairs in equal_groups:
        for p1, p2 in pairs:
            a, b = need(p1), need(p2)
            mid = (a + b) / 2
            along = (b - a) / np.linalg.norm(b - a)
            perp = np.array([-along[1], along[0]])
            for off in (np.arange(count) - (count - 1) / 2) * tick_gap:
                c = mid + along * off
                ax.plot([c[0] - perp[0] * tick_size, c[0] + perp[0] * tick_size],
                        [c[1] - perp[1] * tick_size, c[1] + perp[1] * tick_size], color="black", linewidth=1.5)

    arc_r = scale * 0.13
    # An angle arc with no visible rays connecting its vertex to the two
    # points it's between reads as a floating, disconnected mark (the
    # vertex is often a circle's center, not a polygon corner, so there's no
    # other line that happens to anchor it). Draw a thin ray for any side
    # the model didn't already cover with its own POLYGON/SEGMENT.
    drawn_pairs = {frozenset((s1, s2)) for s1, s2 in segments}
    for names in polygons:
        drawn_pairs.update(frozenset((names[i], names[(i + 1) % len(names)])) for i in range(len(names)))
    for vertex, p1, p2, text in angle_labels:
        v, a, b = need(vertex), need(p1), need(p2)
        for other, pt in ((p1, a), (p2, b)):
            if frozenset((vertex, other)) not in drawn_pairs:
                ax.plot([v[0], pt[0]], [v[1], pt[1]], color="black", linewidth=1.0)
                drawn_pairs.add(frozenset((vertex, other)))
        start, sweep, mid = _angle_between(v, a, b)
        ax.add_patch(Arc(v, 2 * arc_r, 2 * arc_r, angle=0, theta1=start, theta2=start + sweep, color="black", linewidth=1.2))
        label = f"{_fmt_num(sweep)}°" if text.strip().upper() == "AUTO" else text
        mid_rad = np.radians(mid)
        ax.text(v[0] + arc_r * 1.5 * np.cos(mid_rad), v[1] + arc_r * 1.5 * np.sin(mid_rad), label,
                ha="center", va="center", fontsize=10)

    side_off = scale * 0.05
    for p1, p2, text in side_labels:
        a, b = need(p1), need(p2)
        # A label for a side that was never actually drawn (e.g. the model
        # listed all three SIDE_LABELs for a triangle but forgot the
        # POLYGON/SEGMENT line itself) would float with nothing to anchor
        # it -- same fix as the angle rays above.
        if frozenset((p1, p2)) not in drawn_pairs:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="black", linewidth=1.6)
            drawn_pairs.add(frozenset((p1, p2)))
        length = np.linalg.norm(b - a)
        label = _fmt_num(length) if text.strip().upper() == "AUTO" else text
        mid = (a + b) / 2
        along = (b - a) / length
        perp = np.array([-along[1], along[0]])
        ax.text(*(mid + perp * side_off), label, ha="center", va="center", fontsize=10)

    # 45/225 first so two concentric-circle radius labels fall on one clean
    # radial line (rather than visibly bending) before less-aligned angles
    # take over for any additional circles.
    _radius_angles = [45, 225, 135, 315, 100, 280]
    for i, (circle_name, text) in enumerate(radius_labels):
        if circle_name not in circles:
            raise ValueError(f"RADIUS_LABEL references undefined circle '{circle_name}'")
        cx, cy, r = circles[circle_name]
        ang = np.radians(_radius_angles[i % len(_radius_angles)])
        edge = np.array([cx + r * np.cos(ang), cy + r * np.sin(ang)])
        center = np.array([cx, cy])
        ax.plot([cx, edge[0]], [cy, edge[1]], color="black", linewidth=1.0, linestyle=":")
        label = f"r = {_fmt_num(r)}" if text.strip().upper() == "AUTO" else text
        mid = center + (edge - center) * 0.55
        perp = np.array([-(edge - center)[1], (edge - center)[0]])
        perp = perp / (np.linalg.norm(perp) or 1.0)
        ax.text(*(mid + perp * side_off), label, ha="center", va="center", fontsize=10)

    for p1, p2, text in slope_labels:
        a, b = need(p1), need(p2)
        label = text
        if text.strip().upper() == "AUTO":
            label = "slope undefined" if abs(b[0] - a[0]) < 1e-9 else f"slope = {_fmt_num((b[1] - a[1]) / (b[0] - a[0]))}"
        mid = (a + b) / 2
        ax.text(mid[0] + side_off, mid[1] + side_off, label, ha="left", va="bottom", fontsize=10)

    for names in area_labels:
        pts = [need(n) for n in names]
        centroid = np.mean(pts, axis=0)
        ax.text(*centroid, f"Area = {_fmt_num(_polygon_area(pts))}", ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))

    if callouts:
        all_pts = list(points.values()) or [np.array([cx, cy]) for cx, cy, r in circles.values()]
        centroid = np.mean(all_pts, axis=0)
        for target, text in callouts:
            t = need(target)
            direction = t - centroid
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-6 else np.array([0.0, 1.0])
            label_pos = t + direction * scale * 0.45
            ax.annotate(text, xy=t, xytext=label_pos, fontsize=10, color="red", ha="center", va="center",
                        arrowprops=dict(arrowstyle="-|>", color="red", linewidth=1.5))

    # Axis limits always come from the actual rendered geometry (points,
    # circles, and how far callout labels reach) -- never a model-guessed
    # window, which is what cropped circles and clipped shapes in the old
    # free-form-code approach.
    all_x = list(xs)
    all_y = list(ys)
    for target, _ in callouts:
        t = need(target)
        all_x.append(t[0] + scale * 0.5)
        all_y.append(t[1] + scale * 0.5)
    margin = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1.0) * 0.18
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal")
    if show_grid:
        # A real coordinate plane (numbered axes + light grid) -- requested
        # explicitly via GRID: yes, not the default, since most geometry
        # figures (a plain triangle/circle) read better with no axes at all.
        ax.grid(True, color="#dddddd", linewidth=0.7, zorder=0)
        ax.axhline(0, color="black", linewidth=1.0, zorder=1)
        ax.axvline(0, color="black", linewidth=1.0, zorder=1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.axis("off")
    autofix_text_overlaps(fig)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
