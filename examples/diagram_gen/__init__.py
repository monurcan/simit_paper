"""Natural images, diagrams, plots, geometric figures, molecules, circuits,
and graphs via BAGEL.

BAGEL's native image head is for natural images -- it can't draw a flowchart,
a labeled triangle, a circuit, or a molecule. So requests for real-world
scenes go straight to that image head (NATURAL), while everything else uses
BAGEL purely as a *text* model, rendering its output with a real tool. Which
tool depends on the category, chosen for the same reason throughout: give the
model the narrowest, most-familiar task possible and let deterministic code
(or a purpose-built library) handle the rest --

  - natural photos/illustrations -> BAGEL's own image-generation head, no
    text extraction or renderer involved -- generating the pixels is what
    that head is actually trained for.
  - diagrams / graph theory -> model extracts a node/edge list (a narrow
    extraction task); a deterministic layout function or networkx draws it.
  - molecules -> model writes a SMILES string (extremely well-represented in
    LLM training data); RDKit renders the actual structure.
  - circuits -> model extracts a component list; schemdraw draws it (its
    direction-chaining API is unfamiliar enough to a small model that having
    it write schemdraw code directly is unreliable).
  - vectors / free body diagrams -> model extracts a label + (x, y) component
    list (computing components from word directions like "down" or "at 30
    degrees" is a narrow, familiar task); a deterministic renderer draws the
    arrows and places labels with automatic overlap avoidance.
  - Venn diagrams / tables / scenes / IQ-style puzzles -> model extracts a
    small structured spec (set contents, table rows, shape groups, or panel
    layout); a deterministic renderer always draws it the same well-tested
    way, never letting the model invent its own layout code.
  - geometry -> model extracts points and references between them (never
    coordinates it would have to compute by hand -- trig, centering, etc. are
    all the renderer's job); a deterministic renderer draws shapes and
    computes every angle/length/area/slope from the actual coordinates.
  - plots/charts -> model writes matplotlib code directly, since that's the
    one kind of content it has seen enormous amounts of in pretraining.

Package layout:
  - prompts.py       all model-facing prompt text (one constant per pipeline)
  - sandbox.py        safety checks + isolated subprocess execution for
                       model-written code (FIGURE/GEOMETRY)
  - helpers.py        deterministic geometry-notation helpers and automatic
                       text-overlap avoidance, shared across renderers
  - renderers.py      one DSL parser + deterministic renderer per category
  - pipeline.py       generate_X() functions (extract -> retry/repair ->
                       render) and the classifier that picks which one to run
  - verify.py         optional photo-quality verification loop for NATURAL
                       output (on by default; see module docstring for why
                       it's NATURAL-only)
  - test_prompts.py   the benchmark suite (TEST_PROMPTS)
  - cli.py            the test-harness entry point (main())

Start a server first, e.g.:  mstar serve bagel

    python examples/diagram_generation.py
"""

from .pipeline import generate_visualization

__all__ = ["generate_visualization"]
