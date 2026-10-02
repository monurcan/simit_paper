"""Per-category generate_X() functions (DSL extraction + retry/repair loop
+ deterministic render) and the classifier that picks which one to run.

Ported from examples/diagram_gen/pipeline.py (BAGEL/mstar pipeline), adapted
to the model-agnostic UMMClient interface (common/model_clients/base.py)
instead of MStarClient. Renderer/prompt logic is unchanged."""

import base64
import io
import math
import re
import shutil
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw as RDDraw

from common.model_clients.base import UMMClient

from .prompts import (
    GEOMETRY_DSL_PROMPT, GEOMETRY_REPAIR_PROMPT, MASTER_PROMPT, DIAGRAM_DSL_PROMPT, DSL_REPAIR_PROMPT,
    CLASSIFY_PROMPT, REPAIR_PROMPT, MOLECULE_PROMPT, MOLECULE_REPAIR_PROMPT, CIRCUIT_DSL_PROMPT,
    CIRCUIT_REPAIR_PROMPT, GRAPH_DSL_PROMPT, GRAPH_REPAIR_PROMPT, VECTOR_DSL_PROMPT, VECTOR_REPAIR_PROMPT,
    VENN_DSL_PROMPT, VENN_REPAIR_PROMPT, TABLE_DSL_PROMPT, TABLE_REPAIR_PROMPT, SCENE_DSL_PROMPT,
    SCENE_REPAIR_PROMPT, PUZZLE_DSL_PROMPT, PUZZLE_REPAIR_PROMPT,
    SVG_PROMPT, SVG_REPAIR_PROMPT, MERMAID_PROMPT, MERMAID_REPAIR_PROMPT,
    VEGALITE_PROMPT, VEGALITE_REPAIR_PROMPT,
    HTML_PROMPT, HTML_REPAIR_PROMPT,
    HTML_COMPOSITE_PROMPT, HTML_COMPOSITE_REPAIR_PROMPT,
)
from .renderers import (
    parse_dsl, render_flowchart, parse_molecule, lookup_common_molecule, parse_circuit_dsl,
    render_circuit, parse_graph_dsl, render_graph, parse_vector_dsl, render_vectors, parse_venn_dsl,
    render_venn, parse_table_dsl, render_table, parse_scene_dsl, render_scene, parse_puzzle_dsl,
    render_puzzle, parse_geometry_dsl, render_geometry,
)
from .renderers_web import extract_fenced_block, render_svg, render_mermaid, render_vegalite, render_html
from .sandbox import (
    CJK_RE, clean_response, extract_code, is_safe, find_non_ascii_issue, find_known_api_issues, run_script,
)
from .verify import generate_with_verification

def generate_natural(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    verify_strategy: str = "informed", width: int | None = None, height: int | None = None,
    think_mode: bool = True, confidence_threshold: int | None = None,
    discard_unverified: bool = False, criterion: str = "verdict",
    verify_think_mode: bool | None = None, verify_max_attempts: int = 5,
) -> bool:
    """Realistic photos/illustrations go straight to the model's own image-
    generation head -- no DSL, no renderer, no intermediate extraction step.
    That whole narrow-DSL-plus-deterministic-renderer architecture exists
    specifically to compensate for the text head being unreliable at
    free-form layout code; none of that applies here, since generating the
    pixels *is* the thing the image head is actually trained for.

    verify_strategy ("none" / "plain" / "informed", default "informed"):
    NATURAL output benefits from a post-hoc photo-quality check since the
    image head can still miscount/mislabel/misplace things; see verify.py for
    why this is NATURAL-only.

    width/height: pixel size of the generated image.

    verify_think_mode (defaults to think_mode when None): lets the critic/
    revise calls think independently of the generation calls (see BAGEL's
    original pipeline for the A/B test motivating this split)."""

    def _generate_once(prompt: str) -> bool:
        for attempt in range(retries + 1):
            try:
                png_bytes = client.generate_image(prompt, think_mode=think_mode, width=width, height=height)
            except Exception as e:
                print(f"  [attempt {attempt}] image generation error: {e}")
                # NOTE: the original BAGEL/mstar pipeline wrapped this call in
                # a short-timeout client, because a stalled *HTTP server*
                # keeps processing the abandoned request independently --
                # timing out and retrying elsewhere was safe and let the
                # caller's watchdog restart the server. That doesn't have an
                # equivalent here: Omni runs in-process, so there is no
                # separate server to abandon a slow call to -- "timing out" a
                # synchronous in-process GPU call would mean forcibly killing
                # the thread/process actually doing the CUDA work, which is
                # its own can of worms (and per smoke-test evidence, a crash
                # already puts the whole Omni instance into an unrecoverable
                # ShutDown state -- a timeout wouldn't help there either,
                # since the failure is instantaneous, not a hang). Not
                # porting a fake timeout; retrying in place is the honest
                # behavior until real crash-recovery (re-instantiating
                # SensenovaClient) is worth building, which we're deferring
                # until past interactive small-scale verification.
                continue
            Path(out_path).write_bytes(png_bytes)
            print(f"  [attempt {attempt}] success -> {out_path}")
            return True
        print(f"  gave up after {retries + 1} attempts")
        return False

    return generate_with_verification(
        client, request, out_path, _generate_once, strategy=verify_strategy,
        think_mode=verify_think_mode if verify_think_mode is not None else think_mode,
        confidence_threshold=confidence_threshold,
        discard_unverified=discard_unverified, criterion=criterion,
        max_attempts=verify_max_attempts,
    )


def generate_figure(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model to write a matplotlib script for `request`, render it, save to out_path."""
    prompt = MASTER_PROMPT.format(request=request)
    code = None
    for attempt in range(retries + 1):
        # Greedy decoding (temperature=0) is great for the first try, but if
        # the model gets stuck on a broken pattern, retrying at temperature=0
        # with the same effective context just reproduces the identical
        # broken output -- seen verbatim, character-for-character, in
        # testing. Repair attempts need randomness to actually escape it.
        temperature = 0.0 if attempt == 0 else 0.5
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=1536, temperature=temperature)
        code = extract_code(clean_response(response.text))
        if code is None:
            print(f"  [attempt {attempt}] no code block found in response")
            prompt = MASTER_PROMPT.format(request=request)
            continue
        if not is_safe(code):
            print(f"  [attempt {attempt}] generated code failed the safety check, refusing to run it")
            return False

        code_path = Path(out_path).with_suffix(f".attempt{attempt}.py")
        code_path.write_text(code)

        known_issue = find_non_ascii_issue(code) or find_known_api_issues(code)
        if known_issue:
            print(f"  [attempt {attempt}] {known_issue}")
            prompt = REPAIR_PROMPT.format(request=request, code=code, error=known_issue)
            continue

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            ok, error = run_script(code, workdir)
            if ok:
                # shutil.move (not Path.rename): out_path may be on a
                # different filesystem than the tempdir (e.g. out-root on
                # a network filesystem vs the system tempdir), and rename()
                # raises "Invalid cross-device link" in that case.
                shutil.move(str(workdir / "output.png"), out_path)
                code_path.rename(Path(out_path).with_suffix(".py"))
                print(f"  [attempt {attempt}] success -> {out_path}")
                return True
            print(f"  [attempt {attempt}] render failed: {error.splitlines()[-1] if error else 'unknown error'}")
            prompt = REPAIR_PROMPT.format(request=request, code=code, error=error)

    print(f"  gave up after {retries + 1} attempts")
    return False


def generate_geometry(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model only to extract points and references between them (see
    GEOMETRY_DSL_PROMPT); render with the deterministic renderer, which
    computes every angle/length/area/slope from the actual coordinates
    instead of trusting a model-written number. Geometry was the last
    free-form-code category (alongside FIGURE) -- repeated prompt patches on
    the old matplotlib-generation approach kept fixing one failure mode
    (LaTeX delimiters, axis(off), missing circles) only for a new one to
    surface elsewhere (unrounded/wrong computed angles, mispositioned arcs,
    cropped circles), so it was converted to the same narrow-DSL-extraction
    pattern already proven for DIAGRAM/GRAPH/VECTOR/VENN/TABLE/SCENE/PUZZLE."""
    prompt = GEOMETRY_DSL_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        # Same reasoning as generate_figure(): a repair retry at temperature=0
        # with near-identical context just reproduces the same broken output
        # verbatim, so retries after the first need randomness to escape it.
        temperature = 0.0 if attempt == 0 else 0.5
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=768, temperature=temperature)
        cleaned = clean_response(response.text)
        Path(out_path).with_suffix(f".attempt{attempt}.dsl.txt").write_text(cleaned)

        # Corruption sometimes hits only the tail of an otherwise-correct
        # response (e.g. every line fine except the last ANGLE_LABEL) --
        # dropping just that line and rendering whatever's left beats
        # discarding a mostly-good response, since regenerating from scratch
        # risks the retry fumbling a *different* part of the answer instead
        # (observed empirically: retrying on every single dropped line made
        # the overall success rate worse, not better). Only retry when
        # corruption is severe enough that little would survive anyway.
        raw_lines = cleaned.splitlines()
        usable_lines = [line for line in raw_lines if not CJK_RE.search(line)]
        dropped = len(raw_lines) - len(usable_lines)
        if dropped:
            print(f"  [attempt {attempt}] dropped {dropped} non-ASCII line(s)")
            if attempt < retries and dropped > len(raw_lines) / 3:
                prompt = GEOMETRY_REPAIR_PROMPT.format(
                    request=request, previous=cleaned, reason="your answer contained non-English characters",
                )
                continue
        cleaned = "\n".join(usable_lines)

        commands = parse_geometry_dsl(cleaned)
        if not commands:
            print(f"  [attempt {attempt}] no geometry structure lines found in response")
            prompt = GEOMETRY_REPAIR_PROMPT.format(
                request=request, previous=cleaned, reason="no structure lines found",
            )
            continue
        try:
            render_geometry(commands, out_path)
        except Exception as e:
            print(f"  [attempt {attempt}] render error: {e}")
            prompt = GEOMETRY_REPAIR_PROMPT.format(request=request, previous=cleaned, reason=str(e))
            continue
        print(f"  [attempt {attempt}] success -> {out_path} ({len(commands)} commands)")
        return True

    print(f"  gave up after {retries + 1} attempts")
    return False


def generate_diagram(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model only to extract a node/edge list for `request`; render it
    with the deterministic auto-layout renderer (no LLM-written layout code)."""
    prompt = DIAGRAM_DSL_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=512, temperature=0.0)
        cleaned = clean_response(response.text)
        Path(out_path).with_suffix(f".attempt{attempt}.dsl.txt").write_text(cleaned)
        nodes, edges = parse_dsl(cleaned)
        if not nodes:
            print(f"  [attempt {attempt}] no NODE:/EDGE: lines found in response")
            prompt = DSL_REPAIR_PROMPT.format(request=request, reason="no NODE: lines found")
            continue
        try:
            render_flowchart(nodes, edges, out_path)
        except Exception as e:  # defensive: the renderer is deterministic but inputs vary
            print(f"  [attempt {attempt}] render error: {e}")
            prompt = DSL_REPAIR_PROMPT.format(request=request, reason=str(e))
            continue
        print(f"  [attempt {attempt}] success -> {out_path} ({len(nodes)} nodes, {len(edges)} edges)")
        return True

    print(f"  gave up after {retries + 1} attempts")
    return False


def generate_molecule(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Render `request`'s molecule with RDKit. Checks a small table of common
    molecules first (the model's chemistry knowledge proved unreliable even
    for things like ethanol/benzene/glucose); falls back to asking the model for
    a SMILES string only for molecules not in that table."""
    looked_up = lookup_common_molecule(request)
    if looked_up:
        smiles, title = looked_up
        mol = Chem.MolFromSmiles(smiles)
        RDDraw.MolToFile(mol, out_path, size=(450, 450), legend=title)
        print(f"  looked up '{title}' -> {out_path} (SMILES: {smiles})")
        return True

    prompt = MOLECULE_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=128, temperature=0.0)
        smiles, title = parse_molecule(clean_response(response.text))
        if not smiles:
            print(f"  [attempt {attempt}] no SMILES: line found in response")
            prompt = MOLECULE_REPAIR_PROMPT.format(request=request, smiles="(none)", reason="missing")
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [attempt {attempt}] invalid SMILES: {smiles!r}")
            prompt = MOLECULE_REPAIR_PROMPT.format(
                request=request, smiles=smiles, reason="RDKit could not parse it"
            )
            continue
        RDDraw.MolToFile(mol, out_path, size=(450, 450), legend=title)
        print(f"  [attempt {attempt}] success -> {out_path} (SMILES: {smiles})")
        return True

    print(f"  gave up after {retries + 1} attempts")
    return False


def generate_circuit(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model only to extract a component list for `request`; render it
    with the deterministic series/parallel layout (no LLM-written schemdraw code)."""
    prompt = CIRCUIT_DSL_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=256, temperature=0.0)
        cleaned = clean_response(response.text)
        slots = parse_circuit_dsl(cleaned)
        if not slots:
            print(f"  [attempt {attempt}] no COMPONENT:/PARALLEL: lines found in response")
            prompt = CIRCUIT_REPAIR_PROMPT.format(request=request, reason="no COMPONENT: lines found")
            continue
        try:
            render_circuit(slots, out_path)
        except Exception as e:
            print(f"  [attempt {attempt}] render error: {e}")
            prompt = CIRCUIT_REPAIR_PROMPT.format(request=request, reason=str(e))
            continue
        print(f"  [attempt {attempt}] success -> {out_path} ({len(slots)} slots)")
        return True

    print(f"  gave up after {retries + 1} attempts")
    return False


def generate_graph(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model only to extract a node/edge list for `request`; render it
    with networkx's spring layout (no LLM-written layout code)."""
    prompt = GRAPH_DSL_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=512, temperature=0.0)
        cleaned = clean_response(response.text)
        nodes, edges = parse_graph_dsl(cleaned)
        if not nodes:
            print(f"  [attempt {attempt}] no NODE:/EDGE: lines found in response")
            prompt = GRAPH_REPAIR_PROMPT.format(request=request, reason="no NODE: lines found")
            continue
        try:
            render_graph(nodes, edges, out_path)
        except Exception as e:
            print(f"  [attempt {attempt}] render error: {e}")
            prompt = GRAPH_REPAIR_PROMPT.format(request=request, reason=str(e))
            continue
        print(f"  [attempt {attempt}] success -> {out_path} ({len(nodes)} nodes, {len(edges)} edges)")
        return True

    print(f"  gave up after {retries + 1} attempts")
    return False


def generate_vectors(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model only to extract vector labels and (x, y) components; render
    with the deterministic arrow renderer (no LLM-written layout code)."""
    prompt = VECTOR_DSL_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=512, temperature=0.0)
        cleaned = clean_response(response.text)
        Path(out_path).with_suffix(f".attempt{attempt}.dsl.txt").write_text(cleaned)
        origin_box, vectors = parse_vector_dsl(cleaned)
        if not vectors:
            print(f"  [attempt {attempt}] no VECTOR: lines found in response")
            prompt = VECTOR_REPAIR_PROMPT.format(request=request, reason="no VECTOR: lines found")
            continue
        try:
            render_vectors(origin_box, vectors, out_path)
        except Exception as e:
            print(f"  [attempt {attempt}] render error: {e}")
            prompt = VECTOR_REPAIR_PROMPT.format(request=request, reason=str(e))
            continue
        print(f"  [attempt {attempt}] success -> {out_path} ({len(vectors)} vectors)")
        return True

    print(f"  gave up after {retries + 1} attempts")
    return False


def generate_venn(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model only to extract set names and region contents; render with
    the deterministic two-circle renderer (no LLM-written layout code)."""
    prompt = VENN_DSL_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=512, temperature=0.0)
        cleaned = clean_response(response.text)
        Path(out_path).with_suffix(f".attempt{attempt}.dsl.txt").write_text(cleaned)
        fields = parse_venn_dsl(cleaned)
        if not fields["SET_A"] or not fields["SET_B"]:
            print(f"  [attempt {attempt}] missing SET_A/SET_B in response")
            prompt = VENN_REPAIR_PROMPT.format(request=request, reason="missing SET_A or SET_B")
            continue
        try:
            render_venn(fields, out_path)
        except Exception as e:
            print(f"  [attempt {attempt}] render error: {e}")
            prompt = VENN_REPAIR_PROMPT.format(request=request, reason=str(e))
            continue
        print(f"  [attempt {attempt}] success -> {out_path} ({fields['SET_A']!r} / {fields['SET_B']!r})")
        return True

    print(f"  gave up after {retries + 1} attempts")
    return False


def generate_table(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model only to extract title/columns/rows; render with the
    deterministic ax.table() renderer (no LLM-written layout code)."""
    prompt = TABLE_DSL_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=512, temperature=0.0)
        cleaned = clean_response(response.text)
        Path(out_path).with_suffix(f".attempt{attempt}.dsl.txt").write_text(cleaned)
        title, columns, rows = parse_table_dsl(cleaned)
        if not columns or not rows:
            print(f"  [attempt {attempt}] missing COLUMNS or ROW lines in response")
            prompt = TABLE_REPAIR_PROMPT.format(request=request, reason="missing COLUMNS or ROW lines")
            continue
        try:
            render_table(title, columns, rows, out_path)
        except Exception as e:
            print(f"  [attempt {attempt}] render error: {e}")
            prompt = TABLE_REPAIR_PROMPT.format(request=request, reason=str(e))
            continue
        print(f"  [attempt {attempt}] success -> {out_path} ({len(columns)} cols, {len(rows)} rows)")
        return True

    print(f"  gave up after {retries + 1} attempts")
    return False


def generate_scene(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model only to extract shape groups; render with the deterministic
    rejection-sampling placement (no LLM-written layout code)."""
    prompt = SCENE_DSL_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=512, temperature=0.0)
        cleaned = clean_response(response.text)
        Path(out_path).with_suffix(f".attempt{attempt}.dsl.txt").write_text(cleaned)
        title, items = parse_scene_dsl(cleaned)
        if not items:
            print(f"  [attempt {attempt}] no SHAPE: lines found in response")
            prompt = SCENE_REPAIR_PROMPT.format(request=request, reason="no SHAPE: lines found")
            continue
        try:
            render_scene(title, items, out_path)
        except Exception as e:
            print(f"  [attempt {attempt}] render error: {e}")
            prompt = SCENE_REPAIR_PROMPT.format(request=request, reason=str(e))
            continue
        total = sum(c for _, _, c, _ in items)
        print(f"  [attempt {attempt}] success -> {out_path} ({len(items)} groups, {total} shapes)")
        return True

    print(f"  gave up after {retries + 1} attempts")
    return False


def generate_puzzle(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model only to extract the puzzle's layout and panel specs;
    render with the deterministic panel-grid renderer (no LLM-written
    layout code, so panels never overlap and mirror/rotation geometry is
    always exact)."""
    prompt = PUZZLE_DSL_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=512, temperature=0.0)
        cleaned = clean_response(response.text)
        Path(out_path).with_suffix(f".attempt{attempt}.dsl.txt").write_text(cleaned)
        spec = parse_puzzle_dsl(cleaned)
        if not spec["layout"] or not spec["panels"]:
            print(f"  [attempt {attempt}] missing LAYOUT or PANEL lines in response")
            prompt = PUZZLE_REPAIR_PROMPT.format(request=request, reason="missing LAYOUT or PANEL lines")
            continue
        try:
            render_puzzle(spec, out_path)
        except Exception as e:
            print(f"  [attempt {attempt}] render error: {e}")
            prompt = PUZZLE_REPAIR_PROMPT.format(request=request, reason=str(e))
            continue
        print(f"  [attempt {attempt}] success -> {out_path} (layout={spec['layout']}, {len(spec['panels'])} panels)")
        return True

    print(f"  gave up after {retries + 1} attempts")
    return False


def generate_svg(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model to write SVG markup; render to PNG with cairosvg."""
    import json as _json
    prompt = SVG_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        temperature = 0.0 if attempt == 0 else 0.5
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=1024, temperature=temperature)
        cleaned = clean_response(response.text)
        svg_code = extract_fenced_block(cleaned)
        if svg_code is None:
            print(f"  [attempt {attempt}] no fenced code block found")
            prompt = SVG_REPAIR_PROMPT.format(request=request, reason="no SVG code block found in response")
            continue
        # Strip any leading non-SVG content (the model sometimes adds prose before the tag)
        if "<svg" not in svg_code:
            print(f"  [attempt {attempt}] extracted block contains no <svg> tag")
            prompt = SVG_REPAIR_PROMPT.format(request=request, reason="no <svg> element found in the code block")
            continue
        svg_code = svg_code[svg_code.index("<svg"):]
        # Save intermediate SVG for debugging
        Path(out_path).with_suffix(f".attempt{attempt}.svg").write_text(svg_code, encoding="utf-8")
        try:
            render_svg(svg_code, out_path)
        except Exception as e:
            print(f"  [attempt {attempt}] render error: {e}")
            prompt = SVG_REPAIR_PROMPT.format(request=request, reason=str(e))
            continue
        print(f"  [attempt {attempt}] success -> {out_path}")
        return True
    print(f"  gave up after {retries + 1} attempts")
    return False


def generate_mermaid(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model to write Mermaid syntax; render to PNG with mmdc CLI."""
    prompt = MERMAID_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        temperature = 0.0 if attempt == 0 else 0.5
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=768, temperature=temperature)
        cleaned = clean_response(response.text)
        mermaid_code = extract_fenced_block(cleaned)
        if mermaid_code is None:
            print(f"  [attempt {attempt}] no fenced code block found")
            prompt = MERMAID_REPAIR_PROMPT.format(request=request, reason="no fenced code block found")
            continue
        # Save intermediate DSL text for debugging
        Path(out_path).with_suffix(f".attempt{attempt}.mmd").write_text(mermaid_code, encoding="utf-8")
        try:
            render_mermaid(mermaid_code, out_path)
        except Exception as e:
            print(f"  [attempt {attempt}] render error: {e}")
            prompt = MERMAID_REPAIR_PROMPT.format(request=request, reason=str(e))
            continue
        print(f"  [attempt {attempt}] success -> {out_path}")
        return True
    print(f"  gave up after {retries + 1} attempts")
    return False


def _clean_spec_strings(obj):
    """Recursively strip non-ASCII characters from all string values in a JSON spec."""
    if isinstance(obj, str):
        cleaned = CJK_RE.sub("", obj).strip()
        # Remove trailing incomplete words/punctuation caused by the strip
        return cleaned.rstrip("（(")
    if isinstance(obj, dict):
        return {k: _clean_spec_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_spec_strings(v) for v in obj]
    return obj


_VEGALITE_SCHEME_FIXES = {
    "yelloworange": "yelloworangered",
    "redyellow": "redyellowblue",
    "bluered": "redblue",
    "orangered": "orangered",  # valid, keep as-is
}

_MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_SET = set(_MONTH_ORDER)


def _fix_vegalite_spec(spec: dict) -> dict:
    """Patch common model errors before rendering:
    - Replace invalid scheme names (e.g. 'yelloworange' → 'yelloworangered')
    - Fix 'area' mark used where 'rect' is correct (ordinal x + ordinal y = heatmap)
    - Fix month axis sorting to chronological order
    """
    import json as _json
    spec_str = _json.dumps(spec)
    for wrong, correct in _VEGALITE_SCHEME_FIXES.items():
        if wrong == correct:
            continue
        spec_str = spec_str.replace(f'"scheme": "{wrong}"', f'"scheme": "{correct}"')
    spec = _json.loads(spec_str)
    # These patches are best-effort heuristics on top of a shape the model
    # doesn't always get right (e.g. "data": [...] instead of the expected
    # "data": {"values": [...]}), so a malformed spec must fall back to the
    # unpatched spec rather than crash the whole generation attempt.
    try:
        # Fix 'area' mark misused for heatmaps: if both x and y encodings are
        # ordinal/nominal and there's a quantitative color encoding, this
        # should be 'rect'.
        enc = spec.get("encoding", {})
        if isinstance(enc, dict):
            x_type = enc.get("x", {}).get("type", "") if isinstance(enc.get("x"), dict) else ""
            y_type = enc.get("y", {}).get("type", "") if isinstance(enc.get("y"), dict) else ""
            color_type = enc.get("color", {}).get("type", "") if isinstance(enc.get("color"), dict) else ""
            if (spec.get("mark") == "area"
                    and x_type in ("ordinal", "nominal")
                    and y_type in ("ordinal", "nominal")
                    and color_type == "quantitative"):
                spec["mark"] = "rect"
        # Fix month sorting: detect month abbreviations in data and add chronological sort
        data = spec.get("data", {})
        data_values = data.get("values", []) if isinstance(data, dict) else []
        if data_values:
            month_fields = {
                field for row in data_values if isinstance(row, dict)
                for field, value in row.items()
                if value in _MONTH_SET
            }
            if month_fields:
                def _add_month_sort(enc_dict):
                    for axis in ("x", "y"):
                        ax = enc_dict.get(axis, {})
                        if isinstance(ax, dict) and ax.get("field") in month_fields and "sort" not in ax:
                            ax["sort"] = _MONTH_ORDER
                if isinstance(spec.get("encoding"), dict):
                    _add_month_sort(spec["encoding"])
                for layer in spec.get("layer", []):
                    if isinstance(layer, dict) and "encoding" in layer:
                        _add_month_sort(layer["encoding"])
    except Exception as e:
        print(f"  [_fix_vegalite_spec] skipping heuristic patches, spec shape unexpected: {e}")
    return spec


def generate_vegalite(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model to write a Vega-Lite JSON spec; render to PNG with vl-convert."""
    import json as _json
    prompt = VEGALITE_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        temperature = 0.0 if attempt == 0 else 0.5
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=1536, temperature=temperature)
        cleaned = clean_response(response.text)
        spec_text = extract_fenced_block(cleaned)
        if spec_text is None:
            print(f"  [attempt {attempt}] no fenced code block found")
            prompt = VEGALITE_REPAIR_PROMPT.format(request=request, reason="no JSON code block found")
            continue
        Path(out_path).with_suffix(f".attempt{attempt}.json").write_text(spec_text, encoding="utf-8")
        try:
            spec = _json.loads(spec_text)
        except _json.JSONDecodeError as e:
            print(f"  [attempt {attempt}] JSON parse error: {e}")
            prompt = VEGALITE_REPAIR_PROMPT.format(request=request, reason=f"invalid JSON: {e}")
            continue
        if not isinstance(spec, dict):
            # The model sometimes emits a bare JSON array; _fix_vegalite_spec's
            # spec.get(...) then crashes with AttributeError instead of retrying.
            print(f"  [attempt {attempt}] spec is a JSON {type(spec).__name__}, expected an object")
            prompt = VEGALITE_REPAIR_PROMPT.format(
                request=request, reason="the top-level JSON must be an object, not an array"
            )
            continue
        spec = _clean_spec_strings(spec)
        spec = _fix_vegalite_spec(spec)
        try:
            render_vegalite(spec, out_path)
        except Exception as e:
            print(f"  [attempt {attempt}] render error: {e}")
            prompt = VEGALITE_REPAIR_PROMPT.format(request=request, reason=str(e))
            continue
        print(f"  [attempt {attempt}] success -> {out_path}")
        return True
    print(f"  gave up after {retries + 1} attempts")
    return False


_CSS_COLOR_ORPHAN_RE = re.compile(r'(#[0-9a-fA-F]{3,8})\s+[A-Za-z]\w*(?=[\s";,]|$)')
_CSS_UNIT_ORPHAN_RE = re.compile(r'(\d+(?:px|em|rem|vh|vw|%|pt))\s+[A-Za-z]\w*(?=[\s";,]|$)')


def _clean_html(html_code: str) -> str:
    """Strip CJK contamination from HTML/CSS and remove stray fence markers."""
    # Strip any leading fence marker (model may hit token limit before closing ```)
    lines = html_code.splitlines()
    while lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    html_code = "\n".join(lines)
    # Remove CJK characters injected into CSS property values by the model
    html_code = CJK_RE.sub("", html_code)
    # After CJK removal, orphaned English "translation" words remain (e.g. "#3498db acquire").
    # Strip any word token immediately following a CSS hex color or a unit value inside
    # style attributes — those are always model artifacts, not valid CSS.
    def _scrub_style(m: re.Match) -> str:
        content = m.group(1)
        content = _CSS_COLOR_ORPHAN_RE.sub(r'\1', content)
        content = _CSS_UNIT_ORPHAN_RE.sub(r'\1', content)
        return f'style="{content}"'
    html_code = re.sub(r'style="([^"]*)"', _scrub_style, html_code)
    return html_code


def generate_html(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Ask the model to write an HTML page; render to PNG with chromium headless."""
    prompt = HTML_PROMPT.format(request=request)
    for attempt in range(retries + 1):
        temperature = 0.0 if attempt == 0 else 0.5
        response = client.chat(prompt, think_mode=think_mode, max_output_tokens=1536, temperature=temperature)
        cleaned = clean_response(response.text)
        html_code = extract_fenced_block(cleaned)
        if html_code is None:
            # Model may have output raw HTML without fences — check for <html>/<body>
            if "<html" in cleaned.lower() or "<body" in cleaned.lower():
                html_code = cleaned
            else:
                print(f"  [attempt {attempt}] no HTML block found")
                prompt = HTML_REPAIR_PROMPT.format(request=request, reason="no HTML code block found in response")
                continue
        html_code = _clean_html(html_code)
        if "<!DOCTYPE" not in html_code and "<html" not in html_code.lower():
            # Wrap bare fragment in a minimal page shell
            html_code = f"<!DOCTYPE html><html><body style='margin:20px;font-family:sans-serif'>{html_code}</body></html>"
        Path(out_path).with_suffix(f".attempt{attempt}.html").write_text(html_code, encoding="utf-8")
        try:
            render_html(html_code, out_path)
            print(f"  [attempt {attempt}] rendered HTML successfully")
            return True
        except Exception as e:
            print(f"  [attempt {attempt}] render_html failed: {e}")
            prompt = HTML_REPAIR_PROMPT.format(request=request, reason=str(e))
    print(f"  gave up after {retries + 1} attempts")
    return False


def _make_gray_panel(label: str, size: int = 280) -> bytes:
    """Return a simple gray placeholder PNG with a centered label."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (size, size), color="#e8e8e8")
        draw = ImageDraw.Draw(img)
        draw.rectangle([2, 2, size - 3, size - 3], outline="#aaaaaa", width=2)
        # Draw label centered
        text = label or "?"
        try:
            # Try to use a larger font if available
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        except Exception:
            font = None
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((size - tw) // 2, (size - th) // 2), text, fill="#666666", font=font)
        else:
            draw.text((size // 2, size // 2), text, fill="#666666", anchor="mm")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        # Absolute fallback: 1x1 white PNG
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff"
            b"?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
        )


def generate_html_composite(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    think_mode: bool = True,
) -> bool:
    """Extract a multi-panel grid layout, generate each panel independently
    via generate_visualization() (without recursing into HTML_COMPOSITE again),
    embed all panels as base64 in an HTML page, render to PNG."""
    prompt = HTML_COMPOSITE_PROMPT.format(request=request)
    title = ""
    panels: list[tuple[str, str]] = []  # (description, label)

    for attempt in range(retries + 1):
        temperature = 0.0 if attempt == 0 else 0.5
        response = client.chat(
            prompt, think_mode=think_mode, max_output_tokens=768, temperature=temperature,
        )
        cleaned = clean_response(response.text)
        title = ""
        panels = []
        for line in cleaned.splitlines():
            line = line.strip()
            if line.upper().startswith("TITLE:"):
                title = line[6:].strip()
            elif line.upper().startswith("PANEL:"):
                content = line[6:].strip()
                if "|" in content:
                    desc, label = content.rsplit("|", 1)
                    panels.append((desc.strip(), label.strip()))
                else:
                    panels.append((content.strip(), ""))
        if panels:
            break
        print(f"  [composite attempt {attempt}] no PANEL: lines found, retrying...")
        prompt = HTML_COMPOSITE_REPAIR_PROMPT.format(
            request=request, reason="no PANEL: lines found",
        )

    if not panels:
        print("  [composite] gave up: no panels extracted")
        return False

    print(f"  [composite] generating {len(panels)} sub-panels...")

    # Generate each sub-panel independently (depth-limited: allow_composite=False)
    PANEL_SIZE = 280
    sub_images: list[tuple[bytes, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (desc, label) in enumerate(panels):
            sub_path = str(Path(tmp) / f"panel_{i}.png")
            print(f"  [composite] panel {i+1}/{len(panels)} ({label!r}): {desc[:70]}...")
            ok = generate_visualization(
                client, desc, sub_path,
                retries=1,                   # fewer retries to keep total time manageable
                verify_strategy="none",      # skip verification for sub-panels
                width=PANEL_SIZE, height=PANEL_SIZE,
                think_mode=think_mode,
                allow_composite=False,       # prevent infinite recursion
            )
            if ok and Path(sub_path).exists():
                sub_images.append((Path(sub_path).read_bytes(), label))
            else:
                print(f"  [composite] panel {i+1} failed, using placeholder")
                sub_images.append((_make_gray_panel(label, PANEL_SIZE), label))

        # Build HTML page with base64-embedded images
        n = len(sub_images)
        n_cols = min(4, n)
        n_rows = math.ceil(n / n_cols)
        viewport_w = n_cols * (PANEL_SIZE + 30) + 60
        viewport_h = 80 + (50 if title else 0) + n_rows * (PANEL_SIZE + 60) + 40

        panel_divs = ""
        for img_bytes, label in sub_images:
            b64 = base64.b64encode(img_bytes).decode()
            label_html = f'<div class="label">{label}</div>' if label else ""
            panel_divs += f"""
    <div class="panel">
      <img src="data:image/png;base64,{b64}" alt="{label}">
      {label_html}
    </div>"""

        title_html = f'<h2 style="text-align:center;margin:0 0 16px;font-size:20px">{title}</h2>' if title else ""
        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ background:#fff; margin:20px; font-family:Arial,sans-serif; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:20px; justify-content:center; align-items:flex-start; }}
  .panel {{ text-align:center; }}
  .panel img {{ width:{PANEL_SIZE}px; height:{PANEL_SIZE}px; object-fit:contain;
               border:1px solid #ccc; background:#fafafa; display:block; }}
  .label {{ font-size:22px; font-weight:bold; margin-top:8px; color:#111; }}
</style>
</head>
<body>
{title_html}
<div class="grid">{panel_divs}
</div>
</body>
</html>"""

        try:
            render_html(html, out_path, width=viewport_w, height=viewport_h)
            print(f"  [composite] success -> {out_path} ({n} panels, {n_cols}x{n_rows} grid)")
            return True
        except Exception as e:
            print(f"  [composite] render_html failed: {e}")
            return False


CATEGORIES = ("NATURAL", "DIAGRAM", "GRAPH", "MOLECULE", "CIRCUIT", "VECTOR", "VENN", "TABLE", "SCENE", "PUZZLE", "GEOMETRY", "FIGURE", "SVG", "MERMAID", "VEGALITE", "HTML", "HTML_COMPOSITE")

# Regex that catches body-part / botanical descriptions the LLM misclassifies as GEOMETRY.
# Only unambiguous biological terms are included (words like "grass", "stem", "root", "branch"
# are excluded because they appear in diagram/code/flow contexts too).
# Guard: not triggered when the request also contains "diagram/schematic/anatomical" (those are DIAGRAM).
_NATURAL_BODY_PLANT_RE = re.compile(
    r'\b(?:foot|feet|hand|palm|finger|thumb|toe|knee|ankle|elbow|wrist|shoulder|hip'
    r'|eye|nose|mouth|ear|neck|chest|spine|rib|skull|bone|muscle|skin'
    r'|brain|lung|liver|kidney|stomach|intestine|artery|vein|nerve'
    r'|leaf|leaves|petal|sepal|flower|blossom|bloom'
    r'|moss|algae|fern|mushroom|fungi|pollen|spore'
    r'|fur|feather|claw|hoof|paw|snout|beak|gill|whisker)\b',
    re.IGNORECASE,
)
_DIAGRAM_SIGNAL_RE = re.compile(
    r'\b(?:diagram|schematic|anatomical|cross.?section|anatomy|flowchart|food\s+chain)\b',
    re.IGNORECASE,
)

# Regex that catches requests describing N labeled image panels — the LLM reliably misses these.
_COMPOSITE_RE = re.compile(
    r'(?:'
    r'\b(?:image|photo|picture)s?\s+(?:labeled\s+|showing\s+)?[A-Da-d1-5]'         # "image A", "images labeled A"
    r'|\bpanels?\s+(?:labeled|showing)\s+(?:[A-Da-d1-5]|different)'                 # "panels labeled A"
    r'|\bpanel\s+[A-Da-d1-5]\s+(?:shows?|is\b|depicts?|contains?)'                 # "panel A shows", "panel 1 is"
    r'|\b(?:three|four|five|[2-5])\s+(?:image|photo|picture)s?\b'                   # "three images"
    r'|labeled\s+[A-Da-d]\s+through\s+[A-Da-d]'                                     # "labeled A through D"
    r'|labeled\s+[A-Da-d][-–][A-Da-d]'                                              # "labeled A-D"
    r'|\b(?:left|right|top|bottom)\s+panel\b.{0,60}labeled\s+[A-Da-d]'             # "left panel ... labeled A"
    r')',
    re.IGNORECASE,
)


def classify_request(
    client: UMMClient, request: str, allow_composite: bool = True, think_mode: bool = False,
) -> str:
    """Return one of CATEGORIES. Defaults to FIGURE on an unclear answer."""
    # Fast-path: regex detects multi-panel labeled image grids before calling the LLM.
    if allow_composite and _COMPOSITE_RE.search(request):
        return "HTML_COMPOSITE"
    # Fast-path: body-part / botanical descriptions the LLM often misclassifies as GEOMETRY.
    if _NATURAL_BODY_PLANT_RE.search(request) and not _DIAGRAM_SIGNAL_RE.search(request):
        return "NATURAL"
    response = client.chat(
        CLASSIFY_PROMPT.format(request=request),
        think_mode=think_mode, max_output_tokens=20, temperature=0.0,
    )
    answer = clean_response(response.text).strip().upper()
    cats = CATEGORIES if allow_composite else tuple(c for c in CATEGORIES if c != "HTML_COMPOSITE")
    # Check longest match first so HTML_COMPOSITE beats HTML
    for category in sorted(cats, key=len, reverse=True):
        if category in answer:
            return category
    return "FIGURE"


def generate_visualization(
    client: UMMClient, request: str, out_path: str, retries: int = 2,
    verify_strategy: str = "informed", width: int | None = None, height: int | None = None,
    allow_composite: bool = True, verify_all: bool = False,
    think_mode: bool = True, confidence_threshold: int | None = None,
    discard_unverified: bool = False, criterion: str = "verdict",
    verify_think_mode: bool | None = None, verify_max_attempts: int = 5,
    force_natural: bool = False,
) -> bool:
    """Single general entry point: classify the request, then dispatch to
    whichever pipeline fits.

    force_natural=True skips classify_request entirely and treats every
    request as NATURAL (model image-gen head only, no deterministic
    renderers) -- used for the classifier-ablation study. Default False
    keeps the real classifier.

    allow_composite=False is used for recursive sub-panel generation inside
    HTML_COMPOSITE to prevent infinite recursion.

    verify_strategy/width/height only affect NATURAL by default.

    verify_all=True additionally wraps every non-NATURAL category in the
    critic verify/revise loop (generate_with_verification). The default stays
    False because benchmarking showed the strict photo critic lowers average
    quality on structured line-diagram/chart content — opt in only for runs
    that explicitly want all-category verification.

    think_mode controls classification and DSL/code generation calls.
    verify_think_mode (defaults to think_mode when None) controls the
    critic/revise calls independently -- see generate_natural's docstring
    for why "think only in verify" is worth using over "think everywhere".

    confidence_threshold/discard_unverified are forwarded to
    generate_with_verification -- see its docstring.

    verify_max_attempts (default 5, matches generate_with_verification's own
    default): how many generate+verify rounds the critic loop gets. Callers
    that want a single-shot attempt (no self-refine, no revise-on-failure --
    see synthesize_vqa.py's BFS strategy) pass retries=0, verify_max_attempts=1."""
    kind = "NATURAL" if force_natural else classify_request(client, request, allow_composite=allow_composite, think_mode=think_mode)
    print(f"  classified as: {kind}")
    if kind == "NATURAL":
        return generate_natural(
            client, request, out_path, retries,
            verify_strategy=verify_strategy, width=width, height=height,
            think_mode=think_mode, confidence_threshold=confidence_threshold,
            discard_unverified=discard_unverified, criterion=criterion,
            verify_think_mode=verify_think_mode, verify_max_attempts=verify_max_attempts,
        )
    if kind == "HTML_COMPOSITE":
        generate_once = lambda req: generate_html_composite(client, req, out_path, retries, think_mode=think_mode)
    else:
        pipeline = {
            "DIAGRAM": generate_diagram,
            "GRAPH": generate_graph,
            "MOLECULE": generate_molecule,
            "CIRCUIT": generate_circuit,
            "VECTOR": generate_vectors,
            "VENN": generate_venn,
            "TABLE": generate_table,
            "SCENE": generate_scene,
            "PUZZLE": generate_puzzle,
            "GEOMETRY": generate_geometry,
            "FIGURE": generate_figure,
            "SVG": generate_svg,
            "MERMAID": generate_mermaid,
            "VEGALITE": generate_vegalite,
            "HTML": generate_html,
        }[kind]
        generate_once = lambda req: pipeline(client, req, out_path, retries, think_mode=think_mode)
    if verify_all:
        # "score" was calibrated on NATURAL photos and is too strict for
        # DSL/code-rendered categories (see verify.py's _CRITIC_PROMPT_SCORE_STRUCTURED
        # docstring: it scored visually-correct charts at 0-40 over exact-number
        # mismatches the renderer can't be expected to reproduce) -- route those
        # categories to the structural rubric automatically.
        non_natural_criterion = "score_structured" if criterion == "score" else criterion
        return generate_with_verification(
            client, request, out_path, generate_once, strategy=verify_strategy,
            think_mode=verify_think_mode if verify_think_mode is not None else think_mode,
            confidence_threshold=confidence_threshold,
            discard_unverified=discard_unverified, criterion=non_natural_criterion,
            max_attempts=verify_max_attempts,
        )
    return generate_once(request)
