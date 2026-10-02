"""Skill-library ablation: for a fixed set of 3 prompts per category (17
categories), render each prompt twice -- once through the real pipeline
(classify -> deterministic renderer / skill library for structured
categories) and once with force_natural=True (BAGEL's native image-gen head
only, no skill library) -- so the two can be compared side by side."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mstar import MStarClient

from diagram_gen.pipeline import generate_visualization
from diagram_gen.test_prompts import TEST_PROMPTS
from diagram_gen.test_prompts_hard import HARD_TEST_PROMPTS
from diagram_gen.test_prompts_web import WEB_TEST_PROMPTS

ALL_PROMPTS = {**TEST_PROMPTS, **HARD_TEST_PROMPTS, **WEB_TEST_PROMPTS}
ALL_PROMPTS.update({
    "composite_feet": "Three foot type images labeled A, B, C: foot A has a pronounced high arch seen from the side, foot B is completely flat with no visible arch, foot C has a normal moderate arch.",
    "composite_animals": "Four panels showing different animals labeled 1-4: panel 1 shows a golden retriever dog, panel 2 shows a tabby cat, panel 3 shows a green parrot, panel 4 shows a goldfish in water.",
    "composite_charts": "A comparison of three chart types labeled A, B, C: panel A shows a bar chart of monthly sales, panel B shows a pie chart of market share, panel C shows a line graph of revenue growth.",
})

SELECTED = {
    "NATURAL": ["natural_lighthouse_storm", "natural_ikebana", "natural_arctic_foxes"],
    "DIAGRAM": ["diagram_flowchart", "diagram_architecture_fanout", "diagram_state_machine"],
    "GRAPH": ["graph_weighted", "graph_tree", "graph_disconnected"],
    "MOLECULE": ["molecule_aspirin", "molecule_caffeine", "molecule_ibuprofen"],
    "CIRCUIT": ["circuit_led", "circuit_rc", "circuit_three_parallel"],
    "VECTOR": ["math_vector_addition", "vector_subtraction", "vector_three_forces"],
    # venn_animals swapped out: its prompt text ends with "...in this puzzle",
    # which makes the classifier pick PUZZLE instead of VENN (confirmed FAIL).
    "VENN": ["math_venn_diagram", "venn_fruits_overlap", "venn_school_subjects"],
    "TABLE": ["science_table", "table_pass_fail", "table_animals_classification"],
    "SCENE": ["scene_counting", "scene_shape_grid", "scene_size_comparison"],
    # puzzle_number_sequence swapped out: it's a numeric sequence on a number
    # line, which the classifier correctly routes to FIGURE (per its own
    # rules), not PUZZLE -- not a fair PUZZLE example for this comparison.
    "PUZZLE": ["science_puzzle_sequence", "puzzle_shape_pattern", "puzzle_odd_one_out"],
    "GEOMETRY": ["geometry_triangle", "geometry_circle_square", "geometry_angle"],
    "FIGURE": ["plot_bar_chart", "plot_pie_chart", "plot_scatter_trend"],
    "SVG": ["svg_house", "svg_traffic_light", "svg_star"],
    "MERMAID": ["mermaid_login_seq", "mermaid_animals_class", "mermaid_software_gantt"],
    "VEGALITE": ["vegalite_activity_heatmap", "vegalite_countries_bubble", "vegalite_layered_bars_line"],
    "HTML": ["html_invoice", "html_receipt", "html_profile_card"],
    "HTML_COMPOSITE": ["composite_feet", "composite_animals", "composite_charts"],
}

OUT_DIR = Path(__file__).parent / "diagram_outputs_skill_ablation"
WITH_SKILL_DIR = OUT_DIR / "with_skill"
WITHOUT_SKILL_DIR = OUT_DIR / "without_skill"

PORT = 8001


def run_one(category: str, name: str, request: str, force_natural: bool) -> tuple[str, bool]:
    out_dir = WITHOUT_SKILL_DIR if force_natural else WITH_SKILL_DIR
    out_path = str(out_dir / f"{category}_{name}.png")
    tag = f"{category}/{name} ({'without_skill' if force_natural else 'with_skill'})"
    if Path(out_path).exists() and "--resume" in sys.argv:
        print(f"  [skip] {tag} already exists")
        return tag, True
    client = MStarClient(f"http://localhost:{PORT}", timeout=600.0)
    try:
        ok = generate_visualization(client, request, out_path, force_natural=force_natural)
        print(f"  [{'OK' if ok else 'FAIL'}] {tag}")
        return tag, ok
    except Exception as e:
        print(f"  [ERROR] {tag}: {e}")
        return tag, False


def main() -> None:
    WITH_SKILL_DIR.mkdir(parents=True, exist_ok=True)
    WITHOUT_SKILL_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for category, names in SELECTED.items():
        for name in names:
            request = ALL_PROMPTS[name]
            jobs.append((category, name, request, True))   # without skill library
            jobs.append((category, name, request, False))  # with skill library

    # Concurrency=1: the server has one GPU/worker, so overlapping NATURAL
    # image-gen calls just queue behind each other -- and generate_natural's
    # own 90s fail-fast watchdog (see pipeline.py) then mistakes queueing
    # delay for a stalled server and gives up. Sequential avoids that.
    concurrency = 1
    results = {}
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(run_one, c, n, r, f): (c, n, f) for c, n, r, f in jobs}
        for fut in as_completed(futures):
            tag, ok = fut.result()
            results[tag] = ok

    print("\n=== Summary ===")
    n_ok = sum(results.values())
    for tag, ok in sorted(results.items()):
        print(f"  {'OK  ' if ok else 'FAIL'}  {tag}")
    print(f"\n{n_ok}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
