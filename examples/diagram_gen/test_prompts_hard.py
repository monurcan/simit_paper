"""A second, harder benchmark suite -- long, fully-detailed descriptions
spanning every category (including NATURAL), kept separate from
TEST_PROMPTS in test_prompts.py so this never touches or invalidates the
original suite's results. Every description here aims to fully specify the
target image (every number, every relationship) the way a real user request
would, rather than the shorter phrasing used for the original suite."""

HARD_TEST_PROMPTS = {
    # ---------------------------------------------------------------
    # NATURAL -- realistic/illustrative scenes, the model's image head
    # ---------------------------------------------------------------
    "natural_lighthouse_storm": (
        "A weathered lighthouse keeper in a yellow raincoat stands at the top "
        "railing of a tall stone lighthouse during a violent nighttime storm, "
        "lightning illuminating huge crashing waves on the rocks below, rain "
        "streaking sideways in the wind, photorealistic, dramatic cinematic "
        "lighting."
    ),
    "natural_ikebana": (
        "An elderly Japanese woman in a traditional indigo kimono carefully "
        "arranges pink cherry blossom branches in a low ceramic ikebana bowl, "
        "soft afternoon light streaming through a paper shoji window behind "
        "her, delicate watercolor painting style with visible brush texture."
    ),
    "natural_spice_market": (
        "A bustling Moroccan spice market alley at dusk, vendors standing "
        "behind woven baskets piled into colorful pyramids of saffron, "
        "paprika, and turmeric, strings of warm lanterns overhead casting "
        "golden light, oil painting style with rich warm tones."
    ),
    "natural_dew_web": (
        "An extreme macro photograph of a dew-covered spider web strung "
        "between two blades of grass at dawn, individual water droplets "
        "catching the early sunlight like tiny prisms, a single red ladybug "
        "crawling along one strand, shallow depth of field, soft blurred "
        "green background."
    ),
    "natural_steampunk_airship": (
        "A brass-and-copper steampunk airship with large canvas balloon "
        "sails floats above a dense Victorian-era city at golden hour, steam "
        "venting from pipes along its hull, smaller airships in the distant "
        "sky, detailed digital concept art style."
    ),
    "natural_arctic_foxes": (
        "Two fluffy arctic foxes play-fighting and tumbling in fresh deep "
        "snow at night under a vivid green and purple aurora borealis "
        "filling the sky above them, photorealistic wildlife photography, "
        "long exposure starry sky."
    ),

    # ---------------------------------------------------------------
    # GEOMETRY -- multi-step constructions with full numeric detail
    # ---------------------------------------------------------------
    "hard_geom_median": (
        "Draw triangle ABC with vertices A at (0,0), B at (8,0), and C at "
        "(3,6). Draw the median from vertex C to the midpoint of side AB, "
        "mark that midpoint as point M, and label the computed lengths of "
        "sides AC, BC, and the median CM."
    ),
    "hard_geom_inscribed_circumscribed": (
        "Draw a square with side length 6, centered at the origin. Inscribe "
        "a circle inside it that is tangent to all four sides, and also draw "
        "a second, larger circle that passes exactly through all four "
        "corners of the square. Label the radius of each of the two circles."
    ),
    "hard_geom_trapezoid_diagonals": (
        "Draw an isosceles trapezoid with parallel sides of length 10 and 4 "
        "and a height of 5. Draw both diagonals of the trapezoid and label "
        "their lengths (compute them) -- they should come out equal since "
        "the trapezoid is isosceles."
    ),
    "hard_geom_triangle_angle_sum": (
        "Draw a triangle with vertices A(0,0), B(7,0), and C(2,5). Compute "
        "and label all three interior angles of the triangle at their "
        "respective vertices, and also label all three side lengths."
    ),

    # ---------------------------------------------------------------
    # FIGURE -- harder, multi-series / multi-feature plots
    # ---------------------------------------------------------------
    "hard_fig_sin_cos_intersect": (
        "Plot y = sin(x) and y = cos(x) on the same axes for x ranging from "
        "-2*pi to 2*pi, using two different colored lines with a legend "
        "distinguishing them, and mark every point in that range where the "
        "two curves intersect with a visible dot."
    ),
    "hard_fig_grouped_revenue": (
        "Create a grouped bar chart comparing quarterly revenue in millions "
        "of dollars for three product lines across four quarters: Hardware "
        "= [12, 15, 11, 18], Software = [8, 9, 12, 14], Services = [5, 6, 7, "
        "9]. Label the x-axis with Q1 through Q4, the y-axis as 'Revenue "
        "($M)', and include a legend for the three product lines."
    ),
    "hard_fig_dual_axis": (
        "Plot a company's monthly website visitors (in thousands) as a blue "
        "line on the left y-axis -- values [10, 12, 15, 14, 18, 22] for "
        "January through June -- and the conversion rate percentage as a "
        "red line on a separate right y-axis -- values [2.1, 2.3, 2.0, 2.5, "
        "2.8, 3.0] for the same six months."
    ),
    "hard_fig_annotated_histogram": (
        "Draw a histogram of 200 exam scores that are roughly normally "
        "distributed with a mean of 75 and standard deviation of 10, using "
        "15 bins, and draw a vertical dashed line at the mean labeled 'Mean "
        "= 75'."
    ),

    # ---------------------------------------------------------------
    # DIAGRAM / GRAPH -- more steps and branches than the base suite
    # ---------------------------------------------------------------
    "hard_diagram_coffee_order": (
        "Draw a flowchart for a coffee shop order process with these exact "
        "steps and branches: Start, then Customer places order, then Check "
        "if item is in stock -- if YES go to Prepare drink, if NO go to "
        "Notify customer and suggest an alternative. After preparing the "
        "drink, go to Customer pays, then Order complete. After suggesting "
        "an alternative, also route back to Customer places order. Label "
        "the Yes/No branches clearly."
    ),
    "hard_graph_social_network": (
        "Draw a directed graph of a small social network's 'follows' "
        "relationships: Alice follows Bob, Bob follows Carol, Carol follows "
        "Alice, Alice follows Dave, Dave follows Carol, and Dave follows "
        "Bob."
    ),

    # ---------------------------------------------------------------
    # MOLECULE / CIRCUIT
    # ---------------------------------------------------------------
    "hard_molecule_adrenaline": "Show the molecular structure of adrenaline (epinephrine).",
    "hard_circuit_parallel_led": (
        "Draw a circuit with a 12V battery in series with a switch, "
        "connected to two resistors in parallel (a 100 ohm resistor and a "
        "220 ohm resistor), with that parallel combination in series with "
        "an LED, all forming one closed loop."
    ),

    # ---------------------------------------------------------------
    # VECTOR / VENN / TABLE / SCENE / PUZZLE
    # ---------------------------------------------------------------
    "hard_vector_incline": (
        "Draw a free body diagram of a block resting on a frictionless "
        "incline angled at 25 degrees from horizontal, showing three "
        "labeled vectors: gravity pointing straight down, the normal force "
        "perpendicular to the incline surface, and friction is NOT present "
        "since the incline is frictionless -- only show gravity and the "
        "normal force."
    ),
    "hard_venn_mammals_flying": (
        "Draw a Venn diagram of two overlapping sets, 'Mammals' and "
        "'Animals that can fly'. Put 'Bats' in the overlap region (since "
        "bats are flying mammals), put 'Dogs' and 'Cats' in the Mammals-only "
        "region, and put 'Eagles' and 'Sparrows' in the flying-only region."
    ),
    "hard_table_periodic": (
        "Make a table of the first five elements of the periodic table with "
        "columns Name, Symbol, Atomic Number, and Atomic Mass: Hydrogen (H, "
        "1, 1.008), Helium (He, 2, 4.0026), Lithium (Li, 3, 6.94), Beryllium "
        "(Be, 4, 9.0122), Boron (B, 5, 10.81)."
    ),
    "hard_scene_mixed_shapes": (
        "Draw a scene scattered randomly with 4 overlapping blue circles, 3 "
        "red squares that do not overlap each other or anything else, and 2 "
        "green triangles that overlap each other."
    ),
    "hard_puzzle_shape_analogy": (
        "Draw an analogy puzzle with four panels: a small triangle, then a "
        "large triangle, then a small square, then a question mark "
        "representing what comes next in the pattern (it should be a large "
        "square, following the small-to-large pattern, but leave the last "
        "panel as a question mark since the answer isn't given)."
    ),
}
