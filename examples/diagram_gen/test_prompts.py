"""The 204-prompt test/benchmark suite used by cli.main()."""

TEST_PROMPTS = {
    # -- diagrams --
    "diagram_flowchart": (
        "Draw a flowchart of a software release pipeline: Commit Code, then Run CI Tests, "
        "then a decision 'Tests Pass?' -- if no, loop back to Commit Code; if yes, go to "
        "Deploy to Staging, then Manual QA, then Deploy to Production."
    ),
    "diagram_architecture_linear": (
        "Draw a simple system architecture diagram with three boxes in a row: 'Client', "
        "'API Server', 'Database', connected left to right with arrows labeled 'HTTP request' "
        "and 'SQL query'."
    ),
    "diagram_architecture_fanout": (
        "Draw a system diagram: two services, 'Auth Service' and 'Order Service', both write "
        "to a shared 'Database'. 'Order Service' also calls a 'Payment Gateway'."
    ),
    # -- plots --
    "plot_function_roots": (
        "Plot y = x^2 - 4x + 3 for x from -2 to 6. Mark and label the two roots on the x-axis "
        "with red dots."
    ),
    "plot_bar_chart": (
        "Make a bar chart of quarterly revenue: Q1=120, Q2=150, Q3=90, Q4=200 (in thousands "
        "of dollars). Label the y-axis 'Revenue ($k)' and put the value on top of each bar."
    ),
    "plot_pie_chart": (
        "Draw a pie chart of programming language popularity: Python 45%, JavaScript 30%, "
        "Go 15%, Rust 10%."
    ),
    "plot_scatter_trend": (
        "Make a scatter plot of these (x, y) points: (1,2), (2,3.5), (3,5), (4,6), (5,8.5), "
        "(6,9), and draw a best-fit trend line through them."
    ),
    "plot_histogram": (
        "Draw a histogram of these exam scores with 5 bins: 55, 60, 67, 67, 70, 72, 75, 78, "
        "80, 80, 82, 85, 88, 90, 92, 95."
    ),
    # -- geometry --
    "geometry_triangle": (
        "Draw a right triangle with legs of length 5 and 12, mark the right angle, and label "
        "all three side lengths including the hypotenuse (compute it)."
    ),
    "geometry_circle_square": (
        "Draw a circle inscribed in a square. The square has side length 8, centered at the "
        "origin. Label the square's side length and the circle's radius."
    ),
    "geometry_angle": (
        "Draw two rays from a common point forming a 60 degree angle, and label the angle."
    ),
    # -- classical math-VQA style questions --
    "math_trig_sine": (
        "Plot y = sin(x) for x from 0 to 2*pi, and mark the maximum point with its coordinates."
    ),
    "math_area_under_curve": (
        "Shade the area under the curve y = x^2 between x=1 and x=3, and label the shaded "
        "region as the integral."
    ),
    "math_number_line_inequality": (
        "Draw a number line from -5 to 5, with shading showing the region where x > 2. Use "
        "an open circle at 2."
    ),
    "math_vector_addition": (
        "Draw two vectors from the origin: vector A = (3, 2) and vector B = (1, 4), and draw "
        "their vector sum as a third vector, all labeled."
    ),
    "math_unit_circle": (
        "Draw a unit circle and mark the point at 45 degrees with its (x, y) coordinates labeled."
    ),
    "math_venn_diagram": (
        "Draw a Venn diagram with two overlapping circles labeled 'Even Numbers' and "
        "'Multiples of 3', with the number 6 placed in the overlap region."
    ),
    "math_coordinate_distance": (
        "Plot two points A(1,2) and B(4,6) on a coordinate plane, draw the line segment "
        "between them, and label its length (compute it)."
    ),
    # -- scientific figures: tables, physics, puzzles --
    "science_table": (
        "Make a table of student scores: Alice 92 A, Bob 85 B, Carol 78 C. Columns: "
        "Name, Score, Grade."
    ),
    "science_free_body": (
        "Draw a free body diagram of a box on a flat surface with weight, normal force, "
        "an applied force, and friction, all labeled."
    ),
    "science_puzzle_sequence": (
        "Draw a sequence puzzle: 4 panels showing 1, 2, 3, then 4 dots arranged in a "
        "square pattern -- what comes next?"
    ),
    # -- molecules --
    "molecule_aspirin": "Show the molecular structure of aspirin.",
    "molecule_caffeine": "Draw the molecular structure of caffeine.",
    # -- circuits --
    "circuit_led": "Draw a circuit with a 9V battery, a 220 ohm resistor, and an LED in series.",
    "circuit_rc": "Draw a circuit with a battery, a resistor, and a capacitor in series.",
    # -- graph theory --
    "graph_weighted": (
        "Draw a weighted undirected graph with nodes A, B, C, D and edges: A-B weight 5, "
        "B-C weight 3, A-C weight 2, C-D weight 4."
    ),
    "graph_tree": (
        "Draw a binary tree graph: root connects to Left and Right children; Left "
        "connects to LL and LR."
    ),
    # -- abstract/synthetic scenes (MathVista-style) --
    "scene_counting": (
        "A scene contains 3 red circles, 2 blue squares, and 1 green triangle, "
        "scattered around randomly. Draw it."
    ),
    "novel_clock": "Draw a clock face showing the time 3:15, with hour and minute hands.",
    # -- diverse phrasing / robustness stress tests --
    "stress_plot_log": "I need a graph showing how y = log(x) behaves for x between 0.1 and 10.",
    "stress_diagram_loop": (
        "A factory process: Raw Materials go to Assembly, Assembly goes to Quality "
        "Check. If Quality Check fails, send back to Assembly. If it passes, go to "
        "Packaging, then Shipping."
    ),
    "stress_graph_social": (
        "A small social network: Alice knows Bob and Carol. Bob knows Dave. Carol "
        "knows Dave and Eve."
    ),
    # -- harder, more diverse / compound descriptions --
    "hard_log_compound": "Plot y = log(2*x) + 12 for x from 1 to 20, and mark the point where y = 15.",
    "hard_trig_compound": "Plot y = 3*sin(2*x) - cos(x/2) for x from -2*pi to 2*pi.",
    "hard_piecewise": (
        "Graph the piecewise function: y = x^2 for x < 0, and y = 2*x + 1 for x >= 0, "
        "over x in [-3, 3]."
    ),
    "hard_gaussian": (
        "Plot y = exp(-x^2/2) / sqrt(2*pi) (the standard normal distribution) for x from "
        "-4 to 4, and shade the area between x=-1 and x=1."
    ),
    "hard_geom_triangle_area": (
        "A triangle has vertices at (0,0), (6,0), and (3,4). Draw it, and compute and "
        "label all three side lengths and the area."
    ),
    "hard_geom_hexagon": (
        "Draw a regular hexagon inscribed in a circle of radius 4 centered at the "
        "origin, and label the radius."
    ),
    "hard_geom_two_circles": (
        "Draw two circles: one centered at (0,0) with radius 3, another centered at "
        "(5,0) with radius 2."
    ),
    "hard_geom_callout": (
        "Draw a right triangle and use a callout arrow to point out and label the "
        "right angle."
    ),
    "new_boxplot": (
        "Draw a box plot comparing test scores for two classes: Class A = "
        "[55,67,70,72,75,78,80,85,90], Class B = [60,65,68,73,77,82,88,91,95]."
    ),
    "new_stacked_bar": (
        "Make a STACKED bar chart of monthly sales by product: January has Product "
        "X=30, Product Y=20; February has Product X=40, Product Y=15; March has "
        "Product X=35, Product Y=25."
    ),
    "new_timeline": (
        "Draw a timeline of major events: 1969 Moon Landing, 1989 Berlin Wall Falls, "
        "2004 Facebook Founded, 2020 COVID Pandemic Begins."
    ),
    "new_errorbar": (
        "Plot the average height of 4 plant groups with error bars: Group A: 15 "
        "plus/minus 2 cm, Group B: 22 plus/minus 3 cm, Group C: 18 plus/minus 1.5 cm, "
        "Group D: 25 plus/minus 4 cm."
    ),
    "hard_diagram_order": (
        "Draw a flowchart for an online order system: Customer Places Order, then "
        "Check Inventory. If In Stock, go to Process Payment, then Ship Order, then "
        "Delivered. If Out of Stock, go to Notify Customer, then Cancel Order."
    ),
    "hard_graph_deps": (
        "Draw a directed graph representing task dependencies in a build system: Link "
        "depends on Compile. Test depends on Link. Package depends on Test and "
        "Document. Document depends on Compile."
    ),
    "hard_molecule_uncommon": "Show the molecular structure of vanillin.",
    "hard_circuit_series": "Draw a circuit with a battery, two resistors, and a capacitor all in series.",
    "hard_circuit_parallel": "Draw a circuit with a battery connected to two resistors in parallel.",
    # -- extended batch: broader coverage across all categories, extra weight on the
    # newest pipelines (VECTOR, VENN) --
    "vector_subtraction": (
        "Draw vector A = (5, 3) and vector B = (2, 4) from the origin, and draw the "
        "vector A - B also from the origin, all three labeled."
    ),
    "vector_three_forces": (
        "Draw three vectors from the origin representing forces: F1 = (4, 0), "
        "F2 = (0, 3), F3 = (-2, -1), each labeled with its name."
    ),
    "vector_negative_components": (
        "Draw two vectors from the origin: vector A = (-3, 2) and vector B = (4, -1), "
        "both labeled with their components."
    ),
    "fbd_incline": (
        "Draw a free body diagram of a block on a 30 degree frictionless incline, "
        "showing gravity pulling straight down and the normal force perpendicular to "
        "the incline surface."
    ),
    "fbd_tension_ropes": (
        "Draw a free body diagram of a hanging mass supported by two ropes at 45 "
        "degrees on either side, showing the weight straight down and the two tension "
        "forces."
    ),
    "venn_no_overlap_items": (
        "Draw a Venn diagram of two sets, 'Mammals' and 'Reptiles', with Dog and Cat "
        "in Mammals only, Snake and Lizard in Reptiles only, and nothing in the "
        "overlap."
    ),
    "venn_fruits_overlap": (
        "Draw a Venn diagram comparing 'Citrus Fruits' (Lemon, Lime) and 'Round "
        "Fruits' (Apple, Grapefruit), with Orange in the overlap of both sets."
    ),
    "venn_plain_labels": (
        "Draw a Venn diagram of Set P and Set Q with no specific items, just the two "
        "labeled overlapping circles."
    ),
    "diagram_fanin": (
        "Draw a flowchart where both 'Validate Email' and 'Validate Password' lead "
        "into a single 'Create Account' step, which then leads to 'Send Welcome "
        "Email'."
    ),
    "diagram_state_machine": (
        "Draw a state machine: Idle goes to Running on 'start', Running goes to "
        "Paused on 'pause', Paused goes to Running on 'resume', Running goes to Idle "
        "on 'stop'."
    ),
    "diagram_multi_loop": (
        "Draw a flowchart: Start, then Step A, then Step B, then a check 'Step B "
        "OK?' -- if no, go back to Step A; if yes, go to Step C, then a check 'Step C "
        "OK?' -- if no, go back to Step B; if yes, End."
    ),
    "graph_cycle": (
        "Draw an undirected graph that forms a cycle: A connects to B, B connects to "
        "C, C connects to D, D connects to A."
    ),
    "graph_disconnected": (
        "Draw a graph with two separate disconnected components: one with nodes A, "
        "B, C forming a triangle of edges, another with nodes X and Y connected by "
        "one edge."
    ),
    "graph_directed_cycle": (
        "Draw a directed graph with a cycle: A points to B, B points to C, C points "
        "to A."
    ),
    "molecule_glucose": "Draw the molecular structure of glucose.",
    "molecule_ibuprofen": "Show the molecular structure of ibuprofen.",
    "molecule_uncommon_dopamine": "Show the molecular structure of dopamine.",
    "circuit_switch_led": (
        "Draw a circuit with a battery, a switch, and a light bulb (use an LED) all "
        "in series."
    ),
    "circuit_three_parallel": (
        "Draw a circuit with a battery connected to three resistors in parallel."
    ),
    "geom_parallelogram": (
        "Draw a parallelogram with vertices at (0,0), (4,0), (5,3), and (1,3), and "
        "label all four side lengths."
    ),
    "geom_trapezoid": (
        "Draw a trapezoid with parallel sides of length 6 and 3 and height 4, and "
        "label the two parallel sides."
    ),
    "geom_similar_triangles": (
        "Draw two similar right triangles side by side: one with legs 3 and 4, "
        "another scaled by factor 2 with legs 6 and 8, label all six sides."
    ),
    "geom_circle_chord": (
        "Draw a circle of radius 5 with a chord across it, and label the chord's "
        "length and the circle's radius."
    ),
    "math_tangent_line": (
        "Plot y = x^2 for x from -3 to 3, and draw the tangent line at x=1, labeled "
        "with its equation (compute the slope)."
    ),
    "math_asymptote": (
        "Plot y = 1/x for x from -5 to 5 excluding 0, and draw a dashed vertical "
        "asymptote at x=0 and a dashed horizontal asymptote at y=0."
    ),
    "math_probability_bar": (
        "Draw a bar chart showing the probability of rolling each number 1 through 6 "
        "on a fair die, each bar at height 1/6, labeled."
    ),
    "math_complex_plane": (
        "Plot the complex number 3+4i as a vector from the origin on the complex "
        "plane, labeled with its magnitude (compute it)."
    ),
    "puzzle_shape_pattern": (
        "Draw a pattern puzzle with 4 panels: panel 1 has a circle, panel 2 a "
        "square, panel 3 a triangle, and panel 4 shows the answer if the pattern "
        "repeats from panel 1."
    ),
    "scene_shape_grid": (
        "Draw a 3x3 grid of shapes where each cell contains either a circle or a "
        "square, arranged in a checkerboard pattern."
    ),
    "table_larger": (
        "Make a table of quarterly sales by region: North Q1=100 Q2=120, South "
        "Q1=90 Q2=95, East Q1=110 Q2=130, West Q1=80 Q2=85."
    ),
    # -- third batch: scaled up coverage, inspired by broader VQA-style taxonomies
    # (GeoQA/Geometry3K, ScienceQA, ChartQA/PlotQA/FigureQA, IconQA, AI2D) beyond
    # what MathVista alone covers --
    "diagram_long_pipeline": (
        "Draw a flowchart of a compiler pipeline: Source Code, then Lexing, then "
        "Parsing, then Type Checking, then Optimization, then Code Generation, "
        "then Executable."
    ),
    "diagram_multi_branch": (
        "Draw a flowchart: 'Support Ticket' leads to a check 'Severity?' which "
        "branches to three outcomes: 'Low' goes to 'Queue for Later', 'Medium' "
        "goes to 'Assign to Agent', 'High' goes to 'Escalate Immediately'."
    ),
    "diagram_diamond_decision": (
        "Draw a flowchart: Start, then check 'Logged In?' -- if no, go to "
        "'Show Login Page'; if yes, check 'Is Admin?' -- if no, go to "
        "'User Dashboard'; if yes, go to 'Admin Dashboard'."
    ),
    "diagram_onboarding": (
        "Draw a flowchart for employee onboarding: Sign Offer Letter, then "
        "Complete Paperwork, then IT Setup, then Orientation, then Start Role."
    ),
    "diagram_recipe": (
        "Draw a flowchart for making tea: Boil Water, then Add Tea Leaves, then "
        "Steep for 3 Minutes, then Remove Leaves, then Serve."
    ),
    "diagram_error_handling": (
        "Draw a flowchart: 'Run Script', then check 'Error Occurred?' -- if yes, "
        "go to 'Log Error' then back to 'Run Script'; if no, go to 'Report Success'."
    ),
    "diagram_approval_chain": (
        "Draw a flowchart: 'Submit Request', then 'Manager Review' -- if "
        "rejected, go back to 'Submit Request'; if approved, go to 'Finance "
        "Review' -- if rejected, go back to 'Submit Request'; if approved, go "
        "to 'Request Fulfilled'."
    ),
    "diagram_food_chain": (
        "Draw a flowchart of a food chain: Sun, then Grass, then Rabbit, then Fox."
    ),
    "diagram_water_cycle": (
        "Draw a cyclic flowchart of the water cycle: Evaporation leads to "
        "Condensation, which leads to Precipitation, which leads to Collection, "
        "which leads back to Evaporation."
    ),
    # -- graph theory --
    "graph_bipartite": (
        "Draw a bipartite graph: students Alice and Bob each connect to the "
        "courses Math and Physics they are enrolled in; Alice takes Math and "
        "Physics, Bob takes only Physics."
    ),
    "graph_self_loop": (
        "Draw a directed graph with nodes A, B, C where A points to B, B points "
        "to C, and C has a self-loop back to itself."
    ),
    "graph_star": (
        "Draw a star-shaped graph: a central node 'Hub' connected to five leaf "
        "nodes: Leaf1, Leaf2, Leaf3, Leaf4, Leaf5."
    ),
    "graph_complete": (
        "Draw a complete graph on 4 nodes A, B, C, D where every pair of nodes "
        "is connected by an edge."
    ),
    "graph_large_tree": (
        "Draw a tree graph 3 levels deep: Root connects to Left and Right; Left "
        "connects to LeftLeft and LeftRight; Right connects to RightLeft and "
        "RightRight."
    ),
    "graph_weighted_directed": (
        "Draw a directed weighted graph of flight routes: NYC to LA weight 5 "
        "hours, NYC to Chicago weight 2 hours, Chicago to LA weight 4 hours."
    ),
    "graph_grid": (
        "Draw a graph forming a 2x3 grid: nodes A1, A2, A3 in a row connected "
        "left to right, nodes B1, B2, B3 in a row connected left to right, and "
        "each Ai connected to the Bi directly below it."
    ),
    "graph_friendship_large": (
        "Draw a social network graph: Alice knows Bob, Carol, and Dave. Bob "
        "knows Eve. Carol knows Eve and Frank. Dave knows Frank. Eve knows "
        "Frank."
    ),
    "graph_food_web": (
        "Draw a directed graph of a food web: Grass points to Rabbit, Rabbit "
        "points to Fox, Grass points to Mouse, Mouse points to Owl."
    ),
    "graph_road_network": (
        "Draw a weighted undirected graph of cities: CityA to CityB distance "
        "120, CityB to CityC distance 80, CityA to CityC distance 150, CityC to "
        "CityD distance 60."
    ),
    "graph_dependency_diamond": (
        "Draw a directed graph: A depends on B and C; both B and C depend on D."
    ),
    "graph_two_cycles": (
        "Draw an undirected graph with two triangles connected by a single "
        "bridge edge: triangle one has nodes A, B, C; triangle two has nodes D, "
        "E, F; there is one edge connecting C and D."
    ),
    "graph_organizational_chart": (
        "Draw an organizational chart: CEO connects to VP of Engineering and VP "
        "of Sales; VP of Engineering connects to Engineer1 and Engineer2; VP of "
        "Sales connects to SalesRep1 and SalesRep2."
    ),
    # -- molecules --
    "molecule_water": "Show the molecular structure of water.",
    "molecule_methane": "Draw the molecular structure of methane.",
    "molecule_benzene": "Show the molecular structure of benzene.",
    "molecule_acetone": "Draw the molecular structure of acetone.",
    "molecule_urea": "Show the molecular structure of urea.",
    "molecule_ethanol": "Draw the molecular structure of ethanol.",
    "molecule_toluene": "Show the molecular structure of toluene.",
    "molecule_uncommon_serotonin": "Show the molecular structure of serotonin.",
    "molecule_uncommon_menthol": "Draw the molecular structure of menthol.",
    # -- circuits --
    "circuit_two_batteries": (
        "Draw a circuit with two batteries and a resistor, all in series."
    ),
    "circuit_four_components": (
        "Draw a circuit with a battery, two resistors, a capacitor, and an LED, "
        "all in series."
    ),
    "circuit_parallel_led_resistor": (
        "Draw a circuit with a battery connected to an LED and a resistor in "
        "parallel with each other."
    ),
    "circuit_switch_parallel": (
        "Draw a circuit with a battery connected to two parallel branches: one "
        "branch has a switch, the other has a resistor."
    ),
    "circuit_inductor": (
        "Draw a circuit with a battery, a resistor, and an inductor all in "
        "series."
    ),
    "circuit_diode_protection": (
        "Draw a circuit with a battery, a diode, and a resistor all in series."
    ),
    "circuit_capacitor_parallel": (
        "Draw a circuit with a battery connected to two capacitors in parallel."
    ),
    # -- vectors / free body diagrams --
    "vector_named_angle": (
        "Draw a single vector from the origin with magnitude 10 pointing 30 "
        "degrees above the horizontal, labeled with its magnitude and angle."
    ),
    "vector_four_forces_equilibrium": (
        "Draw four force vectors from a common point that are in equilibrium: "
        "Up = (0, 4), Down = (0, -4), Left = (-3, 0), Right = (3, 0), all "
        "labeled."
    ),
    "vector_velocity_decomposition": (
        "Draw an initial velocity vector of magnitude 10 at 45 degrees from the "
        "origin, along with its horizontal and vertical component vectors, all "
        "labeled."
    ),
    "vector_displacement": (
        "Draw two displacement vectors from the origin: 'Walk 1' = (3, 0) and "
        "'Walk 2' = (0, 4), and the total displacement vector, all labeled."
    ),
    "vector_resultant_three": (
        "Draw three vectors from the origin: A = (2, 1), B = (-1, 3), C = (1, "
        "-2), and their resultant sum vector, all four labeled."
    ),
    "vector_named_directions": (
        "Draw two vectors from the same point: a force of 5 pointing due west "
        "and a force of 3 pointing due south, both labeled with their "
        "magnitude and direction."
    ),
    "fbd_circular_motion": (
        "Draw a free body diagram of an object in circular motion: a "
        "centripetal force vector pointing toward the center (left) and a "
        "velocity vector tangent to the path (up), both labeled."
    ),
    "vector_unit_vectors": (
        "Draw the two unit basis vectors from the origin: i-hat = (1, 0) and "
        "j-hat = (0, 1), both labeled."
    ),
    "fbd_elevator": (
        "Draw a free body diagram of a person standing in an elevator: weight "
        "pointing down and the normal support force pointing up, both labeled."
    ),
    "vector_opposite_equal": (
        "Draw two vectors from the origin with equal magnitude and opposite "
        "direction: A = (4, 0) and B = (-4, 0), both labeled."
    ),
    # -- venn diagrams --
    "venn_long_lists": (
        "Draw a Venn diagram of 'Prime Numbers' (2, 3, 5, 7, 11, 13) and 'Odd "
        "Numbers' (1, 3, 5, 7, 9, 11, 13), with the shared values in the "
        "overlap."
    ),
    "venn_school_subjects": (
        "Draw a Venn diagram of students taking 'Math' and students taking "
        "'Science', with Alice and Bob in Math only, Carol and Dave in Science "
        "only, and Eve taking both."
    ),
    "venn_animals": (
        "Draw a Venn diagram of 'Birds' (Eagle, Sparrow) and 'Mammals' (Dog, "
        "Cat), with Bat in the overlap since it has traits of both in this "
        "puzzle."
    ),
    "venn_disjoint_concept": (
        "Draw a Venn diagram of 'Fiction Books' and 'Cookbooks', with no items "
        "shared between the two sets."
    ),
    "venn_numbers_overlap_many": (
        "Draw a Venn diagram of 'Factors of 12' (1, 2, 3, 4, 6, 12) and "
        "'Factors of 18' (1, 2, 3, 6, 9, 18), with the common factors in the "
        "overlap."
    ),
    "venn_three_word_sets": (
        "Draw a Venn diagram of 'Fruits' (Apple, Pear) and 'Vegetables' "
        "(Carrot, Pea), with nothing in the overlap."
    ),
    "venn_colors_shapes": (
        "Draw a Venn diagram of 'Red Items' (Apple, Fire Truck) and 'Round "
        "Items' (Ball, Apple), with Apple in the overlap."
    ),
    # -- geometry (GeoQA/Geometry3K style) --
    "geom_parallel_lines_transversal": (
        "Draw two parallel horizontal lines cut by a single diagonal "
        "transversal line, and label one of the angles formed as 60 degrees."
    ),
    "geom_inscribed_angle": (
        "Draw a circle with an inscribed angle and label it, and draw the "
        "corresponding central angle subtending the same arc and label it as "
        "twice the inscribed angle."
    ),
    "geom_midpoint_slope": (
        "Plot two points A(1,1) and B(5,7), draw the segment between them, "
        "mark the midpoint, and label the segment's slope (compute it)."
    ),
    "geom_rectangle_diagonal": (
        "Draw a rectangle with width 6 and height 8, draw its diagonal, and "
        "label the diagonal's length (compute it)."
    ),
    "geom_isosceles_triangle": (
        "Draw an isosceles triangle with the two equal legs marked using tick "
        "marks, and label the two equal base angles as equal."
    ),
    "geom_regular_pentagon": (
        "Draw a regular pentagon inscribed in a circle of radius 4, and label "
        "one interior angle as 108 degrees."
    ),
    "geom_concentric_circles": (
        "Draw two concentric circles centered at the origin, one with radius 3 "
        "and one with radius 6, and label both radii."
    ),
    "geom_square_diagonal": (
        "Draw a square with side length 5, draw its diagonal, and label the "
        "diagonal's length (compute it)."
    ),
    "geom_3_4_5_triangle": (
        "Draw a right triangle with legs 6 and 8 (a 3-4-5 triangle scaled by "
        "2), mark the right angle, and label all three sides including the "
        "hypotenuse (compute it)."
    ),
    "geom_rhombus": (
        "Draw a rhombus with all four sides marked as equal length using tick "
        "marks, and draw both diagonals."
    ),
    "geom_equilateral_triangle": (
        "Draw an equilateral triangle with all three sides marked as equal "
        "length using tick marks, and label all three angles as 60 degrees."
    ),
    "geom_right_triangle_angle_sum": (
        "Draw a right triangle with one acute angle labeled 35 degrees, mark "
        "the right angle, and label the other acute angle (compute it)."
    ),
    "geom_kite": (
        "Draw a kite shape (a quadrilateral with two pairs of adjacent equal "
        "sides), marking one pair of equal sides with single tick marks and "
        "the other pair with double tick marks."
    ),
    "geom_angle_bisector": (
        "Draw a 60 degree angle between two rays from a common point, draw a "
        "ray bisecting it, and label the two resulting 30 degree half-angles."
    ),
    # -- plots / charts (ChartQA/PlotQA/FigureQA style) --
    "math_multi_series_compare": (
        "Plot three lines on one chart for x from 0 to 10: y = x, y = x^2/5, "
        "and y = x^1.5, with a legend."
    ),
    "math_grouped_bar": (
        "Make a grouped bar chart comparing test scores: Class A and Class B "
        "side by side for three tests: Test 1 (A=80, B=85), Test 2 (A=75, "
        "B=90), Test 3 (A=88, B=82)."
    ),
    "math_donut_chart": (
        "Draw a donut chart (a pie chart with a white circle in the middle) of "
        "market share: Company X 50%, Company Y 30%, Company Z 20%."
    ),
    "math_horizontal_bar": (
        "Draw a horizontal bar chart of city populations: Tokyo 14M, Delhi "
        "11M, Shanghai 9M, Sao Paulo 6M."
    ),
    "math_log_scale": (
        "Plot y = 2^x for x from 0 to 20 using a logarithmic y-axis scale."
    ),
    "math_step_function": (
        "Draw a step function plot of shipping cost by weight: 0-1kg costs $5, "
        "1-2kg costs $8, 2-3kg costs $10, for weight from 0 to 3kg."
    ),
    "math_derivative_compare": (
        "Plot y = x^3 - 3*x and its derivative y' = 3*x^2 - 3 on the same axes "
        "for x from -3 to 3, with a legend."
    ),
    "math_quadratic_vertex": (
        "Plot y = x^2 - 6x + 5 for x from -1 to 7, and mark and label the "
        "vertex of the parabola (compute it)."
    ),
    "math_intersection_two_lines": (
        "Plot the two lines y = 2x + 1 and y = -x + 7 for x from 0 to 5, and "
        "mark and label their intersection point (solve for it)."
    ),
    "math_cumulative_sum": (
        "Draw a bar chart of monthly savings: Jan=100, Feb=150, Mar=120, "
        "Apr=200, with a line overlay showing the cumulative total savings."
    ),
    "math_correlation_scatter": (
        "Make a scatter plot of these (x, y) pairs: (1,3), (2,5), (3,4), "
        "(4,8), (5,9), and label the plot with the computed correlation "
        "coefficient."
    ),
    "math_exponential_growth": (
        "Plot y = 100 * (1.1)^x for x from 0 to 20 (exponential growth), and "
        "mark and label the point where the value doubles to 200."
    ),
    "math_three_way_pie": (
        "Draw a pie chart of a budget: Rent 40%, Food 25%, Savings 20%, "
        "Other 15%."
    ),
    "math_negative_bar": (
        "Make a bar chart of quarterly profit/loss in thousands: Q1=+50, "
        "Q2=-20, Q3=+35, Q4=-10, with bars below zero shown distinctly."
    ),
    "math_dashed_reference_line": (
        "Plot y = sin(x) for x from 0 to 2*pi, and draw a dashed horizontal "
        "reference line at y = 0.5, labeled."
    ),
    "math_piecewise_three": (
        "Graph the piecewise function: y = -x for x < -1, y = 1 for -1 <= x < "
        "1, y = x for x >= 1, over x in [-3, 3]."
    ),
    "math_system_of_inequalities": (
        "Shade the region on a coordinate plane where y > x AND y < 5, for x "
        "and y from -2 to 8."
    ),
    "math_mean_median_marked": (
        "Plot these data points on a number line: 2, 4, 4, 5, 7, 9, 20, and "
        "mark and label both the mean and median (compute them)."
    ),
    "math_percentage_increase": (
        "Make a bar chart comparing 'Before' (50) and 'After' (75) values, "
        "and label the percentage increase (compute it)."
    ),
    "math_trig_unit_values": (
        "Plot y = sin(x) and y = cos(x) for x from 0 to pi/2, and mark the "
        "point where they are equal, labeled with its coordinates."
    ),
    "math_polynomial_degree3": (
        "Plot y = x^3 - 6*x^2 + 11*x - 6 for x from -1 to 5, and mark and "
        "label all three real roots (use np.roots)."
    ),
    # -- tables --
    "table_with_negative": (
        "Make a table of profit/loss by quarter: Q1 +50, Q2 -20, Q3 +35, "
        "Q4 -10. Columns: Quarter, Profit/Loss."
    ),
    "table_pass_fail": (
        "Make a table of exam results: Alice Pass, Bob Fail, Carol Pass, Dave "
        "Pass. Columns: Name, Result."
    ),
    "table_periodic_like": (
        "Make a table of 4 chemical elements: Hydrogen (H, 1), Helium (He, 2), "
        "Lithium (Li, 3), Carbon (C, 6). Columns: Name, Symbol, Atomic Number."
    ),
    "table_schedule": (
        "Make a table of a weekly schedule: Monday Math/Art, Tuesday "
        "Science/Music, Wednesday History/Gym. Columns: Day, Morning, "
        "Afternoon."
    ),
    # -- puzzles / abstract scenes (IconQA/AI2D style) --
    "puzzle_odd_one_out": (
        "Draw 4 shapes in a row: three circles and one square, all the same "
        "size and color, with the odd one out (the square) circled in red."
    ),
    "puzzle_number_sequence": (
        "Draw a sequence on a number line: 2, 4, 6, 8, with a question mark "
        "where the next number (10) would go."
    ),
    "puzzle_symmetry": (
        "Draw a right triangle on the left of a vertical dashed line, and its "
        "mirror image reflected across that line on the right."
    ),
    "puzzle_rotation": (
        "Draw a right triangle, and next to it the same triangle rotated 90 "
        "degrees, with both labeled."
    ),
    "scene_size_comparison": (
        "Draw three circles of different sizes in a row, labeled 'Small', "
        "'Medium', and 'Large'."
    ),
    "scene_overlap_count": (
        "Draw 3 overlapping blue circles and 2 overlapping red circles, "
        "scattered so some circles of the same color overlap each other."
    ),
    "scene_arrow_directions": (
        "Draw 4 arrows from a common center point: one pointing up labeled "
        "'N', one down labeled 'S', one right labeled 'E', one left labeled "
        "'W'."
    ),
    "puzzle_analogy_size": (
        "Draw an analogy puzzle: a small circle next to a large circle, then a "
        "small square next to a large square, then a small triangle next to a "
        "question mark representing what comes next."
    ),
    "puzzle_counting_sides": (
        "Draw 4 shapes side by side: a triangle, a square, a pentagon, and a "
        "hexagon, each labeled with its number of sides."
    ),
    "scene_size_order": (
        "Draw four circles in a row with increasing size from left to right, "
        "unlabeled."
    ),
    "puzzle_complete_the_grid": (
        "Draw a 2x2 grid of cells where three cells contain a circle and the "
        "fourth (bottom-right) contains a question mark instead."
    ),
    # -- science figures (ScienceQA style) --
    "science_orbit": (
        "Draw a simple diagram of a planet orbiting the sun: a small circle "
        "for the sun at the center, a circular dashed orbit path of radius 4, "
        "and a small circle for the planet on the orbit, with the radius "
        "labeled."
    ),
    "science_lever": (
        "Draw a lever (a horizontal bar) resting on a triangular fulcrum at "
        "its center, with the distance from the fulcrum to each end labeled."
    ),
    "science_ph_scale": (
        "Draw a pH scale number line from 0 to 14, labeling 0-6 as 'Acidic', 7 "
        "as 'Neutral', and 8-14 as 'Basic', with a marker at pH 3 labeled "
        "'Lemon Juice'."
    ),
    "science_density_column": (
        "Draw a density column diagram: a rectangular container with three "
        "horizontal layers from bottom to top labeled 'Honey', 'Water', and "
        "'Oil'."
    ),
    "science_periodic_trend": (
        "Make a bar chart of atomic radius (in picometers) for four elements: "
        "Lithium 152, Sodium 186, Potassium 227, Rubidium 248."
    ),
    # -- fourth batch: topping up to ~200 --
    "diagram_retry_loop": (
        "Draw a flowchart: 'Attempt Connection', then check 'Connected?' -- if "
        "no, go back to 'Attempt Connection'; if yes, go to 'Send Data'."
    ),
    "graph_tree_three_children": (
        "Draw a tree graph: Root connects to ChildA, ChildB, and ChildC (three "
        "children from one root)."
    ),
    "circuit_battery_switch_capacitor": (
        "Draw a circuit with a battery, a switch, and a capacitor, all in "
        "series."
    ),
    "vector_three_equal_angles": (
        "Draw three vectors from the origin, each of magnitude 5, spaced 120 "
        "degrees apart, all labeled."
    ),
    "geom_circle_central_angle": (
        "Draw a circle with a central angle of 90 degrees between two radii, "
        "and label the angle."
    ),
    "table_temperature": (
        "Make a table of daily high temperatures: Monday 72F, Tuesday 75F, "
        "Wednesday 68F, Thursday 70F. Columns: Day, High Temp."
    ),
    "molecule_propane": "Draw the molecular structure of propane.",
    "molecule_butane": "Show the molecular structure of butane.",
    "graph_negative_weights": (
        "Draw a weighted directed graph of temperature changes: Morning to "
        "Noon weight +5, Noon to Evening weight -3, Evening to Night weight -8."
    ),
    "circuit_led_resistor_battery_switch": (
        "Draw a circuit with a battery, a switch, a resistor, and an LED, all "
        "in series."
    ),
    "vector_force_at_60": (
        "Draw a single force vector of magnitude 8 at 60 degrees above the "
        "horizontal from the origin, labeled with its magnitude and angle."
    ),
    "table_animals_classification": (
        "Make a table classifying animals: Dog Mammal, Eagle Bird, Snake "
        "Reptile, Frog Amphibian. Columns: Animal, Class."
    ),
    "venn_two_word_categories": (
        "Draw a Venn diagram of 'Tools' (Hammer, Screwdriver) and 'Kitchen "
        "Items' (Spoon, Knife), with Knife in the overlap since it can be both "
        "a tool and a kitchen item in this puzzle."
    ),
    "scene_pattern_colors": (
        "Draw a row of 6 shapes alternating between a red circle and a blue "
        "square."
    ),
    "math_average_line": (
        "Plot these monthly sales values: Jan=80, Feb=95, Mar=70, Apr=110, "
        "May=90, and draw a horizontal dashed line at the average value, "
        "labeled."
    ),
}
