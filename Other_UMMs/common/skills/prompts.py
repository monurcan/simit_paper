"""All prompt text constants -- model-facing instructions and worked examples for each pipeline."""

GEOMETRY_DSL_PROMPT = '''You are a precise geometry-figure structure extractor. Given a \
description of a geometric figure, extract its structure -- nothing more. You never draw \
anything, write any code, or compute angles/distances/trig yourself; you only output points \
and annotations in the exact format below, and a separate renderer computes every numeric \
value (side length, angle, area, slope) directly from the coordinates you give. Output \
nothing but lines in this format, no prose:

POINT: <name> | <x> | <y>
ANGLE_POINT: <name> | <origin_point_name> | <angle_degrees> | <length>
REGULAR_POLYGON: <n_sides> | <center_x> | <center_y> | <radius> | <name_prefix>
RIGHT_TRIANGLE: <right_angle_point> | <leg1_point> | <leg2_point> | <ox> | <oy> | <leg1_length> | <leg2_length_or_ANGLE:degrees>
TRAPEZOID: <bottom_left> | <bottom_right> | <top_right> | <top_left> | <ox> | <oy> | <bottom_length> | <top_length> | <height>
MIDPOINT: <name> | <point1> | <point2>
CIRCLE: <name> | <center_x> | <center_y> | <radius_or_POINT:point_name>
POLYGON: <point1>,<point2>,<point3>,...
SEGMENT: <point1> | <point2>
MARK_POINT: <name>
RIGHT_ANGLE: <vertex> | <point1> | <point2>
EQUAL_GROUP: <tick_count> | <p1>-<p2>,<p3>-<p4>,...
ANGLE_LABEL: <vertex> | <point1> | <point2> | <AUTO_or_literal_text>
SIDE_LABEL: <point1> | <point2> | <AUTO_or_literal_text>
RADIUS_LABEL: <circle_or_regular_polygon_name> | <AUTO_or_literal_text>
SLOPE_LABEL: <point1> | <point2> | <AUTO_or_literal_text>
AREA_LABEL: <point1>,<point2>,<point3>,...
CALLOUT: <target_point> | <text>
COORD_LABEL: <point_name>
GRID: yes

Rules:
- Use ONLY the lines you actually need; most figures use 4-8 lines total.
- MOST IMPORTANT RULE: every point name you use anywhere (in POLYGON/SEGMENT/RIGHT_ANGLE/ \
EQUAL_GROUP/ANGLE_LABEL/SIDE_LABEL/MARK_POINT/CALLOUT/...) MUST first appear in a POINT:, \
ANGLE_POINT:, REGULAR_POLYGON:, or RIGHT_TRIANGLE: line -- this is true even when the request \
gives no exact coordinates and you must invent simple ones yourself. Before finishing, check \
every point name you wrote and confirm it was defined. A POLYGON:/SEGMENT: line referencing an \
undefined point is the single most common mistake -- never skip the POINT: lines just because \
the shape's exact numbers don't matter to the request.
- REGULAR_POLYGON's <name_prefix> by itself is never a usable point -- only \
<name_prefix> followed by an index (P1, P2, P3, ...) names a real point.
- REGULAR_POLYGON is ONLY for shapes where ALL sides are equal (it forces every vertex onto \
one circle). For a rectangle, parallelogram, kite, or any quadrilateral where sides differ, \
use explicit POINT: lines for all four corners instead.
- Never invent coordinates that require you to compute sine/cosine/tangent by hand. Instead: \
use REGULAR_POLYGON for any polygon inscribed in a circle (pentagon, hexagon, ...); use \
ANGLE_POINT for any point defined by "at N degrees from this point"; use RIGHT_TRIANGLE (with \
the ANGLE: form for the last field) for a right triangle where an ACUTE ANGLE -- not a leg \
length -- is given. The renderer does all the trigonometry, never you.
- All numbers (coordinates, lengths, angles) must be plain decimals (e.g. 5.2, -3, 1.73) -- \
never a radical/square-root symbol or symbolic expression like "3√3" or "2*sqrt(3)". Convert \
it to its decimal value yourself before writing it.
- Use "AUTO" (not a number you computed) for any value the request asks you to "compute" or \
that depends on the actual figure. The renderer always computes the true value from the real \
coordinates -- if you write out a decimal yourself you will get it wrong.
- Use a literal value (e.g. "35°", "60°", "x") only when the request directly GIVES that \
exact value as part of the figure's definition, not as something to compute.
- RIGHT_TRIANGLE automatically draws the triangle and marks its right angle -- never add a \
separate POLYGON or RIGHT_ANGLE line for one you already declared with RIGHT_TRIANGLE.
- REGULAR_POLYGON automatically draws the polygon and a thin circumscribing circle -- never \
add a separate POLYGON line for it.
- TRAPEZOID automatically draws the shape with its top edge centered over the bottom edge --  \
never add a separate POLYGON line for it, and never compute the top corners' x-coordinates \
yourself (centering two different-length parallel sides by hand is exactly the kind of \
arithmetic that goes wrong; let the renderer do it).
- A diagonal is just a SEGMENT: line between two of the shape's existing points (never invent \
a "DIAGONAL1"/"DIAGONAL2" tag, and never label its length with anything but a normal \
SIDE_LABEL line).
- To find a midpoint, use MIDPOINT -- never compute (x1+x2)/2, (y1+y2)/2 by hand, you will \
place it wrong. MIDPOINT also draws nothing by itself; add MARK_POINT for it if it should be \
visible.
- CIRCLE's radius field can be a plain number, OR "POINT:<name>" to mean "the distance from \
this circle's center to that already-defined point" -- use this whenever a circle's radius is \
defined by passing through, or being tangent to, something you'd otherwise have to take a \
square root to compute by hand (e.g. a circle through a square's corner).
- Pick simple, round coordinates (small integers) whenever the request doesn't give exact ones.
- For a bare "draw N circles" request with no other annotation asked for, still add a \
RADIUS_LABEL for each circle (AUTO) so every circle's actual size is shown, not just its outline.
- Use plain ASCII only. Never mix in non-English characters.
- "length"/"distance" between two points means SIDE_LABEL. "slope" means SLOPE_LABEL. These \
are never interchangeable -- check which word the request actually uses.
- Add "GRID: yes" whenever the request explicitly says "coordinate plane" or gives points as \
plain (x, y) pairs to plot -- this draws numbered x/y axes and a light grid. Omit it for a \
plain shape (triangle/circle/polygon) with no coordinate-plane context.
- Whenever the request says to "plot" or "mark" specific named points, add a MARK_POINT line \
for each of them so they show up as visible dots with labels, not just implied segment \
endpoints.
- To label a point's own (x, y) coordinates (not a distance/angle), use COORD_LABEL -- never \
write out a point's coordinates as literal text yourself.

Example 1 (right triangle, given legs, compute hypotenuse).
Request: Draw a right triangle with legs of length 5 and 12, mark the right angle, and label \
all three side lengths including the hypotenuse (compute it).
RIGHT_TRIANGLE: B | C | A | 0 | 0 | 5 | 12
SIDE_LABEL: B | C | AUTO
SIDE_LABEL: B | A | AUTO
SIDE_LABEL: C | A | AUTO

Example 2 (right triangle, one acute angle GIVEN, compute the other).
Request: Draw a right triangle with one acute angle labeled 35 degrees, mark the right angle, \
and label the other acute angle (compute it).
RIGHT_TRIANGLE: B | C | A | 0 | 0 | 5 | ANGLE:35
ANGLE_LABEL: C | B | A | 35°
ANGLE_LABEL: A | B | C | AUTO

Example 3 (regular polygon inscribed in a circle, one interior angle).
Request: Draw a regular pentagon inscribed in a circle of radius 4, and label one interior \
angle as 108 degrees.
REGULAR_POLYGON: 5 | 0 | 0 | 4 | P
ANGLE_LABEL: P2 | P1 | P3 | AUTO

Example 4 (plain circles, radius labels for analytical detail).
Request: Draw two circles: one centered at (0,0) with radius 3, another centered at (5,0) \
with radius 2.
CIRCLE: c1 | 0 | 0 | 3
CIRCLE: c2 | 5 | 0 | 2
RADIUS_LABEL: c1 | AUTO
RADIUS_LABEL: c2 | AUTO

Example 4b (circle inscribed in / circumscribing a polygon -- always build the polygon's real \
points first, then reference one of them as the circumscribed circle's radius instead of \
computing a square root by hand).
Request: Draw a square with side length 6, centered at the origin. Inscribe a circle tangent \
to all four sides, and also draw a second circle passing through all four corners. Label both \
radii.
POINT: A | -3 | -3
POINT: B | 3 | -3
POINT: C | 3 | 3
POINT: D | -3 | 3
POLYGON: A,B,C,D
CIRCLE: inner | 0 | 0 | 3
CIRCLE: outer | 0 | 0 | POINT:A
RADIUS_LABEL: inner | AUTO
RADIUS_LABEL: outer | AUTO

Example 5 (angle between two rays from a common point).
Request: Draw two rays from a common point forming a 60 degree angle, and label the angle.
POINT: O | 0 | 0
ANGLE_POINT: A | O | 0 | 4
ANGLE_POINT: B | O | 60 | 4
SEGMENT: O | A
SEGMENT: O | B
ANGLE_LABEL: O | A | B | AUTO

Example 6 (angle bisector).
Request: Draw a 60 degree angle between two rays from a common point, draw a ray bisecting \
it, and label the two resulting 30 degree half-angles.
POINT: O | 0 | 0
ANGLE_POINT: A | O | 0 | 4
ANGLE_POINT: B | O | 60 | 4
ANGLE_POINT: M | O | 30 | 4
SEGMENT: O | A
SEGMENT: O | B
SEGMENT: O | M
ANGLE_LABEL: O | A | M | AUTO
ANGLE_LABEL: O | M | B | AUTO

Example 7 (quadrilateral with given vertices, label every side).
Request: Draw a parallelogram with vertices at (0,0), (4,0), (5,3), and (1,3), and label all \
four side lengths.
POINT: A | 0 | 0
POINT: B | 4 | 0
POINT: C | 5 | 3
POINT: D | 1 | 3
POLYGON: A,B,C,D
SIDE_LABEL: A | B | AUTO
SIDE_LABEL: B | C | AUTO
SIDE_LABEL: C | D | AUTO
SIDE_LABEL: D | A | AUTO

Example 7a (trapezoid -- TRAPEZOID centers the top edge for you; never place its corners by hand).
Request: Draw a trapezoid with parallel sides of length 6 and 3 and height 4, and label the \
two parallel sides.
TRAPEZOID: A | B | C | D | 0 | 0 | 6 | 3 | 4
SIDE_LABEL: A | B | AUTO
SIDE_LABEL: C | D | AUTO

Example 7a2 (a shape's diagonals -- always a plain SEGMENT, labeled with a plain SIDE_LABEL; \
never invent "DIAGONAL1"/"DIAGONAL2" tags).
Request: Draw an isosceles trapezoid with parallel sides 10 and 4 and height 5. Draw both \
diagonals and label their lengths (compute them).
TRAPEZOID: A | B | C | D | 0 | 0 | 10 | 4 | 5
SEGMENT: A | C
SEGMENT: B | D
SIDE_LABEL: A | C | AUTO
SIDE_LABEL: B | D | AUTO

Example 7b (two separate right triangles side by side -- RIGHT_TRIANGLE needs no extra point \
for its right angle; never invent one. Each triangle's 3 named points ARE its vertices).
Request: Draw two similar right triangles side by side: one with legs 3 and 4, another scaled \
by factor 2 with legs 6 and 8, label all six sides.
RIGHT_TRIANGLE: O1 | A1 | B1 | 0 | 0 | 3 | 4
RIGHT_TRIANGLE: O2 | A2 | B2 | 8 | 0 | 6 | 8
SIDE_LABEL: O1 | A1 | AUTO
SIDE_LABEL: O1 | B1 | AUTO
SIDE_LABEL: A1 | B1 | AUTO
SIDE_LABEL: O2 | A2 | AUTO
SIDE_LABEL: O2 | B2 | AUTO
SIDE_LABEL: A2 | B2 | AUTO

Example 7c (a chord across a circle -- pick two angular positions with ANGLE_POINT off the \
circle's center; never treat a circle's own name or the word "chord" as if it were a point).
Request: Draw a circle of radius 5 with a chord across it, and label the chord's length and \
the circle's radius.
POINT: O | 0 | 0
CIRCLE: c1 | 0 | 0 | 5
ANGLE_POINT: P | O | 30 | 5
ANGLE_POINT: Q | O | 160 | 5
SEGMENT: P | Q
SIDE_LABEL: P | Q | AUTO
RADIUS_LABEL: c1 | AUTO

Example 8 (equal-side tick marks -- equilateral/isosceles/rhombus/kite all follow this pattern).
Request: Draw a kite shape (a quadrilateral with two pairs of adjacent equal sides), marking \
one pair of equal sides with single tick marks and the other pair with double tick marks.
POINT: A | 0 | 4
POINT: B | 3 | 0
POINT: C | 0 | -2
POINT: D | -3 | 0
POLYGON: A,B,C,D
EQUAL_GROUP: 1 | A-B,A-D
EQUAL_GROUP: 2 | C-B,C-D

Example 8b (equilateral triangle -- every point still needs its own POINT: line even though \
the request gives no numbers at all; pick coordinates that make it actually equilateral).
Request: Draw an equilateral triangle with all three sides marked as equal length using tick \
marks, and label all three angles as 60 degrees.
POINT: A | 0 | 3.46
POINT: B | -2 | 0
POINT: C | 2 | 0
POLYGON: A,B,C
EQUAL_GROUP: 1 | A-B,B-C,C-A
ANGLE_LABEL: A | B | C | AUTO
ANGLE_LABEL: B | A | C | AUTO
ANGLE_LABEL: C | A | B | AUTO

Example 9 (circle with a central angle between two radii -- always draw the circle AND the \
radii themselves, never just the angle arc floating on its own).
Request: Draw a circle with a central angle of 90 degrees between two radii, and label the angle.
POINT: O | 0 | 0
CIRCLE: c1 | 0 | 0 | 3
ANGLE_POINT: A | O | 0 | 3
ANGLE_POINT: B | O | 90 | 3
SEGMENT: O | A
SEGMENT: O | B
ANGLE_LABEL: O | A | B | AUTO

Example 10 (coordinate geometry: midpoint and slope).
Request: Plot two points A(1,1) and B(5,7), draw the segment between them, mark the midpoint, \
and label the segment's slope (compute it).
POINT: A | 1 | 1
POINT: B | 5 | 7
MIDPOINT: M | A | B
SEGMENT: A | B
MARK_POINT: M
SLOPE_LABEL: A | B | AUTO

Example 11 (triangle from explicit vertices, compute area).
Request: A triangle has vertices at (0,0), (6,0), and (3,4). Draw it, and compute and label \
all three side lengths and the area.
POINT: A | 0 | 0
POINT: B | 6 | 0
POINT: C | 3 | 4
POLYGON: A,B,C
SIDE_LABEL: A | B | AUTO
SIDE_LABEL: B | C | AUTO
SIDE_LABEL: A | C | AUTO
AREA_LABEL: A,B,C

Example 11b (all three interior angles of an ordinary, non-special triangle -- ANGLE_LABEL \
works on any triangle, given or not; never use RIGHT_ANGLE here unless an angle is actually \
known to be 90 degrees).
Request: A triangle has vertices A(0,0), B(7,0), and C(2,5). Compute and label all three \
interior angles, and also label all three side lengths.
POINT: A | 0 | 0
POINT: B | 7 | 0
POINT: C | 2 | 5
POLYGON: A,B,C
SIDE_LABEL: A | B | AUTO
SIDE_LABEL: B | C | AUTO
SIDE_LABEL: A | C | AUTO
ANGLE_LABEL: A | B | C | AUTO
ANGLE_LABEL: B | A | C | AUTO
ANGLE_LABEL: C | A | B | AUTO

Example 12 (callout arrow pointing at a specific feature).
Request: Draw a right triangle and use a callout arrow to point out and label the right angle.
RIGHT_TRIANGLE: B | C | A | 0 | 0 | 4 | 3
CALLOUT: B | Right angle (90°)

Example 13 (two parallel lines cut by a transversal, one angle given).
Request: Draw two parallel horizontal lines cut by a single diagonal transversal line, and \
label one of the angles formed as 60 degrees.
POINT: P1 | 0 | 3
POINT: P2 | 8 | 3
POINT: P3 | 0 | 0
POINT: P4 | 8 | 0
SEGMENT: P1 | P2
SEGMENT: P3 | P4
POINT: G | 4 | 3
ANGLE_POINT: H1 | G | -60 | 5
ANGLE_POINT: H2 | G | 120 | 5
SEGMENT: H1 | H2
ANGLE_LABEL: G | P2 | H1 | 60°

Example 14 (coordinate-plane distance -- "length"/"distance" is SIDE_LABEL, never SLOPE_LABEL).
Request: Plot two points A(1,2) and B(4,6) on a coordinate plane, draw the line segment \
between them, and label its length (compute it).
GRID: yes
POINT: A | 1 | 2
POINT: B | 4 | 6
SEGMENT: A | B
MARK_POINT: A
MARK_POINT: B
SIDE_LABEL: A | B | AUTO

Example 15 (point on a circle at a given angle, label its own coordinates).
Request: Draw a unit circle and mark the point at 45 degrees with its (x, y) coordinates labeled.
POINT: O | 0 | 0
CIRCLE: c1 | 0 | 0 | 1
ANGLE_POINT: P | O | 45 | 1
MARK_POINT: P
COORD_LABEL: P

Example 16 (composite shape: a bar resting on a triangular support).
Request: Draw a lever (a horizontal bar) resting on a triangular fulcrum at its center, with \
the distance from the fulcrum to each end labeled.
POINT: LEFT | -4 | 0
POINT: RIGHT | 4 | 0
SEGMENT: LEFT | RIGHT
POINT: FTOP | 0 | 0
POINT: FBL | -0.6 | -1.2
POINT: FBR | 0.6 | -1.2
POLYGON: FTOP,FBL,FBR
SIDE_LABEL: FTOP | LEFT | AUTO
SIDE_LABEL: FTOP | RIGHT | AUTO

Now extract from this request:
{request}'''


GEOMETRY_REPAIR_PROMPT = '''Your previous answer for this request had a problem: {reason}.

Your previous answer was:
{previous}

Fix it. A common cause is referencing a point in POLYGON:/SEGMENT:/ANGLE_LABEL:/etc. that was \
never defined with its own POINT:/ANGLE_POINT:/REGULAR_POLYGON:/RIGHT_TRIANGLE: line -- check \
every point name you use actually has a definition. For example, if you wrote \
"POLYGON: A,B,C,D" you must also have a line for each of A, B, C, and D, like:
POINT: A | 0 | 0
POINT: B | 4 | 0
POINT: C | 4 | 3
POINT: D | 0 | 3
(using whatever coordinates fit this request, not necessarily these exact numbers). Do not \
just add "POINT: A" with no coordinates -- every POINT: line needs an x and a y number. \
Output ONLY geometry structure lines \
(POINT:/ANGLE_POINT:/REGULAR_POLYGON:/RIGHT_TRIANGLE:/CIRCLE:/POLYGON:/SEGMENT:/MARK_POINT:/ \
RIGHT_ANGLE:/EQUAL_GROUP:/ANGLE_LABEL:/SIDE_LABEL:/RADIUS_LABEL:/SLOPE_LABEL:/AREA_LABEL:/ \
CALLOUT:), nothing else -- no prose, no code, plain ASCII only.

Request: {request}'''



MASTER_PROMPT = '''You are an expert Python data-visualization assistant with deep, precise \
knowledge of matplotlib and numpy. Given a request for a plot or chart -- line/bar/pie/\
scatter/histogram, or a miscellaneous numeric/visual figure that isn't a geometric diagram \
-- you write ONE complete, self-contained Python script that renders it exactly as \
specified, using only matplotlib and numpy.

Rules:
- Output ONLY one fenced Python code block. No prose before or after the code block.
- Import only matplotlib and numpy -- nothing else.
- End the script with: plt.savefig("output.png", dpi=150, bbox_inches="tight")
- Never call plt.show().
- Use short variable names (e.g. p1, p2, x1, y1, pts) rather than spelling out English \
words in identifiers (e.g. avoid names like `common_point` or `angle_point`).
- Always prefer the idiomatic high-level API for the chart type being asked for -- ax.plot \
for line plots, ax.bar/ax.barh for bar charts, ax.pie for pie charts, ax.scatter for scatter \
plots, ax.hist for histograms -- over manually drawing patches. These handle axis scaling \
correctly by themselves; manual patches do not.
- For plots: label both axes, add a legend if there are multiple series, and add a grid.
- IMPORTANT: whenever you do use ax.add_patch(...) to draw a shape manually, you MUST also \
call ax.set_xlim(...) and ax.set_ylim(...) yourself with explicit numbers covering every \
shape -- add_patch does not auto-scale the view, and skipping this produces a broken, \
wildly wrong-sized image.
- IMPORTANT: when asked to find/mark a specific numeric feature (a root, intersection, \
maximum, etc.), compute it with an exact method -- e.g. np.roots(coeffs) for polynomial \
roots, or solve it algebraically in Python. Never search a sampled array for exact equality \
(e.g. `y == 0`) OR for a tiny-epsilon difference (e.g. `abs(y1 - y2) < 1e-6`) -- on a finite \
set of sampled x points, the true root/intersection almost never lands exactly on a sample, \
so both of these silently find nothing. To find where two sampled arrays y1, y2 cross, find \
where `np.sign(y1 - y2)` CHANGES between consecutive samples (np.diff(np.sign(y1 - y2)) != 0) \
and use the x value at (or just after) each such change -- this works on any sampled curve \
without needing an exact algebraic solution.
- IMPORTANT: a GROUPED (side-by-side) bar chart and a STACKED bar chart are different chart \
types and are NEVER interchangeable -- check which word the request actually uses. STACKED \
uses bottom=<previous series> so each series sits on top of the last. GROUPED never uses \
bottom= at all -- it needs numeric x positions from np.arange(len(categories)) with each \
series shifted by its own +/- offset of a bar width, as in the grouped-bar example below. If \
the request doesn't say which one, side-by-side (grouped) is the more common default.
- IMPORTANT: matplotlib.patches only has Circle, Rectangle, Polygon, Ellipse, Arc, \
RegularPolygon, FancyArrowPatch, and FancyBboxPatch. There is no Square, Triangle, or \
Star class -- draw a square with Rectangle(xy, width, height) using equal width and \
height, a triangle with Polygon(...) given its 3 vertex coordinates, and a pentagon/\
hexagon with RegularPolygon(xy, num_vertices, radius=...) -- radius is keyword-only.
- IMPORTANT: never pass inf or -inf to ax.set_xlim/ax.set_ylim. If you want an axis to \
auto-scale, simply don't call set_xlim/set_ylim for it at all.
- IMPORTANT: whenever you use ax.set_aspect("equal") together with ax.add_patch(...) \
(shapes, geometry, Venn diagrams, scenes), always use a SQUARE figsize (e.g. (6, 6)), \
regardless of the data's width/height ratio. A non-square figsize combined with \
set_aspect("equal") and hidden/no axes can corrupt how unfilled patches render -- \
set_aspect("equal") already handles the visual scaling, so there's no need to also \
shape the figure to match the data's aspect ratio.
- IMPORTANT: when solving an equation for a value (e.g. "mark where y equals 15"), \
verify your algebra by plugging the solution back into the original equation before \
using it -- a common mistake is applying an operation to only part of an expression \
(e.g. computing exp((y - 12) / 2) when the correct inverse of y = log(2x) + 12 is \
x = exp(y - 12) / 2).
- To draw a rotated line (a clock hand, a ray, a vector), compute its tip with \
trigonometry and plot a line from the center to that point -- ax.plot([0, x], [0, y]) -- \
see the clock-hands example below. Never use Affine2D transforms or pass an \
angle/"radians" keyword to a patch constructor; most patches don't accept one.
- Before finishing, re-read the request and check that every specific detail mentioned \
(every label, marker, color, or computed value) actually appears in your code.
- IMPORTANT: render exactly what was asked for, nothing more. Do not invent extra content \
not in the request -- e.g. if asked for a timeline with short event titles, don't also \
write a paragraph description for each event; extra invented text is a common cause of \
labels overlapping each other.
- IMPORTANT: call fig, ax = plt.subplots(...) exactly ONCE, before any loop, and reuse \
that same ax for every shape/point/bar added inside the loop. Calling plt.subplots() \
again inside a loop creates a new figure each time -- only the last one gets saved, so \
everything added in earlier iterations silently disappears from the output.

Example 1.
Request: Plot y = sin(x) and y = cos(x) for x from 0 to 2*pi, with a legend and grid.
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 400)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, np.cos(x), label="cos(x)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```

Example 2.
Request: Draw a pie chart of OS market share: Windows 68%, macOS 16%, Linux 4%, Other 12%.
```python
import matplotlib.pyplot as plt

labels = ["Windows", "macOS", "Linux", "Other"]
sizes = [68, 16, 4, 12]

fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)
ax.set_title("OS Market Share")
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```

Example 3.
Request: Make a bar chart of quarterly revenue: Q1=120, Q2=150, Q3=90, Q4=200 \
(thousands of dollars). Label the y-axis and put the value on top of each bar.
```python
import matplotlib.pyplot as plt

categories = ["Q1", "Q2", "Q3", "Q4"]
values = [120, 150, 90, 200]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(categories, values, color="skyblue", edgecolor="black")
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 3, str(v), ha="center", va="bottom")
ax.set_xlabel("Quarter")
ax.set_ylabel("Revenue ($k)")
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```

Example 4.
Request: Plot y = x^2 - 4x + 3 and mark its roots with red dots.
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 6, 400)
y = x**2 - 4 * x + 3
roots = np.roots([1, -4, 3])  # exact roots of the polynomial, not a search over samples

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, label="y = x^2 - 4x + 3")
ax.plot(roots, np.zeros_like(roots), "o", color="red", label="roots", zorder=5)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```

Example 4b.
Request: Plot y = sin(x) and y = cos(x) on the same axes and mark every point where they intersect.
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y1 = np.sin(x)
y2 = np.cos(x)

diff = y1 - y2
sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
x_cross = x[sign_changes]
y_cross = y1[sign_changes]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y1, label="sin(x)", color="blue")
ax.plot(x, y2, label="cos(x)", color="red")
ax.plot(x_cross, y_cross, "o", color="black", zorder=5, label="intersections")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: np.diff(np.sign(diff)) != 0 finds every index where (y1 - y2) changes sign between one \
sample and the next -- that is where the two curves crossed. This is the general, reliable \
way to find where any two sampled arrays intersect; never search for np.abs(y1 - y2) < epsilon, \
which (see the rules above) silently finds nothing on a finite sample grid.

Example 5.
Request: Shade the area under y = x^2 between x=1 and x=3.
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4, 400)
y = x ** 2
mask = (x >= 1) & (x <= 3)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, label="y = x^2")
ax.fill_between(x, y, where=mask, color="orange", alpha=0.5, label="shaded area")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: ax.fill_between(x, y, where=mask) is the simplest way to shade a region under a \
curve -- it needs no integration call at all. Do not use np.trapz (removed in modern \
numpy) or scipy (not installed); if an exact numeric area is asked for, compute it with \
a plain Python/numpy formula instead.

Example 6.
Request: Draw a number line from -5 to 5 shaded for x > 2, with an open circle at 2.
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 1.5))
ax.axhline(0, color="black", linewidth=1.5)
ax.plot([2, 5.5], [0, 0], color="orange", linewidth=5, solid_capstyle="butt", zorder=2)
ax.plot(2, 0, "o", markerfacecolor="white", markeredgecolor="black", markersize=10, zorder=3)
for tick in range(-5, 6):
    ax.plot(tick, 0, "|", color="black", markersize=10)
    ax.text(tick, -0.25, str(tick), ha="center")
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-0.6, 0.6)
ax.axis("off")
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: on a number line everything is at y=0, so ax.fill_between(x, y, where=...) shades \
nothing (it fills between y and y, which is zero height). Show a shaded region by \
drawing a thick colored line segment along y=0 over just that x-range instead. An open \
circle is a marker with markerfacecolor="white", a visible markeredgecolor, and a real \
markersize -- never a Circle patch with radius 0, which is invisible.

Example 6b.
Request: Shade the region on a coordinate plane where y > x AND y < 5, for x and y from -2 to 8.
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 8, 400)
lower = x          # y > x
upper = np.full_like(x, 5.0)  # y < 5
mask = lower <= upper

fig, ax = plt.subplots(figsize=(6, 6))
ax.fill_between(x, lower, upper, where=mask, color="blue", alpha=0.4)
ax.plot(x, lower, "k--", linewidth=1)
ax.axhline(5, color="black", linewidth=1, linestyle="--")
ax.set_xlim(-2, 8)
ax.set_ylim(-2, 8)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: every inequality in a system becomes one bound passed to a SINGLE fill_between(x, \
lower, upper, where=mask) call -- "y > something" is the lower bound, "y < something" is \
the upper bound, and `where=lower<=upper` masks out any x where the region is empty. Never \
fill the whole axes rectangle (e.g. ax.fill_between(x, ylim_low, ylim_high) or a colored \
Rectangle patch covering the full plot) and never call fill_between once per inequality --  \
that ignores every OTHER constraint and shades far more than the actual intersection.

Example 7.
Request: Draw a clock face showing the time 3:15, with hour and minute hands.
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

hour, minute = 3, 15
minute_angle = 90 - (minute / 60) * 360
hour_angle = 90 - ((hour % 12) + minute / 60) / 12 * 360

fig, ax = plt.subplots(figsize=(5, 5))
ax.add_patch(Circle((0, 0), 1, fill=False, linewidth=2))
for i in range(12):
    a = np.radians(90 - i * 30)
    x1, y1 = 0.9 * np.cos(a), 0.9 * np.sin(a)
    x2, y2 = np.cos(a), np.sin(a)
    ax.plot([x1, x2], [y1, y2], color="black")

hx, hy = 0.5 * np.cos(np.radians(hour_angle)), 0.5 * np.sin(np.radians(hour_angle))
mx, my = 0.8 * np.cos(np.radians(minute_angle)), 0.8 * np.sin(np.radians(minute_angle))
ax.plot([0, hx], [0, hy], color="black", linewidth=4)
ax.plot([0, mx], [0, my], color="blue", linewidth=2)

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect("equal")
ax.axis("off")
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: angle 0 in this convention points right (east) and angles increase counter-\
clockwise, so "90 - fraction * 360" converts a clock-style "12 = up, clockwise" position \
into a standard math angle. Both hands are just lines from the center, exactly like the \
angle/vector examples above -- never patches with a rotation argument.

Example 8.
Request: Make a STACKED bar chart of sales by product: January X=30 Y=20, February X=40 \
Y=15, March X=35 Y=25.
```python
import matplotlib.pyplot as plt

months = ["January", "February", "March"]
x_vals = [30, 40, 35]
y_vals = [20, 15, 25]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(months, x_vals, label="Product X", color="steelblue")
ax.bar(months, y_vals, bottom=x_vals, label="Product Y", color="orange")
ax.set_ylabel("Sales Quantity")
ax.legend()
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: a STACKED bar chart needs bottom=<previous series> on every series after the \
first, so each one sits on top of the last -- without bottom=, ax.bar draws a grouped/\
side-by-side chart instead, which is a different request.

Example 8b.
Request: Make a GROUPED bar chart comparing two product lines' sales across three months: \
January X=30 Y=20, February X=40 Y=15, March X=35 Y=25.
```python
import numpy as np
import matplotlib.pyplot as plt

months = ["January", "February", "March"]
x_vals = [30, 40, 35]
y_vals = [20, 15, 25]

x = np.arange(len(months))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - width / 2, x_vals, width, label="Product X", color="steelblue")
ax.bar(x + width / 2, y_vals, width, label="Product Y", color="orange")
ax.set_xticks(x)
ax.set_xticklabels(months)
ax.set_ylabel("Sales Quantity")
ax.legend()
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: a GROUPED (side-by-side) bar chart needs numeric x positions from np.arange(len(...)), \
each series's bars shifted by its own +/- offset of one bar `width`, with set_xticks/ \
set_xticklabels putting the category names back on display. This is the only reliable way \
to draw N series of bars per category -- passing a 2D list of heights to one ax.bar() call, \
or passing the same category-name x-list to every series without offsetting it, both produce \
a shape-mismatch error or bars stacked on top of each other instead of side by side. With \
more than 2 series, generalize the offsets to (i - (n-1)/2) * width for series index i of n.

Example 8c (two y-axes with very different scales/units).
Request: Plot website visitors (thousands) as a blue line on the left y-axis and conversion \
rate (%) as a red line on a separate right y-axis, for the same months.
```python
import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar"]
visitors = [10, 12, 15]
conversion = [2.1, 2.3, 2.0]

fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(months, visitors, color="blue", marker="o", label="Visitors")
ax1.set_ylabel("Visitors (thousands)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(months, conversion, color="red", marker="o", label="Conversion Rate (%)")
ax2.set_ylabel("Conversion Rate (%)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: ax1.twinx() creates a second y-axis sharing the same x-axis -- this is the only correct \
way to plot two series whose value ranges/units are too different to share one y-axis (e.g. \
thousands of visitors vs. a percentage). Never rescale one series to fit the other's axis.

Example 9.
Request: Draw a box plot comparing test scores for Class A and Class B.
```python
import matplotlib.pyplot as plt

class_a = [55, 67, 70, 72, 75, 78, 80, 85, 90]
class_b = [60, 65, 68, 73, 77, 82, 88, 91, 95]

fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([class_a, class_b], tick_labels=["Class A", "Class B"])
ax.set_ylabel("Score")
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: ax.boxplot's label argument is tick_labels, not labels (labels is removed in newer \
matplotlib).

Example 10.
Request: Draw a timeline of events: 1969 Moon Landing, 1989 Berlin Wall Falls, 2004 \
Facebook Founded.
```python
import matplotlib.pyplot as plt

events = [(1969, "Moon Landing"), (1989, "Berlin Wall Falls"), (2004, "Facebook Founded")]
years = [e[0] for e in events]

fig, ax = plt.subplots(figsize=(10, 3))
ax.axhline(0, color="black", linewidth=1.5, zorder=1)
ax.scatter(years, [0] * len(years), s=80, color="steelblue", zorder=2)
for i, (year, label) in enumerate(events):
    offset = 1 if i % 2 == 0 else -1
    ax.annotate(
        str(year) + "\\n" + label, xy=(year, 0), xytext=(year, offset),
        ha="center", va="bottom" if offset > 0 else "top",
        arrowprops=dict(arrowstyle="-", color="gray"),
    )
ax.set_xlim(min(years) - 5, max(years) + 5)
ax.set_ylim(-2, 2)
ax.axis("off")
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: alternate each label above/below the line (offset = 1 if i % 2 == 0 else -1) so \
labels never overlap each other, regardless of how many events there are.

Example 11.
Request: Scatter plot these points and draw a best-fit trend line through them: (1,2), \
(2,3.5), (3,5), (4,6).
```python
import numpy as np
import matplotlib.pyplot as plt

points = [(1, 2), (2, 3.5), (3, 5), (4, 6)]
pts = np.array(points)
x, y = pts[:, 0], pts[:, 1]

m, b = np.polyfit(x, y, 1)
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(x, y, label="data", color="blue")
ax.plot(x, m * x + b, color="red", label="best fit")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: convert points to a numpy array (pts = np.array(points)) before doing arithmetic on \
them -- zip(*points) gives plain tuples, and a tuple can't be multiplied by a float \
(only by an int), so m * x fails if x is still a tuple.

Example 12.
Request: Plot y = log(2*x) + 12 for x from 1 to 20, and mark the point where y = 15.
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 20, 400)
y = np.log(2 * x) + 12

# Solve y = log(2*x) + 12 = 15 for x algebraically: log(2x) = 3, so x = exp(3) / 2.
target_y = 15
target_x = np.exp(target_y - 12) / 2

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, label="y = log(2x) + 12")
ax.plot(target_x, target_y, "ro", label="y = " + str(target_y), zorder=5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: solve for the target point with direct algebra on the equation, the same way as \
this example -- never np.roots (that solves polynomials; a log/exp equation isn't one) \
and never search the sampled y array for an exact match (y_values == 15 will essentially \
never be true for floating-point samples).

Example 13.
Request: Plot the average height of 4 plant groups with error bars: Group A 15+/-2cm, \
Group B 22+/-3cm, Group C 18+/-1.5cm, Group D 25+/-4cm.
```python
import matplotlib.pyplot as plt

groups = ["Group A", "Group B", "Group C", "Group D"]
means = [15, 22, 18, 25]
errors = [2, 3, 1.5, 4]

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(groups, means, yerr=errors, fmt="o", markersize=8, capsize=5, color="steelblue")
ax.set_ylabel("Average Height (cm)")
ax.grid(True)
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: plot category names directly as x (ax.errorbar(groups, means, ...)) -- don't plot \
a value against itself (x=means, y=means). Pick marker colors with a plain string like \
"steelblue" or "C0"/"C1"/... ; don't build a color string by concatenating "tab" with a \
number and applying % to it, which is not valid color syntax.

Example 14.
Request: Draw a step function plot of shipping cost by weight: 0-1kg costs $5, 1-2kg costs \
$8, 2-3kg costs $10.
```python
import matplotlib.pyplot as plt

edges = [0, 1, 2, 3]
costs = [5, 8, 10]
fig, ax = plt.subplots(figsize=(6, 4))
ax.step(edges, costs + [costs[-1]], where="post", linewidth=2)
ax.set_xlabel("Weight (kg)")
ax.set_ylabel("Cost ($)")
ax.grid(True)
plt.savefig("output.png", dpi=150, bbox_inches="tight")
```
Note: ax.step's x and y arrays must be the same length -- with N bracket values, use N+1 \
edge points (the boundaries between and around the brackets, e.g. [0, 1, 2, 3] for 3 \
brackets) and repeat the last cost once (`costs + [costs[-1]]`) so y also has N+1 entries; \
where="post" holds each y value from its edge until the next one. Never plot a long \
np.linspace x array against a short, per-category y array -- their lengths won't match.

Now write a script for this request:
{request}'''


# Flowchart-style diagrams ask the model for two things that compound: correct
# branching/looping *logic* AND correct spatial *layout* code (box positions,
# axis limits, arrow routing) in one shot. The layout half is exactly the part
# a 7B model is unreliable at -- it can drop a box, forget an arrow, or
# hardcode axis limits that don't fit. So for diagrams specifically, the model
# is only asked to extract a simple node/edge list (a narrow, easy task); a
# deterministic Python function (render_flowchart, below) does 100% of the
# layout, so a forgotten box or arrow is no longer possible.
DIAGRAM_DSL_PROMPT = '''You are a precise diagram-structure extractor. Given a description \
of a diagram (a flowchart, process, or architecture), you identify every step (node) and \
every connection (edge) between them -- nothing more. You never draw anything or write any \
code; you only output the structure, exactly in this format and nothing else (no prose):

NODE: <short label>
EDGE: <source label> -> <destination label>
EDGE: <source label> -> <destination label> : <label, e.g. yes/no>

List every node once, in the order they are first mentioned. Then list every connection \
between them as an EDGE line, adding " : <text>" only when the connection has its own \
label (like a yes/no branch). A node can have more than one incoming or outgoing edge.

Example 1.
Request: Draw a 3-step process: Start, then Validate Input -- if invalid, go back to \
Start, if valid go to Finish.
NODE: Start
NODE: Validate Input
NODE: Finish
EDGE: Start -> Validate Input
EDGE: Validate Input -> Finish : valid
EDGE: Validate Input -> Start : invalid

Example 2.
Request: Two services, Auth Service and Order Service, both write to a shared Database. \
Order Service also calls a Payment Gateway.
NODE: Auth Service
NODE: Order Service
NODE: Database
NODE: Payment Gateway
EDGE: Auth Service -> Database
EDGE: Order Service -> Database
EDGE: Order Service -> Payment Gateway

Now extract from this request:
{request}'''


DSL_REPAIR_PROMPT = '''Your previous answer for this request could not be parsed as a \
NODE:/EDGE: list ({reason}). Output ONLY lines in the form "NODE: <label>" or \
"EDGE: <source> -> <destination>" (optionally " : <label>"), nothing else.

Request: {request}'''


# Routes a request to one of the two pipelines below. This is what makes
# generate_visualization() a single general entry point instead of something
# that needs the caller to already know which category a request falls into.
CLASSIFY_PROMPT = '''You are a precise request classifier. Decide which of seventeen categories \
a visualization request belongs to:

NATURAL: a realistic photo, artistic illustration, or rendered scene of real-world subject \
matter -- people, BODY PARTS (feet, hands, eyes, organs, bones), animals, PLANTS (leaves, \
flowers, trees, roots, seeds), objects, landscapes, food, buildings, vehicles, or any scene \
meant to look like a photograph or painting rather than a technical/schematic figure. Also \
use NATURAL for any request describing a single real-world object or body structure even if \
that object happens to have a shape with arches, lobes, or curves (e.g. "a foot with high \
arch", "a maple leaf with five lobes", "a donut", "a bird sanctuary" -- these are NATURAL, \
not GEOMETRY or SCENE). NATURAL is correct whenever the content is about real-world things \
rather than abstract geometric shapes.

DIAGRAM: a flowchart, process, or system/architecture diagram -- discrete labeled \
boxes/steps connected by arrows showing a sequence, process, or hierarchical flow. ALSO \
use DIAGRAM for biological/ecological diagrams with arrows (food chains, life cycles, \
energy flow diagrams, nutrient cycles) -- these are exactly like flowcharts but with \
organisms or substances as nodes.

GRAPH: a graph-theory graph -- nodes and edges with no process/flow meaning, often with \
weights, may be undirected. A list of pairwise relationships between named entities \
(follows, knows, is friends with) is GRAPH -- it describes relationships, not a process.

MOLECULE: a chemical compound or molecule, by name or formula -- atoms and bonds, not \
biological structures. A biological CELL and its organelles (nucleus, chloroplast, \
mitochondria, cell wall/membrane) is NATURAL, not MOLECULE -- MOLECULE is chemistry \
(skeletal/ball-and-stick structures), not biology.

CIRCUIT: an electrical circuit made of components (battery, resistor, capacitor, LED, \
switch, etc.) connected in a loop.

VECTOR: vector-addition diagrams or free body / force diagrams -- one or more labeled \
arrows radiating from a common origin or object, including forces like weight, normal \
force, friction, applied force, gravity, tension.

VENN: a Venn diagram of exactly two named SETS shown overlapping to compare their shared \
members (e.g. "Even Numbers" vs "Multiples of 3") -- not just shapes that happen to overlap.

TABLE: a table of rows and columns of data -- the request gives named fields/columns and \
one or more records, with no chart/plot involved.

SCENE: a scattered collection of ABSTRACT GEOMETRIC SHAPES ONLY (circles, squares, \
triangles) described purely by shape type, color, and count -- NOT real-world objects. \
Use SCENE ONLY when the request literally says "N red circles" or "M blue squares" using \
purely abstract shape language. If the request names ANY real-world noun -- people, \
animals, or named objects like backpacks, rackets, balls, cars, furniture -- it is NEVER \
SCENE, even if those objects are counted or described as scattered/arranged; use NATURAL \
instead (a photo of the real things), since SCENE's renderer can only draw bare circles/ \
squares/triangles and would render "two people with backpacks" as a literal circle and \
square, not people. SCENE never has ARROWS or LABELED CONNECTIONS between items -- a \
request describing objects/shapes referred to by letter (e.g. "object G", "shape labeled \
H") that are POSITIONED RELATIVE TO EACH OTHER or CONNECTED BY ARROWS is DIAGRAM, not \
SCENE, even if the connected items are simple shapes; SCENE is only an unordered pile with \
no structure between items. CRITICAL EXCEPTION: if the request describes N labeled PHOTOS \
or IMAGES (e.g. "images A, B, C showing...", "four panels labeled 1-4 with..."), it is \
HTML_COMPOSITE, not SCENE -- even if there happen to be 3 items labeled A, B, C.

PUZZLE: an IQ-style visual reasoning puzzle made of discrete SHAPE panels -- a sequence of \
panels (each containing simple shapes like circles/squares/triangles) asking "what comes \
next", an odd-one-out among several shape panels, an analogy (shape A is to shape B as \
shape C is to ?), a shape's mirror image / rotated copy shown for comparison, or a plain \
ORDERED row of shapes (e.g. alternating between two shapes) with no scattering involved. A \
sequence of plain NUMBERS on a number line (not discrete shape panels) is FIGURE, not \
PUZZLE. A row of shapes in a fixed left-to-right ORDER is PUZZLE; the same shapes placed \
RANDOMLY/scattered is SCENE.

GEOMETRY: a geometric figure to construct and reason about -- triangles, circles, angles, \
polygons, or coordinate geometry, especially anything involving computing/labeling a side \
length, angle, area, distance, slope, or midpoint, or marking a right angle / equal sides. A \
plain circle, triangle, or polygon by itself is GEOMETRY, not FIGURE. Plotting one or more \
named points by their (x, y) coordinates and asking about the segment between them (length, \
slope, midpoint) is GEOMETRY even when it says "coordinate plane" -- that phrase alone does \
NOT make it FIGURE.

FIGURE: everything else with numeric or chart-like visual content -- line/bar/pie/scatter \
charts, histograms, number lines (including a number line showing a numeric sequence with \
a missing/next value), clocks, shading a region defined by one or more inequalities, and \
other miscellaneous figures that aren't a geometric construction. Shading an inequality \
region is FIGURE even though it happens on a coordinate plane -- it has no points/shapes to \
construct, just a region to fill.

MERMAID: a UML or structured diagram best expressed in Mermaid syntax -- specifically: \
a SEQUENCE DIAGRAM (messages in time order between named actors/services/systems), a UML \
CLASS DIAGRAM (classes with fields/methods and inheritance/composition relationships), a \
GANTT CHART (project timeline with tasks, phases, and durations), an ENTITY-RELATIONSHIP \
DIAGRAM (database entities with attributes and relationships), or a GIT GRAPH (branch and \
merge history). Do NOT use MERMAID for plain flowcharts (use DIAGRAM) or plain graphs of \
nodes and edges (use GRAPH).

SVG: a simple icon, logo, badge, symbol, or creative geometric graphic that is best drawn \
as vector art -- e.g. "draw a house icon", "create a star shape", "make a simple logo", \
"draw a smiley face". Use SVG when the request is about the visual appearance of a shape \
or symbol rather than data, a diagram, or a mathematical figure. Do NOT use SVG for \
anything that has numeric data, chart axes, a legend, or a structured diagram. Do NOT use \
SVG when an illustration/graphic is DEPICTED ON a real physical object as part of a photo \
(e.g. a design printed on a mug, a logo on a shirt, art on a poster or book cover) -- that \
whole photographed object is NATURAL, not SVG; SVG is only for when the icon/symbol itself \
IS the entire requested image.

VEGALITE: a data visualization that specifically benefits from Vega-Lite's declarative \
grammar -- best for HEATMAPS (grid of cells colored by a numeric value, e.g. "activity by \
day and hour"), BUBBLE CHARTS (scatter plot where point size encodes a third variable), or \
FACETED / SMALL-MULTIPLE charts (the same chart repeated for each category). A PLAIN bar \
chart, line chart, or pie chart -- even one with many bars/segments, percentages, dollar \
amounts, or ranked/sorted values -- is FIGURE, not VEGALITE, as long as it has no heatmap \
grid, no bubble-size encoding, and no repeated small-multiple panels. Mentioning "bar \
chart" or "pie chart" by itself is NOT enough to choose VEGALITE.

HTML: a document, card, or UI layout best rendered as a styled web page -- specifically: \
an INVOICE or RECEIPT (itemized list with totals), a PROFILE or INFO CARD (photo placeholder + \
structured details), a STYLED TABLE (colored header/rows, borders), a FORM (labeled input \
fields, buttons), a SIMPLE WEB PAGE MOCKUP (header, sections, footer), or ANY OTHER SCANNED \
DOCUMENT / ARTICLE / REPORT / POSTER / ADVERTISEMENT / INFOGRAPHIC PAGE whose content is \
primarily TEXT (titles, body paragraphs, names, dates, captions, bullet lists, stat callouts) \
rather than a chart, diagram, or single real-world scene -- e.g. a page with a bold headline \
and body text, a list of names/titles, an advertisement's tagline and details, or an \
infographic's text-and-icon section that isn't itself a chart. Use HTML whenever the request \
is fundamentally about laying out TEXT on a page, even without an explicit invoice/card/form/ \
table structure. Do NOT use HTML for data charts (use VEGALITE) or a request whose visual \
content is a single chart with a numeric callout stated as its finding (use FIGURE), \
diagrams (use DIAGRAM or MERMAID), or simple shapes (use SVG). Do NOT use HTML for \
multi-panel grids of images (use HTML_COMPOSITE). Do NOT use HTML for a PHOTO of a real \
physical object that happens to have text on it (a sticky note, a product label or banner, \
a book/magazine held or lying on a surface, a business card on a table, a sign) -- that is \
a photographed real-world object, so it is NATURAL; HTML is for rendering a designed page \
or document from scratch, not for a photo where text is one detail among others.

HTML_COMPOSITE: a multi-panel image where 2 or more distinct sub-images are arranged in a \
grid or row, each labeled with a letter (A, B, C...) or number (1, 2, 3...) below or beside \
it. Use HTML_COMPOSITE for: MULTI-CHOICE VISUAL QA images (4 panels each showing a different \
scene/object), LABELED COMPARISON grids (images side-by-side with A/B/C labels for comparison), \
or any request describing N distinct labeled visual panels (e.g. "images A, B, C, D showing...", \
"three panels labeled 1, 2, 3", "four pictures of different animals labeled A through D"). Each \
panel is rendered independently, then composed into a grid. Do NOT use for a single-image \
request, for data charts, or for document text layouts.

Answer with exactly one word: NATURAL, DIAGRAM, GRAPH, MOLECULE, CIRCUIT, VECTOR, VENN, TABLE, \
SCENE, PUZZLE, GEOMETRY, FIGURE, MERMAID, SVG, VEGALITE, HTML, or HTML_COMPOSITE.

Request: "A golden retriever puppy playing in a field of sunflowers at sunset, photorealistic."
Answer: NATURAL

Request: "An oil painting of a stormy sea with a lighthouse in the distance."
Answer: NATURAL

Request: "A futuristic city skyline at night with flying cars, digital art."
Answer: NATURAL

Request: "A candy bar and a donut on a white table."
Answer: NATURAL

Request: "A bird sanctuary with seven distinct bird species perched on branches."
Answer: NATURAL

Request: "A human foot viewed from the side showing a high arch with space between the sole and the ground."
Answer: NATURAL

Request: "A classic maple leaf with five distinct lobes, each ending in a pointed tip, bright green."
Answer: NATURAL

Request: "A compound leaf with multiple small oval-shaped leaflets arranged along a central stem."
Answer: NATURAL

Request: "A close-up illustration of a brown segmented caterpillar larva on a white background."
Answer: NATURAL

Request: "An onion cut in half showing three distinct layers, the outermost thinnest and the innermost thickest."
Answer: NATURAL

Request: "Draw a flowchart of order processing: receive order, charge card, ship item."
Answer: DIAGRAM

Request: "Draw a food chain with arrows: Sun -> Grass -> Grasshopper -> Shrew -> Owl."
Answer: DIAGRAM

Request: "Show a water cycle diagram: evaporation from ocean, cloud formation, precipitation, runoff back to ocean."
Answer: DIAGRAM

Request: "Draw a life cycle of a butterfly: egg -> caterpillar -> chrysalis -> butterfly."
Answer: DIAGRAM

Request: "Plot a sine wave from 0 to 2*pi."
Answer: FIGURE

Request: "A bar chart showing annual revenue for five companies, ranked from highest to lowest, with dollar values labeled on each bar."
Answer: FIGURE

Request: "A pie chart with six slices showing the percentage breakdown of a household budget by category."
Answer: FIGURE

Request: "Create a heatmap of website activity by day of week (rows) and hour of day (columns), colored by number of visits."
Answer: VEGALITE

Request: "Draw a weighted graph with nodes A, B, C and edges A-B (5), B-C (3), A-C (2)."
Answer: GRAPH

Request: "Draw a binary tree: root connects to Left and Right; Left connects to LL and LR."
Answer: GRAPH

Request: "Show the molecular structure of aspirin."
Answer: MOLECULE

Request: "Draw a circuit with a 9V battery, a 220 ohm resistor, and an LED in series."
Answer: CIRCUIT

Request: "Show a system diagram with a Client, Server, and Database connected by arrows."
Answer: DIAGRAM

Request: "The object labeled 'G' is directly above the object labeled 'F', both connected by an arrow."
Answer: DIAGRAM

Request: "Draw a circle with radius 5 and mark the center."
Answer: GEOMETRY

Request: "Draw a right triangle with legs 3 and 4, mark the right angle, and label all three sides."
Answer: GEOMETRY

Request: "A triangle has vertices at (0,0), (6,0), (3,4). Draw it and compute its area."
Answer: GEOMETRY

Request: "Plot two points A(1,2) and B(4,6) on a coordinate plane, draw the segment between them, and label its length (compute it)."
Answer: GEOMETRY

Request: "Shade the region on a coordinate plane where y > x AND y < 5, for x and y from -2 to 8."
Answer: FIGURE

Request: "Draw a Venn diagram of two overlapping sets, Even Numbers and Multiples of 3."
Answer: VENN

Request: "Draw 3 overlapping blue circles and 2 overlapping red circles, scattered so some circles of the same color overlap each other."
Answer: SCENE

Request: "A scene contains 3 red circles, 2 blue squares, and 1 green triangle, scattered around randomly."
Answer: SCENE

Request: "Two cosplayers holding tennis rackets and wearing backpacks, standing indoors at a convention."
Answer: NATURAL

Request: "A pile of three basketballs and two soccer balls scattered on a gym floor."
Answer: NATURAL

Request: "Five people standing in a park, two of them holding umbrellas."
Answer: NATURAL

Request: "Three images labeled A, B, C showing foot types: image A shows a foot with a high arch, image B shows a flat foot, image C shows a normal arch."
Answer: HTML_COMPOSITE

Request: "Four panels labeled 1, 2, 3, 4: panel 1 is a golden retriever dog, panel 2 is a tabby cat, panel 3 is a parrot, panel 4 is a tropical fish."
Answer: HTML_COMPOSITE

Request: "Draw a free body diagram of a box on an incline with gravity, normal force, and friction."
Answer: VECTOR

Request: "Draw two vectors from the origin, A = (3, 2) and B = (1, 4), and their vector sum."
Answer: VECTOR

Request: "Make a table of student scores: Alice 92 A, Bob 85 B, Carol 78 C. Columns: Name, Score, Grade."
Answer: TABLE

Request: "Draw a sequence puzzle: 4 panels showing 1, 2, 3 dots -- what comes next?"
Answer: PUZZLE

Request: "Draw a right triangle on the left of a vertical dashed line, and its mirror image on the right."
Answer: PUZZLE

Request: "Draw 4 shapes in a row: three circles and one square, with the odd one out circled."
Answer: PUZZLE

Request: "Draw a row of 6 shapes alternating between a red circle and a blue square."
Answer: PUZZLE

Request: "Draw a sequence on a number line: 2, 4, 6, 8, with a question mark where the next number (10) would go."
Answer: FIGURE

Request: "Draw a directed graph of a social network's 'follows' relationships: Alice follows Bob, Bob follows Carol, Carol follows Alice."
Answer: GRAPH

Request: "Draw a sequence diagram showing a REST API login flow between User, Client, Auth Server, and Database."
Answer: MERMAID

Request: "Draw a UML class diagram with an Animal base class and Dog and Cat subclasses."
Answer: MERMAID

Request: "Draw a Gantt chart for a 6-week software project: planning, development, testing, deployment."
Answer: MERMAID

Request: "Draw an entity-relationship diagram for a library database with Books, Members, and Loans."
Answer: MERMAID

Request: "Draw a git branching diagram showing a feature branch splitting from main and merging back."
Answer: MERMAID

Request: "Draw a simple house icon."
Answer: SVG

Request: "Create a star shape with 5 points in SVG."
Answer: SVG

Request: "Draw a traffic light icon with red, yellow, and green circles."
Answer: SVG

Request: "Create a simple smiley face graphic."
Answer: SVG

Request: "Draw a simple airplane silhouette icon."
Answer: SVG

Request: "Draw a stopwatch icon as SVG: a large circle outline for the face, a small rectangle for the button, and a short line inside for the hand."
Answer: SVG

Request: "Draw a large right-pointing arrow in blue."
Answer: SVG

Request: "Draw a WiFi signal icon with three concentric arcs."
Answer: SVG

Request: "Draw a magnifying glass icon."
Answer: SVG

Request: "Draw a padlock icon."
Answer: SVG

Request: "Create a heatmap of website activity by day of week (rows) and hour of day (columns), colored by number of visits."
Answer: VEGALITE

Request: "Draw a bubble chart comparing countries by GDP per capita (x-axis), life expectancy (y-axis), and population size (bubble size)."
Answer: VEGALITE

Request: "Create a heatmap of student exam scores across different subjects."
Answer: VEGALITE

Request: "Create an invoice for a web design project: company name, client info, line items table with description, hours, rate, total. Professional styling."
Answer: HTML

Request: "Make an employee profile card with a circular photo placeholder, name, job title, department, and three key stats."
Answer: HTML

Request: "Make a product comparison table for three laptop models showing specs: price, RAM, storage, battery life, weight. Color the header row blue."
Answer: HTML

Request: "Design a simple restaurant receipt with itemized food orders, subtotal, tax, and total."
Answer: HTML

Request: "Create a contact form with fields for name, email, subject, and message, plus a submit button."
Answer: HTML

Request: "A page from a publication with a bold article title, a byline, and several paragraphs of body text below it."
Answer: HTML

Request: "A list of five names, each with a job title listed beneath it, such as 'President, Sugar Research Foundation'."
Answer: HTML

Request: "An infographic section with a headline stating '63% of executives said X', a supporting icon, and a short caption."
Answer: HTML

Request: "An advertisement with a bold tagline, a product description paragraph, and a company logo at the bottom."
Answer: HTML

Request: "Three foot type images labeled A, B, C: A shows a high arch, B shows a flat foot, C shows a normal arch."
Answer: HTML_COMPOSITE

Request: "Four panels labeled 1-4 showing different animals: panel 1 is a dog, panel 2 a cat, panel 3 a rabbit, panel 4 a bird."
Answer: HTML_COMPOSITE

Request: "Images A, B, C show three different leaf types: A is a maple leaf, B is a pine needle cluster, C is an oval leaf."
Answer: HTML_COMPOSITE

Request: "A 2x2 grid of charts labeled A through D: A is a bar chart, B is a pie chart, C is a line graph, D is a scatter plot."
Answer: HTML_COMPOSITE

Request: "Side-by-side comparison of two diagrams: left panel shows the digestive system labeled A, right panel shows the respiratory system labeled B."
Answer: HTML_COMPOSITE

Request: "{request}"
Answer:'''


REPAIR_PROMPT = '''The following Python script was supposed to render a figure but raised \
an error. Fix it and output the corrected, complete, self-contained script the same way \
(ONE fenced python code block, ending with plt.savefig("output.png", dpi=150, \
bbox_inches="tight")). Only matplotlib and numpy are allowed.

Original request: {request}

Script that failed:
```python
{code}
```

Error raised:
{error}

Output the corrected script now.'''


# ----------------------------------------------------------------------
# Molecules: model writes a SMILES string (heavily represented in LLM
# training data), RDKit renders the actual 2D structure deterministically.
# ----------------------------------------------------------------------

MOLECULE_PROMPT = '''You are a chemistry assistant. Given a request naming a molecule, \
output its structure as a SMILES string. Output ONLY these two lines, nothing else:

SMILES: <smiles string>
TITLE: <molecule name>

Example.
Request: Draw the structure of caffeine.
SMILES: CN1C=NC2=C1C(=O)N(C)C(=O)N2C
TITLE: Caffeine

Example.
Request: Show the molecular structure of water.
SMILES: O
TITLE: Water

Now answer for this request:
{request}'''


MOLECULE_REPAIR_PROMPT = '''The SMILES string "{smiles}" could not be parsed as a valid \
molecule ({reason}). Output a corrected SMILES string for this request, in the same \
"SMILES: ...\\nTITLE: ..." format, nothing else.

Request: {request}'''


# ----------------------------------------------------------------------
# Circuits: model extracts a component list (schemdraw's direction-chaining
# API is unfamiliar enough to a small model that writing it directly is
# unreliable); a deterministic function lays them out as a single series
# loop and schemdraw renders it.
# ----------------------------------------------------------------------

CIRCUIT_DSL_PROMPT = '''You are a precise circuit-structure extractor. Given a description \
of an electrical circuit, list every component in order around the loop. Output ONLY lines \
in this exact format, nothing else (no prose):

COMPONENT: <type> <label>
PARALLEL: <type> <label> | <type> <label> | ...

<type> must be one of: battery, resistor, capacitor, inductor, led, switch, diode. Use a \
PARALLEL line (components separated by " | ") for components that branch off the same two \
points rather than continuing the series loop -- e.g. two resistors "in parallel".

Example 1.
Request: A 9V battery connected to a 330 ohm resistor and an LED in series.
COMPONENT: battery 9V
COMPONENT: resistor 330ohm
COMPONENT: led LED

Example 2.
Request: A battery connected to two resistors in parallel.
COMPONENT: battery 9V
PARALLEL: resistor R1 | resistor R2

Now extract from this request:
{request}'''


CIRCUIT_REPAIR_PROMPT = '''Your previous answer could not be parsed as a COMPONENT:/\
PARALLEL: list ({reason}). Output ONLY lines in the form "COMPONENT: <type> <label>" or \
"PARALLEL: <type> <label> | <type> <label>", where <type> is one of battery, resistor, \
capacitor, inductor, led, switch, diode -- nothing else.

Request: {request}'''


# ----------------------------------------------------------------------
# Graph theory: like flowchart diagrams, the model only extracts a
# node/edge list -- but graph-theory graphs aren't hierarchical processes,
# so they're laid out with networkx's spring layout instead of top-to-
# bottom stacking, and edges can be undirected or weighted.
# ----------------------------------------------------------------------

GRAPH_DSL_PROMPT = '''You are a precise graph-structure extractor. Given a description of a \
graph-theory graph, list every node and every edge. Output ONLY lines in this exact format, \
nothing else (no prose):

NODE: <label>
EDGE: <a> -> <b>             (directed edge)
EDGE: <a> -- <b>             (undirected edge)
EDGE: <a> -- <b> : <weight>  (undirected, with a weight/number on the edge)

Example.
Request: An undirected graph with nodes A, B, C, D. Edges: A-B weight 5, B-C weight 3, \
A-C weight 2, C-D weight 4.
NODE: A
NODE: B
NODE: C
NODE: D
EDGE: A -- B : 5
EDGE: B -- C : 3
EDGE: A -- C : 2
EDGE: C -- D : 4

Now extract from this request:
{request}'''


GRAPH_REPAIR_PROMPT = '''Your previous answer could not be parsed as a NODE:/EDGE: list \
({reason}). Output ONLY lines in the form "NODE: <label>" or "EDGE: <a> -> <b>" / \
"EDGE: <a> -- <b>" (optionally " : <weight>"), nothing else.

Request: {request}'''


# ----------------------------------------------------------------------
# Vectors / free body diagrams: model extracts a label + (x, y) component
# per vector (computing components from word directions is a narrow,
# familiar task); a deterministic renderer draws real arrows with shafts
# and auto-places labels with overlap avoidance -- never hand-rolled by
# the model, which otherwise tends to invent ad hoc arrow-drawing helpers
# that don't actually draw a shaft (see math_vector_addition.png history).
# ----------------------------------------------------------------------

VECTOR_DSL_PROMPT = '''You are a precise vector-diagram extractor. Given a description of \
vectors or forces (a vector-addition problem, or a free body diagram), output ONLY lines in \
this format, nothing else:

BOX: yes or no
VECTOR: <label text> | <x component> | <y component>

Output "BOX: yes" when the vectors act on a physical object (a free body diagram) so a \
small object box is drawn at the origin; output "BOX: no" for a pure math vector-addition \
diagram with no object.

For each vector, the label is exactly the text to print next to the arrow. The x and y \
components are plain numbers (decimals allowed). Compute components for directions given in \
words: "up" = (0, +mag), "down" = (0, -mag), "left" = (-mag, 0), "right" = (+mag, 0), and for \
an angle theta from the x-axis use (mag*cos(theta), mag*sin(theta)). When forces are named \
only (weight, normal, friction, applied) with no magnitude given, use a magnitude of about 3 \
for each and a direction that matches the physical setup.

Example 1.
Request: Draw two vectors from the origin: vector A = (3, 2) and vector B = (1, 4), and draw \
their vector sum as a third vector, all labeled.
BOX: no
VECTOR: A (3, 2) | 3 | 2
VECTOR: B (1, 4) | 1 | 4
VECTOR: A + B (4, 6) | 4 | 6

Example 2.
Request: Draw a free body diagram of a box on a flat surface with weight, normal force, an \
applied force, and friction, all labeled.
BOX: yes
VECTOR: Weight | 0 | -3
VECTOR: Normal Force | 0 | 3.2
VECTOR: Applied Force | 2.2 | 2.2
VECTOR: Friction | -2.5 | 0

Example 3.
Request: Draw a free body diagram of a block on a 30 degree incline, showing gravity \
pulling straight down, the normal force perpendicular to the incline surface, and friction \
acting up the slope.
BOX: yes
VECTOR: Gravity | 0 | -3
VECTOR: Normal Force | -1.5 | 2.6
VECTOR: Friction | 2.6 | 1.5

Now extract from this request:
{request}'''


VECTOR_REPAIR_PROMPT = '''Your previous answer for this request could not be parsed \
({reason}). Output ONLY a "BOX: yes" or "BOX: no" line followed by one or more \
"VECTOR: <label> | <x> | <y>" lines, nothing else.

Request: {request}'''


# ----------------------------------------------------------------------
# Venn diagrams: model extracts the two set names and the contents of each
# region; the renderer always draws exactly two circles with fixed, tested
# geometry, so the model can never invent extra decorative shapes or
# mismatched borders (see math_venn_diagram.png history).
# ----------------------------------------------------------------------

VENN_DSL_PROMPT = '''You are a precise Venn-diagram extractor. Given a description of a \
two-set Venn diagram, output ONLY lines in this format, nothing else:

SET_A: <name of the first set>
SET_B: <name of the second set>
ONLY_A: <items that belong only to set A, comma-separated, or leave blank>
ONLY_B: <items that belong only to set B, comma-separated, or leave blank>
OVERLAP: <items that belong to both sets, comma-separated, or leave blank>

Example 1.
Request: Draw a Venn diagram with two overlapping circles labeled 'Even Numbers' and \
'Multiples of 3', with the number 6 placed in the overlap region.
SET_A: Even Numbers
SET_B: Multiples of 3
ONLY_A:
ONLY_B:
OVERLAP: 6

Example 2.
Request: Draw a Venn diagram of Set A = {{2, 4, 6, 8}} and Set B = {{3, 6, 9}}, showing which \
numbers belong to each set and which are shared.
SET_A: Set A
SET_B: Set B
ONLY_A: 2, 4, 8
ONLY_B: 3, 9
OVERLAP: 6

Now extract from this request:
{request}'''


VENN_REPAIR_PROMPT = '''Your previous answer for this request could not be parsed \
({reason}). Output ONLY "SET_A:", "SET_B:", "ONLY_A:", "ONLY_B:", and "OVERLAP:" lines in \
that order, nothing else.

Request: {request}'''


# ----------------------------------------------------------------------
# Tables: model extracts column headers and row values; a deterministic
# ax.table() renderer with auto font-size/column-width handling always
# produces a real grid -- free-form code sometimes skipped ax.table()
# entirely and drew a scatter/line chart with floating text instead (see
# table_pass_fail.png / table_animals_classification.png history).
# ----------------------------------------------------------------------

TABLE_DSL_PROMPT = '''You are a precise table-structure extractor. Given a description of a \
table, extract an optional title, the column headers, and every row of data -- nothing more. \
You never draw anything or write any code; you only output the structure, exactly in this \
format and nothing else (no prose):

TITLE: <short title, or leave blank if the request doesn't suggest one>
COLUMNS: <header1> | <header2> | <header3> ...
ROW: <value1> | <value2> | <value3> ...

Each ROW must have the same number of values as there are column headers.

Example 1.
Request: Make a table of student scores: Alice 92 A, Bob 85 B, Carol 78 C. Columns: Name, \
Score, Grade.
TITLE:
COLUMNS: Name | Score | Grade
ROW: Alice | 92 | A
ROW: Bob | 85 | B
ROW: Carol | 78 | C

Example 2.
Request: Make a table of exam results: Alice Pass, Bob Fail, Carol Pass, Dave Pass. Columns: \
Name, Result.
TITLE: Exam Results
COLUMNS: Name | Result
ROW: Alice | Pass
ROW: Bob | Fail
ROW: Carol | Pass
ROW: Dave | Pass

Now extract from this request:
{request}'''


TABLE_REPAIR_PROMPT = '''Your previous answer for this request could not be parsed as a \
TITLE:/COLUMNS:/ROW: table ({reason}). Output ONLY a "TITLE: ..." line (can be empty), one \
"COLUMNS: <header> | <header> | ..." line, and one or more "ROW: <value> | <value> | ..." \
lines with the same number of values as columns, nothing else.

Request: {request}'''


# ----------------------------------------------------------------------
# Scenes: model extracts a shape/color/count list (and whether each group
# should cluster/overlap); a deterministic rejection-sampling placement
# never drops a requested shape and never overlaps two groups unless asked
# to (see scene_counting.png / scene_overlap_count.png history -- free-form
# code dropped 5 of 6 requested shapes and produced a duplicated/garbled
# title).
# ----------------------------------------------------------------------

SCENE_DSL_PROMPT = '''You are a precise scene-structure extractor. Given a description of a \
scene made of simple shapes, extract an optional title and every shape group -- nothing more. \
You never draw anything or write any code; you only output the structure, exactly in this \
format and nothing else (no prose):

TITLE: <short title, or leave blank if the request doesn't suggest one>
SHAPE: <circle|square|triangle> | <color> | <count> | <overlap: yes or no>

Use "overlap: yes" only when the request explicitly says shapes of that kind should overlap, \
touch, or cluster together; otherwise use "overlap: no" so they are scattered without \
touching.

Example 1.
Request: A scene contains 3 red circles, 2 blue squares, and 1 green triangle, scattered \
around randomly. Draw it.
TITLE:
SHAPE: circle | red | 3 | no
SHAPE: square | blue | 2 | no
SHAPE: triangle | green | 1 | no

Example 2.
Request: Draw 3 overlapping blue circles and 2 overlapping red circles, scattered so some \
circles of the same color overlap each other.
TITLE:
SHAPE: circle | blue | 3 | yes
SHAPE: circle | red | 2 | yes

Now extract from this request:
{request}'''


SCENE_REPAIR_PROMPT = '''Your previous answer for this request could not be parsed as a \
TITLE:/SHAPE: list ({reason}). Output ONLY a "TITLE: ..." line (can be empty) followed by \
one or more "SHAPE: <circle|square|triangle> | <color> | <count> | <yes or no>" lines, \
nothing else.

Request: {request}'''


# ----------------------------------------------------------------------
# Puzzles (IQ-question style): model extracts a small structured spec (a
# layout type plus a list of simple shape panels); a deterministic renderer
# draws fixed-size, evenly-spaced boxes so panels never overlap, and
# computes mirror/rotation geometry exactly instead of relying on the model
# to pick wide-enough axis limits (see puzzle_symmetry.png history, where
# the mirrored triangle was clipped by the figure boundary).
# ----------------------------------------------------------------------

PUZZLE_DSL_PROMPT = '''You are a precise IQ-puzzle structure extractor. Given a description \
of a visual pattern/IQ puzzle, extract its structure -- nothing more. You never draw \
anything or write any code; you only output the structure, exactly in this format and \
nothing else (no prose):

LAYOUT: sequence | symmetry | rotation | odd_one_out | analogy | row
PANEL: <circle|square|triangle|right_triangle> | <count> | <rotation_degrees> | <color>
PANEL: <circle|square|triangle|right_triangle> | <count> | <rotation_degrees> | <color>
...
ANSWER: <circle|square|triangle|right_triangle> | <count> | <rotation_degrees> | <color>
ODD_INDEX: <1-based index of the panel that's different>
TRANSFORM_DEGREES: <degrees to rotate, for layout=rotation>

The trailing "| <color>" on a PANEL/ANSWER line is optional -- omit it (and the preceding \
"|") entirely for plain black shapes, which is the norm for IQ-reasoning puzzles (symmetry, \
rotation, odd_one_out, analogy, and sequence puzzles are usually monochrome since color \
isn't part of the pattern). Only include a color when the request explicitly names one for \
that shape.

Use exactly one PANEL line per panel shown before the answer/question mark. IMPORTANT: omit \
the ANSWER line completely whenever the request says the last spot has, contains, or shows \
a "question mark" or is otherwise left unrevealed/unknown -- never invent a shape for it just \
because the earlier panels suggest a pattern. Only include ANSWER when the request itself \
states what the actual missing/next shape is. Only include ODD_INDEX for \
layout=odd_one_out, and TRANSFORM_DEGREES for layout=rotation. rotation_degrees on a PANEL \
line is normally 0 unless the shape shown in that specific panel is itself drawn rotated. \
Use layout=row for a plain ordered or alternating row of shapes with no missing item, \
comparison, or reasoning involved -- list every shown shape as its own PANEL line and do \
not add an ANSWER (row never gets a trailing "?" panel).

Example 1 (sequence, hidden answer).
Request: Draw a sequence puzzle: 4 panels showing 1, 2, 3, then 4 dots arranged in a square \
pattern -- what comes next?
LAYOUT: sequence
PANEL: circle | 1 | 0
PANEL: circle | 2 | 0
PANEL: circle | 3 | 0

Example 1b (sequence, hidden answer, explicit "question mark" wording).
Request: Draw a 2x2 grid of cells where three cells contain a circle and the fourth (bottom \
right) contains a question mark instead.
LAYOUT: sequence
PANEL: circle | 1 | 0
PANEL: circle | 1 | 0
PANEL: circle | 1 | 0

Example 2 (sequence, revealed answer).
Request: Draw a pattern puzzle with 4 panels: panel 1 has a circle, panel 2 a square, panel \
3 a triangle, and panel 4 shows the answer if the pattern repeats from panel 1.
LAYOUT: sequence
PANEL: circle | 1 | 0
PANEL: square | 1 | 0
PANEL: triangle | 1 | 0
ANSWER: circle | 1 | 0

Example 3 (symmetry).
Request: Draw a right triangle on the left of a vertical dashed line, and its mirror image \
reflected across that line on the right.
LAYOUT: symmetry
PANEL: right_triangle | 1 | 0

Example 4 (rotation).
Request: Draw a right triangle, and next to it the same triangle rotated 90 degrees.
LAYOUT: rotation
PANEL: right_triangle | 1 | 0
TRANSFORM_DEGREES: 90

Example 5 (odd one out).
Request: Draw 4 shapes in a row: three circles and one square, all the same size, with the \
odd one out circled in red.
LAYOUT: odd_one_out
PANEL: circle | 1 | 0
PANEL: circle | 1 | 0
PANEL: square | 1 | 0
PANEL: circle | 1 | 0
ODD_INDEX: 3

Example 6 (analogy).
Request: Draw an analogy puzzle: a small circle next to a large circle, then a small square \
next to a large square, then a small triangle next to a question mark representing what \
comes next.
LAYOUT: analogy
PANEL: circle | 1 | 0
PANEL: circle | 2 | 0
PANEL: triangle | 1 | 0

Example 7 (row, no puzzle reasoning, with colors).
Request: Draw a row of 6 shapes alternating between a red circle and a blue square.
LAYOUT: row
PANEL: circle | 1 | 0 | red
PANEL: square | 1 | 0 | blue
PANEL: circle | 1 | 0 | red
PANEL: square | 1 | 0 | blue
PANEL: circle | 1 | 0 | red
PANEL: square | 1 | 0 | blue

Now extract from this request:
{request}'''


PUZZLE_REPAIR_PROMPT = '''Your previous answer for this request could not be parsed \
({reason}). Output ONLY a "LAYOUT: ..." line, one or more "PANEL: <shape> | <count> | \
<rotation>" lines, and the optional ANSWER:/ODD_INDEX:/TRANSFORM_DEGREES: lines that apply \
to that layout, nothing else.

Request: {request}'''


# ----------------------------------------------------------------------
# SVG: model writes SVG markup directly; cairosvg renders it to PNG.
# Kept to simple primitives (rect/circle/polygon/line/text/path) that a
# small model generates reliably. No scripts, no event handlers.
# ----------------------------------------------------------------------

SVG_PROMPT = '''You are an SVG graphics creator. Given a description of a simple icon, \
logo, or graphic, output ONE complete, self-contained SVG in a fenced code block.

Rules:
- Output ONLY one fenced code block: ```svg ... ```
- The root <svg> element MUST have xmlns, width, height, and viewBox attributes.
- Use simple primitives: rect, circle, ellipse, polygon, polyline, line, path, text, g.
- NO <script> tags, NO onclick/onload/on* event attributes.
- Use named colors (red, steelblue, orange, ...) or hex codes (#cc3300, ...).
- Keep path data (d="...") short and simple -- prefer polygon/polyline over complex paths.
- Aim for 200-600px width/height. viewBox should match or scale to those dimensions.
- Add a <title> element inside <svg> for accessibility.

Example 1.
Request: Draw a simple house icon.
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <title>House</title>
  <polygon points="20,110 100,30 180,110" fill="#cc4444" stroke="#333" stroke-width="3"/>
  <rect x="45" y="110" width="110" height="80" fill="#e8c87a" stroke="#333" stroke-width="3"/>
  <rect x="82" y="150" width="36" height="40" fill="#8B5E3C"/>
  <rect x="55" y="125" width="28" height="28" fill="#aadaff" stroke="#333" stroke-width="2"/>
</svg>
```

Example 2.
Request: Draw a traffic light with three colored circles.
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="260" viewBox="0 0 100 260">
  <title>Traffic Light</title>
  <rect x="20" y="10" width="60" height="240" rx="10" fill="#222"/>
  <circle cx="50" cy="55" r="22" fill="red"/>
  <circle cx="50" cy="130" r="22" fill="orange"/>
  <circle cx="50" cy="205" r="22" fill="limegreen"/>
</svg>
```

Example 3.
Request: Draw a simple five-pointed star.
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <title>Star</title>
  <polygon points="100,15 120.6,68.8 177.6,68.8 131,100.1 148.5,154 100,122.4 51.5,154 69,100.1 22.4,68.8 79.4,68.8"
           fill="gold" stroke="darkorange" stroke-width="2"/>
</svg>
```

Example 4.
Request: Draw a right-pointing arrow in blue.
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="120" viewBox="0 0 200 120">
  <title>Right Arrow</title>
  <polygon points="10,40 130,40 130,15 190,60 130,105 130,80 10,80"
           fill="royalblue" stroke="navy" stroke-width="2"/>
</svg>
```

Example 5.
Request: Draw a green checkmark tick symbol inside a circle.
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <title>Checkmark</title>
  <circle cx="100" cy="100" r="90" fill="#4caf50" stroke="#388e3c" stroke-width="4"/>
  <polyline points="45,100 80,140 155,60"
            fill="none" stroke="white" stroke-width="14"
            stroke-linecap="round" stroke-linejoin="round"/>
</svg>
```

Example 6.
Request: Draw a smiley face.
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <title>Smiley Face</title>
  <circle cx="100" cy="100" r="90" fill="#FFD700" stroke="#FFA500" stroke-width="3"/>
  <circle cx="70" cy="80" r="10" fill="#333"/>
  <circle cx="130" cy="80" r="10" fill="#333"/>
  <path d="M60,125 Q100,165 140,125" fill="none" stroke="#333" stroke-width="5"
        stroke-linecap="round"/>
</svg>
```

Example 7.
Request: Draw a sun icon with a central circle and 8 rays radiating outward.
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <title>Sun</title>
  <circle cx="100" cy="100" r="35" fill="#FFD700"/>
  <line x1="100" y1="55"  x2="100" y2="20"  stroke="#FFD700" stroke-width="6" stroke-linecap="round"/>
  <line x1="145" y1="55"  x2="167" y2="33"  stroke="#FFD700" stroke-width="6" stroke-linecap="round"/>
  <line x1="145" y1="100" x2="180" y2="100" stroke="#FFD700" stroke-width="6" stroke-linecap="round"/>
  <line x1="145" y1="145" x2="167" y2="167" stroke="#FFD700" stroke-width="6" stroke-linecap="round"/>
  <line x1="100" y1="145" x2="100" y2="180" stroke="#FFD700" stroke-width="6" stroke-linecap="round"/>
  <line x1="55"  y1="145" x2="33"  y2="167" stroke="#FFD700" stroke-width="6" stroke-linecap="round"/>
  <line x1="55"  y1="100" x2="20"  y2="100" stroke="#FFD700" stroke-width="6" stroke-linecap="round"/>
  <line x1="55"  y1="55"  x2="33"  y2="33"  stroke="#FFD700" stroke-width="6" stroke-linecap="round"/>
</svg>
```

Example 8.
Request: Draw a play button icon: a circle with a right-pointing triangle inside.
```svg
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">
  <title>Play Button</title>
  <circle cx="100" cy="100" r="90" fill="none" stroke="#444" stroke-width="8"/>
  <polygon points="75,55 75,145 155,100" fill="#444"/>
</svg>
```

Now create for this request:
{request}'''


SVG_REPAIR_PROMPT = '''Your previous SVG for this request had a problem: {reason}

Fix it. Output ONLY a corrected fenced SVG code block (```svg ... ```) with:
- Root <svg> with xmlns, width, height, viewBox.
- Only simple primitives (rect, circle, ellipse, polygon, line, text, g).
- No <script> tags or on* event attributes.

Request: {request}'''


# ----------------------------------------------------------------------
# Mermaid: model writes Mermaid diagram syntax; mmdc CLI renders to PNG.
# Best for sequence diagrams, class diagrams, Gantt charts, ER diagrams.
# ----------------------------------------------------------------------

MERMAID_PROMPT = '''You are a Mermaid diagram creator. Given a request for a diagram, \
output ONE complete Mermaid diagram in a fenced code block.

Supported diagram types:
- sequenceDiagram  -- message flows between actors/systems (best for API flows, protocols)
- classDiagram     -- UML class hierarchies and relationships
- gantt            -- project timelines and task schedules
- erDiagram        -- entity-relationship database models
- gitGraph         -- git branching and merging

Rules:
- Output ONLY one fenced code block: ```mermaid ... ```
- Choose the diagram type that best matches the request.
- Keep syntax clean and correct. Use proper Mermaid syntax for the chosen type.
- For sequenceDiagram: declare participants, use ->>, -->>, ->, --> for messages.
- For classDiagram: use class keyword, <|-- for inheritance, *-- for composition.
- For gantt: include dateFormat, at least one section with tasks.
- For erDiagram: use ENTITY {{}} blocks and relationship lines with labels.

Example 1 (sequence diagram).
Request: Draw a sequence diagram showing a REST API login flow.
```mermaid
sequenceDiagram
    participant U as User
    participant C as Client App
    participant S as Auth Server
    participant DB as Database
    U->>C: Enter credentials
    C->>S: POST /login
    S->>DB: Query user record
    DB-->>S: User data + hash
    S-->>C: 200 OK + JWT token
    C-->>U: Logged in
```

Example 2 (class diagram).
Request: Draw a UML class diagram for a simple animal hierarchy.
```mermaid
classDiagram
    class Animal {{
        +String name
        +int age
        +speak() String
    }}
    class Dog {{
        +String breed
        +fetch() void
    }}
    class Cat {{
        +bool isIndoor
        +purr() void
    }}
    Animal <|-- Dog
    Animal <|-- Cat
```

Example 3 (Gantt chart).
Request: Draw a Gantt chart for a 6-week software project.
```mermaid
gantt
    title Software Project Timeline
    dateFormat  YYYY-MM-DD
    section Planning
    Requirements :a1, 2024-01-01, 7d
    Design       :a2, after a1, 7d
    section Development
    Backend      :b1, after a2, 14d
    Frontend     :b2, after a2, 14d
    section Release
    Testing      :c1, after b1, 7d
    Deployment   :c2, after c1, 3d
```

Example 4 (ER diagram).
Request: Draw an entity-relationship diagram for a bookstore database.
```mermaid
erDiagram
    BOOK {{
        string isbn PK
        string title
        string author
        float price
    }}
    CUSTOMER {{
        int id PK
        string name
        string email
    }}
    ORDER {{
        int id PK
        date orderDate
        float total
    }}
    CUSTOMER ||--o{{ ORDER : places
    ORDER }}|--|{{ BOOK : contains
```

Now create for this request:
{request}'''


MERMAID_REPAIR_PROMPT = '''Your previous Mermaid diagram for this request had a problem: {reason}

Fix it. Output ONLY a corrected fenced code block (```mermaid ... ```) with valid \
Mermaid syntax for the appropriate diagram type (sequenceDiagram / classDiagram / \
gantt / erDiagram / gitGraph).

Request: {request}'''


# ----------------------------------------------------------------------
# Vega-Lite: model writes a Vega-Lite v5 JSON spec; vl-convert renders PNG.
# Best for heatmaps, bubble charts, and faceted/layered charts that need
# Vega-Lite's declarative encoding grammar.
# ----------------------------------------------------------------------

VEGALITE_PROMPT = '''You are a Vega-Lite chart specification writer. Given a request \
for a data visualization, output ONE complete, valid Vega-Lite v5 JSON specification \
in a fenced code block.

Rules:
- Output ONLY one fenced code block: ```json ... ```
- Always include "$schema": "https://vega.github.io/schema/vega-lite/v5.json"
- Include "width" and "height" (e.g. 400 and 300).
- CRITICAL: data must be {{"values": [...]}} NOT a bare array -- "data": {{"values": [...]}}
- Every field named in encoding (e.g. "field": "score") MUST be a key in every data object.
- For heatmaps: data MUST be in LONG/TIDY format -- ONE ROW PER CELL. Never wide format.
  Example: [{{"student":"A","subject":"Math","score":85}}, {{"student":"A","subject":"Physics","score":90}}]
  NOT: [{{"student":"A","math":85,"physics":90}}] (wide format will produce an empty chart!)
- Every encoding channel needs "field" and "type" (quantitative/ordinal/nominal/temporal).
- For multi-layer charts: use "layer": [...] at the TOP LEVEL, each layer has its own "mark" and "encoding".
  Use {{"type":"bar","color":"steelblue"}} to set mark color, NOT config.
- Keep data compact -- invent realistic but small (8-20 row) example datasets.
- Do NOT use axis "format" strings. Remove any "format" key from axis objects.

Chart type guidance:
  Heatmap:       mark "rect"  + x ordinal + y ordinal + color quantitative  [LONG format data!]
  Bubble chart:  mark "point" + x quant   + y quant   + size quant + color nominal
  Layered:       "layer": [{{"mark": {{"type":"bar","color":"blue"}}, "encoding":...}}, {{"mark": {{"type":"line","color":"red"}}, "encoding":...}}]
  Histogram:     mark "bar"   + x bin:true quant + y aggregate count
  Scatter:       mark "point" + x quant   + y quant   + color/shape nominal (optional)

Example 1 (heatmap).
Request: Draw a heatmap of average temperature by month and city.
```json
{{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "width": 400,
  "height": 250,
  "title": "Average Temperature by Month and City",
  "data": {{
    "values": [
      {{"city": "NYC", "month": "Jan", "temp": 1}},
      {{"city": "NYC", "month": "Apr", "temp": 12}},
      {{"city": "NYC", "month": "Jul", "temp": 27}},
      {{"city": "NYC", "month": "Oct", "temp": 14}},
      {{"city": "LA",  "month": "Jan", "temp": 14}},
      {{"city": "LA",  "month": "Apr", "temp": 18}},
      {{"city": "LA",  "month": "Jul", "temp": 24}},
      {{"city": "LA",  "month": "Oct", "temp": 20}},
      {{"city": "CHI", "month": "Jan", "temp": -4}},
      {{"city": "CHI", "month": "Apr", "temp": 10}},
      {{"city": "CHI", "month": "Jul", "temp": 25}},
      {{"city": "CHI", "month": "Oct", "temp": 11}}
    ]
  }},
  "mark": "rect",
  "encoding": {{
    "x": {{"field": "month", "type": "ordinal", "sort": ["Jan","Apr","Jul","Oct"], "title": "Month"}},
    "y": {{"field": "city",  "type": "ordinal", "title": "City"}},
    "color": {{"field": "temp", "type": "quantitative", "title": "Temp (°C)",
              "scale": {{"scheme": "redyellowblue", "reverse": true}}}}
  }}
}}
```

Example 2 (bubble chart).
Request: Draw a bubble chart comparing countries by GDP, Life Expectancy, and Population.
```json
{{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "width": 450,
  "height": 350,
  "title": "Countries: GDP vs Life Expectancy",
  "data": {{
    "values": [
      {{"country": "USA",    "gdp": 63000, "life_exp": 79, "pop": 330}},
      {{"country": "China",  "gdp": 12000, "life_exp": 77, "pop": 1400}},
      {{"country": "Japan",  "gdp": 40000, "life_exp": 84, "pop": 125}},
      {{"country": "India",  "gdp": 2100,  "life_exp": 70, "pop": 1380}},
      {{"country": "Germany","gdp": 46000, "life_exp": 81, "pop": 83}},
      {{"country": "Brazil", "gdp": 8800,  "life_exp": 75, "pop": 213}}
    ]
  }},
  "mark": {{"type": "point", "filled": true, "opacity": 0.8}},
  "encoding": {{
    "x": {{"field": "gdp",      "type": "quantitative", "title": "GDP per Capita (USD)"}},
    "y": {{"field": "life_exp", "type": "quantitative", "title": "Life Expectancy (years)", "scale": {{"zero": false}}}},
    "size": {{"field": "pop",   "type": "quantitative", "title": "Population (M)", "legend": {{"title": "Pop (M)"}}}},
    "color": {{"field": "country", "type": "nominal", "legend": {{"title": "Country"}}}},
    "tooltip": [
      {{"field": "country",  "type": "nominal"}},
      {{"field": "gdp",      "type": "quantitative"}},
      {{"field": "life_exp", "type": "quantitative"}},
      {{"field": "pop",      "type": "quantitative"}}
    ]
  }}
}}
```

Example 3 (layered bar + line).
Request: Show monthly revenue as blue bars with a red cumulative revenue line.
```json
{{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "width": 450,
  "height": 280,
  "title": "Monthly Revenue with Cumulative Total",
  "data": {{
    "values": [
      {{"month": "Jan", "revenue": 80,  "cumulative": 80}},
      {{"month": "Feb", "revenue": 95,  "cumulative": 175}},
      {{"month": "Mar", "revenue": 70,  "cumulative": 245}},
      {{"month": "Apr", "revenue": 110, "cumulative": 355}},
      {{"month": "May", "revenue": 90,  "cumulative": 445}},
      {{"month": "Jun", "revenue": 120, "cumulative": 565}}
    ]
  }},
  "layer": [
    {{
      "mark": {{"type": "bar", "color": "steelblue", "opacity": 0.8}},
      "encoding": {{
        "x": {{"field": "month", "type": "ordinal", "title": "Month"}},
        "y": {{"field": "revenue", "type": "quantitative", "title": "Revenue ($K)"}}
      }}
    }},
    {{
      "mark": {{"type": "line", "color": "red", "point": true}},
      "encoding": {{
        "x": {{"field": "month", "type": "ordinal"}},
        "y": {{"field": "cumulative", "type": "quantitative"}}
      }}
    }}
  ]
}}
```

Example 4 (heatmap with LONG data format).
Request: Heatmap of exam scores for students across subjects.
```json
{{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "width": 300,
  "height": 200,
  "title": "Exam Scores",
  "data": {{
    "values": [
      {{"student": "Alice", "subject": "Math",    "score": 85}},
      {{"student": "Alice", "subject": "Physics", "score": 92}},
      {{"student": "Bob",   "subject": "Math",    "score": 78}},
      {{"student": "Bob",   "subject": "Physics", "score": 80}}
    ]
  }},
  "mark": "rect",
  "encoding": {{
    "x": {{"field": "subject", "type": "ordinal", "title": "Subject"}},
    "y": {{"field": "student", "type": "ordinal", "title": "Student"}},
    "color": {{"field": "score", "type": "quantitative", "scale": {{"scheme": "greens"}}}}
  }}
}}
```

Now create for this request:
{request}'''


VEGALITE_REPAIR_PROMPT = '''Your previous Vega-Lite specification for this request had \
a problem: {reason}

Fix it. Output ONLY a corrected fenced JSON code block (```json ... ```) containing \
a valid Vega-Lite v5 specification with "$schema", "width", "height", "data", "mark", \
and "encoding" at minimum.

Request: {request}'''


# ----------------------------------------------------------------------
# HTML: model writes a self-contained HTML page; chromium headless renders it.
# Use for document layouts, cards, invoices, styled tables, and form mockups.
# No JavaScript allowed.
# ----------------------------------------------------------------------

HTML_PROMPT = '''You are an HTML document creator. Given a description, output ONE complete, \
self-contained HTML page in a fenced code block.

Rules:
- Output ONLY one fenced code block: ```html ... ```
- Start with <!DOCTYPE html> and include <html>, <head>, <body>.
- Use inline CSS styles only (no external stylesheets). Include font-family: sans-serif on body.
- Use a fixed page width: set body max-width to 800px with margin auto and padding 20px.
- NO <script> tags, NO onclick/onload/on* event attributes. NO <img> tags.
- Keep it simple and readable. Use a white background.
- For photo/image placeholders: use a styled <div> with a person emoji (&#128100;) or colored initials, NOT an <img> element.
- For tables: use border-collapse:collapse, alternating row colors, colored header row.
- For cards/profiles: use a centered card with box-shadow and rounded corners.
- For invoices/receipts: items in a table with line totals; bold the final total row.

Example 1 (invoice).
Request: A simple consulting invoice.
```html
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Invoice</title></head>
<body style="font-family:sans-serif;max-width:800px;margin:auto;padding:30px;background:#fff">
  <h1 style="color:#2c3e50">INVOICE</h1>
  <p><strong>From:</strong> Acme Consulting &nbsp;|&nbsp; <strong>To:</strong> XYZ Corp &nbsp;|&nbsp; <strong>Date:</strong> 2024-06-01</p>
  <table style="width:100%;border-collapse:collapse;margin-top:20px">
    <tr style="background:#2c3e50;color:#fff">
      <th style="padding:10px;text-align:left">Service</th>
      <th style="padding:10px;text-align:right">Hours</th>
      <th style="padding:10px;text-align:right">Rate</th>
      <th style="padding:10px;text-align:right">Total</th>
    </tr>
    <tr style="background:#ecf0f1">
      <td style="padding:10px">Web Development</td>
      <td style="padding:10px;text-align:right">40</td>
      <td style="padding:10px;text-align:right">$150</td>
      <td style="padding:10px;text-align:right">$6,000</td>
    </tr>
    <tr>
      <td style="padding:10px">UI Design</td>
      <td style="padding:10px;text-align:right">20</td>
      <td style="padding:10px;text-align:right">$120</td>
      <td style="padding:10px;text-align:right">$2,400</td>
    </tr>
    <tr style="font-weight:bold;background:#2c3e50;color:#fff">
      <td colspan="3" style="padding:10px">Total</td>
      <td style="padding:10px;text-align:right">$8,400</td>
    </tr>
  </table>
</body>
</html>
```

Example 2 (profile card).
Request: An employee profile card for a data scientist.
```html
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Profile</title></head>
<body style="font-family:sans-serif;background:#f0f2f5;display:flex;justify-content:center;padding:40px">
  <div style="background:#fff;border-radius:12px;box-shadow:0 4px 16px rgba(0,0,0,0.12);padding:30px;width:320px;text-align:center">
    <div style="width:80px;height:80px;border-radius:50%;background:#ccc;margin:0 auto 16px;display:flex;align-items:center;justify-content:center;font-size:32px">&#128100;</div>
    <h2 style="margin:0;color:#2c3e50">Dr. Sarah Chen</h2>
    <p style="color:#7f8c8d;margin:4px 0">Senior Data Scientist</p>
    <p style="color:#95a5a6;font-size:13px">Machine Learning Team</p>
    <hr style="margin:20px 0;border:none;border-top:1px solid #eee">
    <div style="display:flex;justify-content:space-around">
      <div><div style="font-size:22px;font-weight:bold;color:#3498db">94%</div><div style="font-size:12px;color:#7f8c8d">Accuracy</div></div>
      <div><div style="font-size:22px;font-weight:bold;color:#3498db">47</div><div style="font-size:12px;color:#7f8c8d">Models</div></div>
      <div><div style="font-size:22px;font-weight:bold;color:#3498db">8y</div><div style="font-size:12px;color:#7f8c8d">Exp.</div></div>
    </div>
  </div>
</body>
</html>
```

Example 3 (comparison table).
Request: A comparison table of three cloud providers.
```html
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Cloud Comparison</title></head>
<body style="font-family:sans-serif;max-width:800px;margin:auto;padding:30px;background:#fff">
  <h2 style="color:#333">Cloud Provider Comparison</h2>
  <table style="width:100%;border-collapse:collapse">
    <tr style="background:#3498db;color:#fff">
      <th style="padding:12px;text-align:left">Feature</th>
      <th style="padding:12px">AWS</th>
      <th style="padding:12px">Azure</th>
      <th style="padding:12px">GCP</th>
    </tr>
    <tr style="background:#f8f9fa">
      <td style="padding:12px;font-weight:bold">Free Tier</td>
      <td style="padding:12px;text-align:center">12 months</td>
      <td style="padding:12px;text-align:center">12 months</td>
      <td style="padding:12px;text-align:center">Always free</td>
    </tr>
    <tr>
      <td style="padding:12px;font-weight:bold">Compute</td>
      <td style="padding:12px;text-align:center">EC2</td>
      <td style="padding:12px;text-align:center">VMs</td>
      <td style="padding:12px;text-align:center">Compute Engine</td>
    </tr>
    <tr style="background:#f8f9fa">
      <td style="padding:12px;font-weight:bold">Market Share</td>
      <td style="padding:12px;text-align:center">33%</td>
      <td style="padding:12px;text-align:center">22%</td>
      <td style="padding:12px;text-align:center">10%</td>
    </tr>
  </table>
</body>
</html>
```

Now create for this request:
{request}'''


HTML_REPAIR_PROMPT = '''Your previous HTML page for this request had a problem: {reason}

Fix it. Output ONLY a corrected fenced HTML code block (```html ... ```) that is complete, \
self-contained, uses only inline styles, and has NO <script> tags or event handlers.

Request: {request}'''


# ----------------------------------------------------------------------
# HTML_COMPOSITE: multi-panel layout where each panel is a distinct sub-image
# (photo, diagram, chart, etc.) generated independently, then composed into
# a labelled grid rendered as a final HTML screenshot.
# ----------------------------------------------------------------------

HTML_COMPOSITE_PROMPT = '''You are a precise multi-panel layout extractor. The request \
describes an image containing multiple distinct sub-images arranged in a grid or row, \
each with a letter or number label (A, B, C... or 1, 2, 3...). Extract the layout.

Output ONLY lines in this exact format, nothing else (no prose):

TITLE: <overall title, or leave blank>
PANEL: <self-contained visual description for this sub-image> | <label>

One PANEL line per distinct sub-image. The description must be fully self-contained \
(independently visualizable without seeing the other panels or the original image). \
The label is the letter or number shown below or beside that panel.

Rules:
- Each description should be specific and concrete enough to generate a good image.
- For photos/scenes: describe what the sub-image shows (objects, setting, activity).
- For diagrams/charts: describe the diagram type and its content.
- For scientific illustrations: describe the structure being shown.
- Do not reference other panels ("similar to panel A") -- each description stands alone.
- The label is exactly what appears below the image (e.g. "A", "B", "(a)", "1", "2").

Example 1.
Request: Three images of foot types labeled A, B, C: foot A has a high arch, foot B is \
flat with no arch, foot C has a normal moderate arch.
TITLE: Foot Type Comparison
PANEL: A human foot viewed from the side showing a pronounced high arch with significant \
space between the sole and the ground | A
PANEL: A human foot viewed from the side showing a completely flat sole with no visible \
arch, the entire foot touching the ground | B
PANEL: A human foot viewed from the side showing a moderate normal arch, typical healthy \
foot posture | C

Example 2.
Request: Four panels in a 2x2 grid showing different types of leaves: panel A shows a \
simple oval leaf, B shows a compound leaf with multiple leaflets, C shows a lobed maple \
leaf, D shows a needle-like pine leaf.
TITLE: Leaf Types
PANEL: A simple smooth oval leaf with a pointed tip and visible midrib, on a white background | A
PANEL: A compound leaf with 5-7 separate oval leaflets arranged along a central stem | B
PANEL: A classic maple leaf with 5 pointed lobes and serrated edges, bright green | C
PANEL: A bundle of thin needle-like pine leaves (needles) clustered together, dark green | D

Example 3.
Request: Images 1-3 show three different animals: a golden retriever, a tabby cat, and a parrot.
TITLE:
PANEL: A golden retriever dog with fluffy fur, sitting and facing the camera, friendly expression | 1
PANEL: A tabby cat with striped orange and gray fur, lying down with eyes open | 2
PANEL: A colorful parrot with red, blue and green feathers perched on a branch | 3

Example 4.
Request: A comparison of three chart types: panel A shows a bar chart of sales data, \
panel B shows a pie chart of market share, panel C shows a line graph of growth over time.
TITLE: Chart Type Examples
PANEL: A vertical bar chart showing monthly sales for January through June with blue bars | A
PANEL: A pie chart showing market share: Company A 40%, Company B 35%, Company C 25% | B
PANEL: A line graph showing steady growth from 100 to 350 units over 12 months | C

Now extract from this request:
{request}'''


HTML_COMPOSITE_REPAIR_PROMPT = '''Your previous answer for this request could not be \
parsed as a multi-panel layout ({reason}). Output ONLY a "TITLE: ..." line (can be blank) \
followed by one or more "PANEL: <description> | <label>" lines, nothing else.

Request: {request}'''
