"""
Canonical tasks for SYMMETRY detection.

These tasks test whether the framework correctly identifies SYMMETRY
representation as the best fit when the task involves reflectional,
rotational, or translational symmetry.

Ground truth: Symmetry > Object for symmetry-detection tasks because
object representation sees individual entities but misses the global
regularity that relates them.
"""

from ..specification import CanonicalTask, grid


# ─────────────────────────────────────────────────────────────────────
# Task: Horizontal reflection symmetry
# ─────────────────────────────────────────────────────────────────────

HORIZONTAL_REFLECTION = CanonicalTask(
    task_id="horizontal_reflection",
    name="Horizontal Reflection Symmetry",
    category="symmetry",
    failure_mode="reflection_ambiguity",
    description=(
        "A grid where the top half is a perfect mirror of the bottom half. "
        "Symmetry representation should detect the reflection axis and encode "
        "only half the grid. Object representation would enumerate individual "
        "pixel pairs but miss the global reflection pattern."
    ),
    justification=(
        "Symmetry representation compresses the grid by factor ~2 (encode one "
        "half + reflection rule). Object representation treats symmetric pairs "
        "as unrelated objects. The hypothesis space for 'generate symmetric "
        "output' is O(1) in symmetry rep vs O(10^(HW/2)) in object rep."
    ),
    train_inputs=[
        grid([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0],
            [0, 0, 0, 0, 0],  # axis of symmetry
            [6, 7, 8, 9, 0],
            [1, 2, 3, 4, 5],
        ]),
    ],
    best_representation="symmetry",
    worst_representation="pixel",
    tags={"symmetry", "reflection", "compression"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: Rotational symmetry (order 4)
# ─────────────────────────────────────────────────────────────────────

ROTATIONAL_SYMMETRY_4 = CanonicalTask(
    task_id="rotational_symmetry_4",
    name="Rotational Symmetry (Order 4)",
    category="symmetry",
    failure_mode="rotation_ambiguity",
    description=(
        "A grid with 4-fold rotational symmetry. Rotating 90° produces the "
        "same pattern. Symmetry representation compresses by factor 4. "
        "Object representation would see 4 distinct quadrants as unrelated."
    ),
    justification=(
        "4-fold rotational symmetry means the grid is fully determined by one "
        "quadrant. Symmetry representation encodes 1/4 of the pixels + rotation "
        "rule. The hypothesis 'this pattern has 4-fold symmetry' is trivial to "
        "verify in symmetry rep and requires 10^(HW) search in pixel rep."
    ),
    train_inputs=[
        grid([
            [1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 1, 1],
        ]),
    ],
    best_representation="symmetry",
    worst_representation="pixel",
    tags={"symmetry", "rotation", "order_4"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: Translational symmetry (wallpaper pattern)
# ─────────────────────────────────────────────────────────────────────

TRANSLATIONAL_SYMMETRY = CanonicalTask(
    task_id="translational_symmetry",
    name="Translational Symmetry (Wallpaper)",
    category="symmetry",
    failure_mode="translation_ambiguity",
    description=(
        "A 2×2 pattern repeated across the entire 8×8 grid (tiling). "
        "Symmetry representation compresses this to the fundamental tile "
        "plus translation vectors. Object representation would enumerate "
        "each tile instance as a separate object."
    ),
    justification=(
        "Translational symmetry means the entire grid is a repetition of a "
        "small tile. Detecting this in pixel space requires discovering the "
        "repetition period, which is exponential in the period size. Symmetry "
        "representation makes the repetition explicit and trivial."
    ),
    train_inputs=[
        grid([
            [1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4],
        ]),
    ],
    best_representation="symmetry",
    worst_representation="pixel",
    tags={"symmetry", "translation", "tiling"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: No symmetry (negative test)
# ─────────────────────────────────────────────────────────────────────

NO_SYMMETRY = CanonicalTask(
    task_id="no_symmetry",
    name="Random Pattern — No Symmetry",
    category="symmetry",
    failure_mode="symmetry_overapplication",
    description=(
        "A grid with no discernible symmetry. Symmetry representation should "
        "have LOW applicability here. Object representation should be preferred. "
        "This tests that the framework does NOT blindly pick symmetry for "
        "everything."
    ),
    justification=(
        "A symmetry representation applied to random data should detect no "
        "non-trivial symmetry group, resulting in low applicability α. The "
        "framework should correctly rank object or pixel above symmetry for "
        "this task."
    ),
    train_inputs=[
        grid([
            [1, 7, 3, 9, 2],
            [4, 6, 1, 8, 5],
            [9, 2, 7, 3, 1],
            [5, 8, 4, 6, 9],
            [3, 1, 9, 2, 7],
        ]),
    ],
    best_representation="object",  # symmetry should NOT win here
    worst_representation="symmetry",
    tags={"symmetry", "negative", "applicability"},
)
