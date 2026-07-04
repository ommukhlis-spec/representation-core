"""
Canonical tasks for COUNTING / DISCRETE OBJECT detection.

These tasks test whether the framework correctly identifies OBJECT
representation as the best fit when the task involves counting,
identifying, or manipulating discrete entities.

Ground truth: Object > Pixel for all counting tasks.
"""

from ..specification import CanonicalTask, grid


# ─────────────────────────────────────────────────────────────────────
# Task: Count discrete objects
# ─────────────────────────────────────────────────────────────────────

COUNT_DISCRETE_OBJECTS = CanonicalTask(
    task_id="count_discrete_objects",
    name="Count Discrete Objects",
    category="counting",
    failure_mode="object_merge_ambiguity",
    description=(
        "A grid containing 4 clearly separated objects (colored blobs). "
        "The task is to count them. Object representation should identify "
        "exactly 4 connected components. Pixel representation sees 100 "
        "independent pixels and cannot count objects without search."
    ),
    justification=(
        "Object representation encodes the grid as N discrete entities. "
        "Counting is O(1) after encoding. Pixel representation requires "
        "learning the concept of 'object' from scratch, with a hypothesis "
        "space of 10^100 possible groupings."
    ),
    train_inputs=[
        grid([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 3, 0, 0, 0, 4, 4, 0],
            [0, 0, 0, 0, 0, 0, 0, 4, 4, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
    ],
    best_representation="object",
    worst_representation="pixel",
    tags={"counting", "discrete", "segmentation"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: Count by color
# ─────────────────────────────────────────────────────────────────────

COUNT_BY_COLOR = CanonicalTask(
    task_id="count_by_color",
    name="Count Objects by Color",
    category="counting",
    failure_mode="color_grouping_ambiguity",
    description=(
        "A grid with objects of 3 different colors scattered across the grid. "
        "Object representation naturally groups by color. Pixel representation "
        "treats each pixel independently."
    ),
    justification=(
        "Object representation extracts color as a primitive property of each "
        "object. Counting per color is a trivial aggregation. Pixel representation "
        "must learn both object boundaries AND color grouping."
    ),
    train_inputs=[
        grid([
            [1, 0, 0, 0, 2, 0, 0, 0, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
    ],
    best_representation="object",
    worst_representation="pixel",
    tags={"counting", "color", "grouping"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: Uniform object size
# ─────────────────────────────────────────────────────────────────────

UNIFORM_OBJECT_SIZE = CanonicalTask(
    task_id="uniform_object_size",
    name="Detect Uniform Object Size",
    category="counting",
    failure_mode="size_invariance_blindness",
    description=(
        "All 5 objects have exactly the same size (4 pixels each, 2×2 blocks). "
        "Object representation makes this invariant trivially detectable. "
        "Pixel representation must test 10^100 possible groupings to discover "
        "this regularity."
    ),
    justification=(
        "The invariant 'all objects have equal size' is O(1) after object "
        "encoding (check obj.size for each of N objects). In pixel space, "
        "detecting this requires first discovering object boundaries, then "
        "comparing sizes — the first step alone is exponential."
    ),
    train_inputs=[
        grid([
            [1, 1, 0, 0, 0, 2, 2, 0, 0, 0],
            [1, 1, 0, 0, 0, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 3, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 3, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 4, 0, 0, 0, 5, 5, 0, 0, 0],
            [4, 4, 0, 0, 0, 5, 5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
    ],
    best_representation="object",
    worst_representation="pixel",
    tags={"counting", "invariant", "size"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: Many small objects (fragmentation stress test)
# ─────────────────────────────────────────────────────────────────────

MANY_SMALL_OBJECTS = CanonicalTask(
    task_id="many_small_objects",
    name="Many Small Objects — Fragmentation",
    category="counting",
    failure_mode="fragmentation_ambiguity",
    description=(
        "A grid with 15 single-pixel objects scattered across the grid. "
        "Object representation should detect them as individual entities "
        "but with high fragmentation (low applicability). This tests whether "
        "the framework correctly identifies that object rep is STILL the best "
        "choice despite low quality."
    ),
    justification=(
        "Even with fragmented objects, the object representation is the best "
        "available lens for counting discrete entities. Applicability α should "
        "be low (fragmented) but tension should still be lowest among available "
        "representations. Pixel representation has no concept of 'object' at all."
    ),
    train_inputs=[
        grid([
            [1, 0, 0, 2, 0, 0, 3, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 6, 0, 0, 7, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 8, 0, 0, 9, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # color 0 = 10? No, 0 is background
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 3, 0, 0, 4],  # reusing colors 2,3,4
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
    ],
    best_representation="object",
    worst_representation="pixel",
    tags={"counting", "fragmentation", "stress_test"},
)
