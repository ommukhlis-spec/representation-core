"""
Canonical tasks for TOPOLOGY detection.

These tasks test whether the framework correctly identifies TOPOLOGY
representation as the best fit when the task involves connectivity,
holes, containment, or Euler characteristics.

Object representation fails here because topology is about global
structure, not discrete entities. A hole is the ABSENCE of objects —
something object representation cannot encode directly.
"""

from ..specification import CanonicalTask, grid


# ─────────────────────────────────────────────────────────────────────
# Task: Single enclosed hole
# ─────────────────────────────────────────────────────────────────────

SINGLE_HOLE = CanonicalTask(
    task_id="single_hole",
    name="Detect a Single Enclosed Hole",
    category="topology",
    failure_mode="hole_detection_failure",
    description=(
        "A ring of color 1 enclosing an empty region (hole). The task is to "
        "detect the hole. Topology representation sees the hole as a topological "
        "feature (genus). Object representation sees only the ring itself but "
        "cannot represent 'empty space surrounded by object' as a first-class entity."
    ),
    justification=(
        "A hole is defined by what is NOT there — it's a topological feature. "
        "Object representation can encode the ring pixels but has no primitive "
        "for 'empty region enclosed by object.' Topology representation makes "
        "holes explicit via Euler characteristic or connectivity analysis. "
        "Detecting holes in pixel space requires discovering the concept of "
        "'enclosure' from first principles."
    ),
    train_inputs=[
        grid([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]),
    ],
    best_representation="topology",
    worst_representation="pixel",
    tags={"topology", "hole", "connectivity"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: Multiple holes (counting topological features)
# ─────────────────────────────────────────────────────────────────────

MULTIPLE_HOLES = CanonicalTask(
    task_id="multiple_holes",
    name="Count Multiple Holes",
    category="topology",
    failure_mode="hole_counting_failure",
    description=(
        "A shape with 3 distinct holes. Topology representation counts holes "
        "via Euler characteristic. Object representation would need to first "
        "discover the concept of 'hole' and then enumerate them."
    ),
    justification=(
        "Counting holes is a topological invariant computation (Betti numbers). "
        "In topology representation, this is a well-defined, efficient operation. "
        "In object or pixel representation, 'hole' is not a primitive — it must "
        "be learned from scratch."
    ),
    train_inputs=[
        grid([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]),
    ],
    best_representation="topology",
    worst_representation="pixel",
    tags={"topology", "hole_counting", "betti"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: Connectivity — are two regions connected?
# ─────────────────────────────────────────────────────────────────────

CONNECTIVITY_CHECK = CanonicalTask(
    task_id="connectivity_check",
    name="Check If Two Regions Are Connected",
    category="topology",
    failure_mode="connectivity_ambiguity",
    description=(
        "Two colored blobs connected by a thin bridge. The task is to determine "
        "whether they form one connected component. Topology representation "
        "answers this via connectivity analysis. Object representation may "
        "incorrectly segment them as two objects if the bridge is narrow."
    ),
    justification=(
        "Connectivity is a topological property. Determining whether two regions "
        "are connected requires analyzing the adjacency graph of pixels. Topology "
        "representation has connectivity as a primitive. Object representation "
        "using connected-components is actually doing topology implicitly — but "
        "a pure object representation would need to decide whether the bridge "
        "is 'wide enough' to count as a connection, which is ambiguous."
    ),
    train_inputs=[
        grid([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 2, 2, 0],
            [0, 1, 1, 0, 2, 2, 0],
            [0, 0, 0, 1, 0, 0, 0],  # thin bridge connecting both regions
            [0, 0, 0, 0, 0, 0, 0],
        ]),
    ],
    best_representation="topology",
    worst_representation="pixel",
    tags={"topology", "connectivity", "bridge"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: Nested containment
# ─────────────────────────────────────────────────────────────────────

NESTED_CONTAINMENT = CanonicalTask(
    task_id="nested_containment",
    name="Nested Containment (Object in Hole in Object)",
    category="topology",
    failure_mode="containment_hierarchy_failure",
    description=(
        "An object inside a hole, inside another object. This tests whether "
        "the representation can handle hierarchical topological structure. "
        "Topology representation naturally handles nesting levels."
    ),
    justification=(
        "Nested containment creates a hierarchy of topological features: "
        "outer object → hole → inner object. This is naturally represented "
        "as a tree in topology representation. Object representation would "
        "see three disconnected entities and miss the containment relationship."
    ),
    train_inputs=[
        grid([
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 0, 0, 0, 0, 0, 0, 2],
            [2, 0, 1, 1, 1, 1, 0, 2],
            [2, 0, 1, 0, 0, 1, 0, 2],
            [2, 0, 1, 0, 3, 1, 0, 2],  # object 3 inside hole inside object 1
            [2, 0, 1, 0, 0, 1, 0, 2],
            [2, 0, 1, 1, 1, 1, 0, 2],
            [2, 0, 0, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
        ]),
    ],
    best_representation="topology",
    worst_representation="pixel",
    tags={"topology", "containment", "nesting", "hierarchy"},
)
