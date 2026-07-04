"""
Canonical tasks for REGION-based reasoning.

Region representation is the right lens when:
  - Boundaries between areas matter more than discrete objects
  - The grid is continuous (gradients, overlapping zones)
  - Spatial relationships (adjacency, enclosure) are primary
  - Object boundaries are ambiguous or overlapping

Object representation FAILS on these tasks because:
  - Connected-component segmentation breaks on overlapping regions
  - Object primitives cannot represent continuous spatial zones
"""

from ..specification import CanonicalTask, grid


# ─────────────────────────────────────────────────────────────────────
# Task: Overlapping regions (Venn diagram)
# ─────────────────────────────────────────────────────────────────────

OVERLAPPING_REGIONS = CanonicalTask(
    task_id="overlapping_regions",
    name="Overlapping Regions (Venn-like)",
    category="region",
    failure_mode="overlap_ambiguity",
    description=(
        "Two colored regions that partially overlap, creating a third zone "
        "where both colors mix. Object representation fails because the "
        "overlap zone belongs to BOTH regions — violating the discrete "
        "ownership assumption of object segmentation."
    ),
    justification=(
        "When two regions overlap, a pixel may 'belong' to both. Object "
        "representation requires each pixel to belong to exactly one object. "
        "Region representation can model overlapping zones as separate "
        "spatial entities with graded membership, or as three non-overlapping "
        "zones (A-only, B-only, A∩B)."
    ),
    train_inputs=[
        grid([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 3, 2, 2, 0],  # zone 3 = overlap of 1 and 2
            [0, 0, 0, 2, 2, 2, 0],
            [0, 0, 0, 2, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]),
    ],
    best_representation="region",
    worst_representation="object",  # object rep fails on overlap
    tags={"region", "overlap", "venn"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: Continuous gradient (no discrete objects)
# ─────────────────────────────────────────────────────────────────────

CONTINUOUS_GRADIENT = CanonicalTask(
    task_id="continuous_gradient",
    name="Continuous Color Gradient",
    category="region",
    failure_mode="gradient_discretization_failure",
    description=(
        "A grid where color smoothly transitions from 1 to 9 across columns. "
        "There are NO discrete objects. Object representation would fragment "
        "this into dozens of tiny single-pixel 'objects'. Region representation "
        "can model it as a continuous spatial gradient."
    ),
    justification=(
        "When the grid is continuous (gradient, blur, noise), object "
        "representation fragments it into meaningless tiny components. "
        "Region representation can model spatial continuity directly. "
        "This is the canonical failure mode for object representation."
    ),
    train_inputs=[
        grid([
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
        ]),
    ],
    best_representation="region",
    worst_representation="object",
    tags={"region", "gradient", "continuous"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: Spatial adjacency zone
# ─────────────────────────────────────────────────────────────────────

SPATIAL_ADJACENCY_ZONES = CanonicalTask(
    task_id="spatial_adjacency_zones",
    name="Spatial Adjacency Zones",
    category="region",
    failure_mode="adjacency_ambiguity",
    description=(
        "Three colored clusters with the task focused on the BOUNDARY between "
        "them. Which colors share the longest border? Region representation "
        "models adjacency directly. Object representation sees three unrelated "
        "objects."
    ),
    justification=(
        "Spatial adjacency between regions is a fundamentally spatial concept. "
        "While object representation can compute adjacency via pixel neighbors, "
        "this is an implicit use of topological/regional thinking layered on top "
        "of object primitives. Region representation has adjacency as a native "
        "primitive."
    ),
    train_inputs=[
        grid([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 2, 2, 2, 0],
            [0, 1, 1, 0, 2, 2, 2, 0],
            [0, 1, 1, 0, 2, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]),
    ],
    best_representation="region",
    worst_representation="pixel",
    also_acceptable=["object"],  # object rep can handle this via adjacency graph
    tags={"region", "adjacency", "boundary"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: Region filling (flood fill)
# ─────────────────────────────────────────────────────────────────────

REGION_FILLING = CanonicalTask(
    task_id="region_filling",
    name="Region Filling (Flood Fill)",
    category="region",
    failure_mode="interior_detection_failure",
    description=(
        "A bounded empty region surrounded by a colored border. The task is "
        "to fill the interior. Region representation naturally identifies "
        "'interior of closed boundary' as a spatial primitive."
    ),
    justification=(
        "Region filling requires identifying the interior of a closed curve. "
        "This is a spatial/geometric operation (point-in-polygon, flood fill). "
        "Region representation has spatial closure as a primitive. Object "
        "representation would need to discover the concept of 'inside'."
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
    best_representation="region",
    worst_representation="pixel",
    tags={"region", "filling", "interior", "boundary"},
)
