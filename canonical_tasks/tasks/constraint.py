"""
Canonical tasks for CONSTRAINT-based reasoning.

Constraint representation is the right lens when:
  - The task is fundamentally about RULES, not objects or space
  - Regularities are relational (x is always above y)
  - The pattern is a logical formula, not a geometric pattern
  - Invariants are the primary content, not the substrate

Constraint representation encodes observations as sets of logical
constraints that must hold, rather than as structured spatial data.
"""

from ..specification import CanonicalTask, grid


# ─────────────────────────────────────────────────────────────────────
# Task: Relational constraint (color ordering)
# ─────────────────────────────────────────────────────────────────────

COLOR_ORDERING_RULE = CanonicalTask(
    task_id="color_ordering_rule",
    name="Color Ordering Rule (Red above Blue)",
    category="constraint",
    failure_mode="relational_rule_blindness",
    description=(
        "Every red pixel (2) is directly above a blue pixel (1). This is a "
        "pure relational constraint. Constraint representation encodes this "
        "as: ∀(r,c): grid[r,c]=2 ⇒ grid[r+1,c]=1. Object representation "
        "would see individual red and blue objects but miss the global rule."
    ),
    justification=(
        "The rule 'red is always above blue' is a logical constraint. "
        "Constraint representation makes this the primary content of the "
        "encoding. Object or pixel representation would need to discover "
        "this rule by searching over possible relational predicates — a "
        "search problem that constraint representation solves by design."
    ),
    train_inputs=[
        grid([
            [0, 2, 0, 0, 2, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 2, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]),
    ],
    best_representation="constraint",
    worst_representation="pixel",
    tags={"constraint", "relational", "rule"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: Conservation constraint
# ─────────────────────────────────────────────────────────────────────

COLOR_CONSERVATION = CanonicalTask(
    task_id="color_conservation",
    name="Color Conservation",
    category="constraint",
    failure_mode="conservation_law_blindness",
    description=(
        "The number of pixels of each color is conserved across input-output "
        "pairs, even though positions change. This is a global conservation "
        "law. Constraint representation encodes the invariant directly: "
        "∀c: count(input, c) = count(output, c)."
    ),
    justification=(
        "Conservation laws are natural primitives in constraint representation. "
        "They are expressed as invariants across transformations. Object or "
        "pixel representation would need to discover conservation as a "
        "coincidental regularity among many possible regularities."
    ),
    train_inputs=[
        grid([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ]),
    ],
    best_representation="constraint",
    worst_representation="pixel",
    tags={"constraint", "conservation", "invariant"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: Sequential pattern
# ─────────────────────────────────────────────────────────────────────

SEQUENTIAL_PATTERN = CanonicalTask(
    task_id="sequential_pattern",
    name="Sequential Pattern (Arithmetic)",
    category="constraint",
    failure_mode="sequence_pattern_blindness",
    description=(
        "A sequence of rows where each row contains the next number in an "
        "arithmetic progression. The constraint is sequential and numerical. "
        "Neither object nor symmetry representation captures this well."
    ),
    justification=(
        "Sequential/numerical patterns are constraint-like: they express "
        "relationships between positions in a sequence. Constraint "
        "representation can encode 'row[i] = row[i-1] + 1' directly. "
        "Object or symmetry representation has no primitive for sequential "
        "or arithmetic relationships."
    ),
    train_inputs=[
        grid([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4],
            [5, 5, 5, 5],
        ]),
    ],
    best_representation="constraint",
    worst_representation="pixel",
    also_acceptable=["symmetry"],  # could be seen as translational
    tags={"constraint", "sequence", "arithmetic"},
)

# ─────────────────────────────────────────────────────────────────────
# Task: XOR / exclusion constraint
# ─────────────────────────────────────────────────────────────────────

XOR_PATTERN = CanonicalTask(
    task_id="xor_pattern",
    name="XOR / Mutual Exclusion Pattern",
    category="constraint",
    failure_mode="logical_constraint_blindness",
    description=(
        "For every 2×2 block in the grid, exactly two diagonal cells are "
        "filled (checkerboard-like). This is a local logical constraint. "
        "Constraint representation expresses it as: "
        "∀ 2×2 blocks: (tl ∧ br) ⊕ (tr ∧ bl)."
    ),
    justification=(
        "Boolean/logical constraints are the native language of constraint "
        "representation. This XOR pattern is easy to express as a logical "
        "formula but requires discovering the 2×2 block structure and the "
        "XOR relationship in other representations."
    ),
    train_inputs=[
        grid([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ]),
    ],
    best_representation="constraint",
    worst_representation="pixel",
    also_acceptable=["symmetry"],  # also a checkerboard symmetry
    tags={"constraint", "xor", "logical", "boolean"},
)
