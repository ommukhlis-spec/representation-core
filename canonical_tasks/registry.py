"""
Task Registry — the complete catalog of canonical representation tasks.

This registry maps each representation to its "reason to exist":
a set of canonical tasks where it is the unambiguously best choice.

A new representation is ADMITTED to the zoo only if:
  1. It has at least one canonical task where it uniquely wins
  2. Its admission does not regress existing pass rates
"""

from typing import Dict, List

from .specification import CanonicalTask
from .tasks import (
    COUNT_DISCRETE_OBJECTS,
    COUNT_BY_COLOR,
    UNIFORM_OBJECT_SIZE,
    MANY_SMALL_OBJECTS,
    HORIZONTAL_REFLECTION,
    ROTATIONAL_SYMMETRY_4,
    TRANSLATIONAL_SYMMETRY,
    NO_SYMMETRY,
    SINGLE_HOLE,
    MULTIPLE_HOLES,
    CONNECTIVITY_CHECK,
    NESTED_CONTAINMENT,
    OVERLAPPING_REGIONS,
    CONTINUOUS_GRADIENT,
    SPATIAL_ADJACENCY_ZONES,
    REGION_FILLING,
    COLOR_ORDERING_RULE,
    COLOR_CONSERVATION,
    SEQUENTIAL_PATTERN,
    XOR_PATTERN,
)


# ─────────────────────────────────────────────────────────────────────
# Full suite
# ─────────────────────────────────────────────────────────────────────

ALL_TASKS: List[CanonicalTask] = [
    # ── Object representation canonical tasks ──
    COUNT_DISCRETE_OBJECTS,
    COUNT_BY_COLOR,
    UNIFORM_OBJECT_SIZE,
    MANY_SMALL_OBJECTS,

    # ── Symmetry representation canonical tasks ──
    HORIZONTAL_REFLECTION,
    ROTATIONAL_SYMMETRY_4,
    TRANSLATIONAL_SYMMETRY,
    NO_SYMMETRY,  # negative test: symmetry should LOSE

    # ── Topology representation canonical tasks ──
    SINGLE_HOLE,
    MULTIPLE_HOLES,
    CONNECTIVITY_CHECK,
    NESTED_CONTAINMENT,

    # ── Region representation canonical tasks ──
    OVERLAPPING_REGIONS,
    CONTINUOUS_GRADIENT,
    SPATIAL_ADJACENCY_ZONES,
    REGION_FILLING,

    # ── Constraint representation canonical tasks ──
    COLOR_ORDERING_RULE,
    COLOR_CONSERVATION,
    SEQUENTIAL_PATTERN,
    XOR_PATTERN,
]


# ─────────────────────────────────────────────────────────────────────
# Per-representation indexing
# ─────────────────────────────────────────────────────────────────────

def tasks_by_category() -> Dict[str, List[CanonicalTask]]:
    """Group tasks by category."""
    groups: Dict[str, List[CanonicalTask]] = {}
    for task in ALL_TASKS:
        if task.category not in groups:
            groups[task.category] = []
        groups[task.category].append(task)
    return groups


def tasks_for_representation(rep_name: str) -> List[CanonicalTask]:
    """Get all tasks where rep_name is the declared best representation."""
    return [t for t in ALL_TASKS if t.best_representation == rep_name]


def reason_to_exist() -> Dict[str, List[str]]:
    """
    Map each representation to its canonical tasks.

    This answers: "Why does this representation exist?"

    Returns:
        {rep_name: [task_id_1, task_id_2, ...]}
    """
    reasons: Dict[str, List[str]] = {}
    for task in ALL_TASKS:
        rep = task.best_representation
        if rep not in reasons:
            reasons[rep] = []
        reasons[rep].append(task.task_id)
    return reasons


def failure_mode_mapping() -> List[Dict[str, str]]:
    """
    The explicit Failure Mode → Task → Representation mapping.

    This is the scientific justification for every representation:
    each row answers "what failure mode does this representation solve?"

    Returns a list of dicts with keys:
      failure_mode, task_id, task_name, required_representation
    """
    rows = []
    for task in ALL_TASKS:
        rows.append({
            "failure_mode": task.failure_mode,
            "task_id": task.task_id,
            "task_name": task.name,
            "required_representation": task.best_representation,
        })
    # Sort by failure mode
    rows.sort(key=lambda r: r["failure_mode"])
    return rows


def print_failure_mode_table():
    """Print the Failure Mode → Task → Representation table as markdown."""
    rows = failure_mode_mapping()
    print("| Failure Mode | Canonical Task | Required Representation |")
    print("|-------------|---------------|------------------------|")
    for row in rows:
        print(f"| {row['failure_mode']} | {row['task_name']} | **{row['required_representation']}** |")


def export_tasks_json(output_dir: str):
    """
    Export all canonical tasks as JSON files.

    This transforms the task suite from Python code into a shareable
    dataset format — analogous to ImageNet, GLUE, or MMLU, but for
    representation benchmarking.

    Each task gets one JSON file with:
      - metadata (task_id, name, category, failure_mode, justification)
      - ground_truth (best_representation, worst_representation)
      - grids (train_inputs, train_outputs as 2D arrays)
    """
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Export individual task files
    for task in ALL_TASKS:
        task_dict = {
            "task_id": task.task_id,
            "name": task.name,
            "category": task.category,
            "failure_mode": task.failure_mode,
            "description": task.description,
            "justification": task.justification,
            "best_representation": task.best_representation,
            "worst_representation": task.worst_representation,
            "also_acceptable": task.also_acceptable,
            "tags": sorted(task.tags),
            "train_inputs": [obs.grid.tolist() for obs in task.train_inputs],
            "train_outputs": [obs.grid.tolist() for obs in task.train_outputs],
        }
        fpath = os.path.join(output_dir, f"{task.task_id}.json")
        with open(fpath, "w") as f:
            json.dump(task_dict, f, indent=2)

    # Export manifest
    manifest = {
        "total_tasks": len(ALL_TASKS),
        "categories": list(tasks_by_category().keys()),
        "representations": list(reason_to_exist().keys()),
        "failure_modes": [row["failure_mode"] for row in failure_mode_mapping()],
        "tasks": [t.task_id for t in ALL_TASKS],
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported {len(ALL_TASKS)} tasks to {output_dir}/")
    return output_dir


def print_registry():
    """Print a human-readable summary of the registry."""
    reasons = reason_to_exist()
    total = len(ALL_TASKS)
    print(f"Canonical Task Registry: {total} tasks across {len(reasons)} representations\n")

    for rep, task_ids in sorted(reasons.items()):
        print(f"  [{rep}] — {len(task_ids)} canonical tasks:")
        for tid in task_ids:
            task = next(t for t in ALL_TASKS if t.task_id == tid)
            print(f"    • {tid}: {task.name}")
        print()

    print(f"Categories: {list(tasks_by_category().keys())}")
